"""An uploaded clip, turned into checked frames, after the response has gone.

This is the module that exists because the version without it took the whole
product down. `POST /api/photo-sets/{id}/video` read the upload into memory,
decoded sixty frames, and ran face detection, sharpness analysis and an object
store upload on every one of them - all inside the request handler, on the
event loop. One ordinary 31-second 1080p clip therefore stopped every other
request in the process, failed the platform health check on the same port, and
returned 502 to the customer who uploaded it and to everybody else on the site.

Four decisions, each of them the reason a specific failure cannot recur.

*The request does not do the work.* It stores the bytes, writes a row, and
returns a job id. Nothing that takes seconds happens before the response, so
the endpoint's own latency no longer depends on the length of the video.

*Nothing CPU-bound touches the event loop, including in here.* Every decode
and every per-frame check goes through asyncio.to_thread, and the request
handler does not await any of it - it awaits the row being written and nothing
else. Moving the stall from the handler into a thread the handler then waits
on would have been the same outage with more code. OpenCV releases the
interpreter lock across decode, detection and encode, so a worker thread doing
this work genuinely leaves the loop free rather than merely appearing to.

*Progress is counted, never estimated.* The frame total is read from container
metadata before any decoding, so the bar has a denominator from the first
poll, and the numerator is written after each frame. A family watching this
sees "40 of 60 frames, 31 usable", which is the thing they said was missing.

*A job that stops reporting is failed, not left running.* Same posture as
avatar/splat/service.py, and for the same reason: the tasks live in memory and
the rows live in the database, so a restart loses the tasks and keeps the rows,
and a row with no task behind it past the point the work would have finished
is reported as lost rather than left claiming to be forty percent done.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.models import PhotoSetStatus, TrainingStatus, VideoIngestJob
from avatar.gateway.tenancy import TenantError
from avatar.ingest.service import add_photo, get_photo_set
from avatar.ingest.video import FrameReader, VideoError, plan_frames
from avatar.storage.base import BlobStore
from avatar.storage.keys import source_clip_key

# How long the parts either side of the frame loop are allowed to take: the
# metadata read, and pushing the original clip to object storage. The store is
# across a network in deployment, and a 75MB upload on a slow link is minutes.
CLIP_SETUP_TIMEOUT_S = 300.0

# How long one frame may take: seek, decode, JPEG encode, face detection,
# sharpness, and a store round trip. Generous on purpose - this number only
# decides when a job is declared lost, and declaring a live job dead is worse
# than declaring a dead one late.
PER_FRAME_TIMEOUT_S = 8.0

# How long past its own deadline a job may sit before it is failed. Reaching
# this means the process running it went away without writing an outcome,
# which is the case the row exists to catch.
ABANDONED_SLACK_S = 60.0

# Visible on the first poll, so a chosen file clearly landed. The clip is
# being read at this point and there is no frame count yet.
MIN_PROGRESS = 0.02

# The bar does not reach the end before the job says it finished. The last
# frame is not the last of the work: the row still has to be written.
MAX_RUNNING_PROGRESS = 0.99


SessionSource = Callable[[], AsyncIterator[AsyncSession]]


class VideoBusy(RuntimeError):
    """A clip is already being processed into this photo set.

    Two at once would interleave frame numbering into one set and race on the
    stored source clip, and the second upload would silently win.
    """


@dataclass(frozen=True)
class StartedIngest:
    """A clip now being processed, and what to poll for."""

    job_id: str
    photo_set_id: str


class VideoIngestService:
    """Turns uploaded clips into checked frames, and reports on the doing.

    Holds the running jobs in memory and their outcomes in the database. That
    split is what makes a lost job detectable: see _expire_if_lost.
    """

    def __init__(self, store: BlobStore, sessions: SessionSource):
        self._store = store
        self._sessions = sessions
        self._jobs: dict[str, asyncio.Task] = {}

    @property
    def running(self) -> tuple[str, ...]:
        """Job ids this process is currently processing. Empty between jobs."""
        return tuple(self._jobs)

    @asynccontextmanager
    async def _own_session(self) -> AsyncIterator[AsyncSession]:
        """A session for work that outlives the request that started it.

        The request's own session is closed the moment the response is sent,
        and this job writes sixty rows afterwards.
        """
        source = self._sessions()
        session = await source.__anext__()
        try:
            yield session
        finally:
            await source.aclose()

    async def start(
        self,
        db: AsyncSession,
        photo_set_id: str,
        owner_id: str,
        *,
        clip_path: Path,
        filename: str,
        content_type: str,
        size_bytes: int,
    ) -> StartedIngest:
        """Record the job and begin. Returns as soon as the row is committed.

        Takes a path rather than bytes. The clip is already on disk by the
        time this is called and reading it back here would put the whole
        upload in memory inside the request, which is the half of the original
        bug that showed up as the machine disappearing rather than as a stall.

        Raises TenantError for a set that is not this owner's, ValueError for
        one that no longer accepts uploads, and VideoBusy when a clip is
        already being processed into it. Every one of them leaves no row and
        no task, so a refusal costs nothing and explains itself.
        """
        photo_set = await get_photo_set(db, photo_set_id, owner_id)

        # The same rule add_photo applies, checked before a job exists rather
        # than sixty times inside one. A set that has passed validation is not
        # open to more material; finding that out per frame would leave a job
        # that ran to completion having added nothing.
        if photo_set.status not in (PhotoSetStatus.UPLOADING, PhotoSetStatus.REJECTED):
            raise ValueError(
                f"photo set is {photo_set.status.value} and no longer accepts uploads"
            )

        existing = (
            await db.execute(
                select(VideoIngestJob).where(
                    VideoIngestJob.photo_set_id == photo_set_id,
                    VideoIngestJob.owner_id == owner_id,
                    VideoIngestJob.status.in_(
                        (TrainingStatus.QUEUED, TrainingStatus.RUNNING)
                    ),
                )
            )
        ).scalars().first()
        if existing is not None:
            raise VideoBusy(
                f"a clip is already being read into this photo set (job {existing.id})"
            )

        job = VideoIngestJob(
            owner_id=owner_id,
            photo_set_id=photo_set_id,
            status=TrainingStatus.RUNNING,
            filename=filename,
            size_bytes=size_bytes,
            started_at=datetime.now(UTC),
        )
        db.add(job)
        await db.commit()

        task = asyncio.create_task(
            self._run(job.id, owner_id, photo_set_id, clip_path, content_type)
        )
        self._jobs[job.id] = task
        task.add_done_callback(lambda _t, jid=job.id: self._jobs.pop(jid, None))

        logger.info(
            f"video job {job.id} started for photo set {photo_set_id} "
            f"({size_bytes // 1024}KB from {filename!r})"
        )
        return StartedIngest(job_id=job.id, photo_set_id=photo_set_id)

    async def _run(
        self,
        job_id: str,
        owner_id: str,
        photo_set_id: str,
        clip_path: Path,
        content_type: str,
    ) -> None:
        """The processing itself. Every exit writes something a customer reads."""
        started = time.monotonic()
        try:
            async with self._own_session() as db:
                examined, usable, planned = await self._process(
                    db, job_id, owner_id, photo_set_id, clip_path, content_type
                )
        except asyncio.CancelledError:
            # The row was marked cancelled by whoever asked for it, before the
            # task was touched. Writing it again from a cancelled task is how
            # a cancel ends up hanging on a database call.
            raise
        except VideoError as exc:
            # The clip itself is the problem, and the customer can act on it:
            # the wrong file, a container nothing can decode, a clip longer
            # than the ceiling. The words are the validator's, not a generic
            # apology.
            await self._record_failure(job_id, str(exc))
            return
        except BaseException as exc:  # noqa: BLE001 - every failure must leave a row
            await self._record_failure(job_id, str(exc) or exc.__class__.__name__)
            return
        finally:
            # The spooled upload is ours and nobody else's; it goes whether the
            # job worked, failed, or was cancelled mid-frame.
            with contextlib.suppress(OSError):
                clip_path.unlink(missing_ok=True)

        await self._record_success(job_id, examined, usable)
        logger.info(
            f"video job {job_id}: {usable} usable of {examined} frames checked "
            f"({planned} planned) in {time.monotonic() - started:.1f}s"
        )

    async def _process(
        self,
        db: AsyncSession,
        job_id: str,
        owner_id: str,
        photo_set_id: str,
        clip_path: Path,
        content_type: str,
    ) -> tuple[int, int, int]:
        """Read the clip, keep it, and check every frame it yields."""
        # Metadata only, but it still opens a decoder, so it goes off the loop
        # like everything else here.
        plan = await asyncio.to_thread(plan_frames, clip_path)
        await self._write_plan(db, job_id, len(plan))

        # The clip is kept, not just sampled. It is the base the lip-sync
        # renderer drives, and the head moves in the result because the head
        # moved in the recording.
        #
        # Read and pushed before the frames so that a job interrupted halfway
        # through the frame loop still leaves the footage the likeness build
        # can use. `data` goes out of scope immediately; it is the only point
        # at which the whole upload is in memory, and it is here rather than
        # in the request handler on purpose.
        data = await asyncio.to_thread(clip_path.read_bytes)
        await self._store.put(
            owner_id, source_clip_key(owner_id, photo_set_id), data, content_type
        )
        del data

        reader = FrameReader(clip_path)
        await asyncio.to_thread(reader.open)
        examined = 0
        usable = 0
        try:
            for index, seconds in enumerate(plan.offsets):
                frame = await asyncio.to_thread(reader.read_at, seconds)
                if frame is None:
                    continue
                try:
                    photo = await add_photo(
                        db, self._store, photo_set_id, owner_id,
                        f"frame-{index:04d}.jpg", "image/jpeg", frame,
                    )
                except ValueError:
                    # One unusable frame is not a failed upload. A clip is
                    # expected to contain blinks, blur and turns away from
                    # camera; rejecting the whole video for them would reject
                    # every real video.
                    continue
                examined += 1
                usable += 1 if photo.accepted else 0
                # Written per frame rather than at the end. A count that only
                # appears once the work is done is not progress.
                await self._write_counts(db, job_id, examined, usable)
        finally:
            await asyncio.to_thread(reader.close)

        if examined == 0:
            raise VideoError("no frame of that video could be decoded")
        return examined, usable, len(plan)

    async def _write_plan(self, db: AsyncSession, job_id: str, planned: int) -> None:
        job = await db.get(VideoIngestJob, job_id)
        if job is None:
            return
        job.frames_planned = planned
        await db.commit()

    async def _write_counts(
        self, db: AsyncSession, job_id: str, examined: int, usable: int
    ) -> None:
        job = await db.get(VideoIngestJob, job_id)
        if job is None:
            return
        job.frames_examined = examined
        job.frames_usable = usable
        await db.commit()

    async def _record_failure(self, job_id: str, message: str) -> None:
        async with self._own_session() as db:
            job = await db.get(VideoIngestJob, job_id)
            if job is None:
                return
            job.status = TrainingStatus.FAILED
            job.finished_at = datetime.now(UTC)
            # What actually went wrong, not a generic apology. A family told
            # "something went wrong" has been told nothing, and support less.
            job.error = message
            await db.commit()
        logger.error(f"video job {job_id} failed: {message}")

    async def _record_success(self, job_id: str, examined: int, usable: int) -> None:
        async with self._own_session() as db:
            job = await db.get(VideoIngestJob, job_id)
            if job is None:
                return
            job.status = TrainingStatus.SUCCEEDED
            job.finished_at = datetime.now(UTC)
            job.frames_examined = examined
            job.frames_usable = usable
            job.error = None
            await db.commit()

    async def read(self, db: AsyncSession, job_id: str, owner_id: str) -> dict:
        """What to tell the customer about this clip, at any point in its life."""
        job = await self._owned_job(db, job_id, owner_id)
        await self._expire_if_lost(db, job)

        return {
            "id": job.id,
            "photo_set_id": job.photo_set_id,
            "status": job.status.value,
            "filename": job.filename,
            "progress": self._progress(job),
            "frames_planned": job.frames_planned,
            "frames_examined": job.frames_examined,
            "frames_usable": job.frames_usable,
            "error": job.error,
        }

    async def cancel(self, db: AsyncSession, job_id: str, owner_id: str) -> dict:
        """Stop reading a clip. Safe on one that has already finished.

        The row is written before the task is touched, so the cancelled task
        has nothing left to record and cannot be interrupted while recording
        it.
        """
        job = await self._owned_job(db, job_id, owner_id)
        if job.status in (TrainingStatus.QUEUED, TrainingStatus.RUNNING):
            job.status = TrainingStatus.CANCELLED
            job.finished_at = datetime.now(UTC)
            job.error = "reading the clip was cancelled"
            await db.commit()

        task = self._jobs.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()

        return {"id": job.id, "status": job.status.value}

    async def _owned_job(
        self, db: AsyncSession, job_id: str, owner_id: str
    ) -> VideoIngestJob:
        """This tenant's job, or an indistinguishable refusal.

        Ownership is part of the query rather than a check on the result, so
        there is no window in which another family's row exists in memory.
        """
        job = (
            await db.execute(
                select(VideoIngestJob)
                .where(
                    VideoIngestJob.id == job_id,
                    VideoIngestJob.owner_id == owner_id,
                )
                # The job writes its counts from its own session, so a read
                # must take the database's answer rather than whatever this
                # one loaded earlier. Without this a status page keeps
                # reporting a finished job as still running.
                .execution_options(populate_existing=True)
            )
        ).scalar_one_or_none()
        if job is None:
            raise TenantError("no such video job")
        return job

    def _deadline_s(self, job: VideoIngestJob) -> float:
        """The longest this job could honestly still be working.

        Derived from the frame count rather than fixed, because a four-second
        clip and a two-minute one are an order of magnitude apart and one
        constant would either declare the short one dead while it ran or leave
        the long one hanging for minutes after its process died.
        """
        frames = job.frames_planned or 1
        return CLIP_SETUP_TIMEOUT_S + frames * PER_FRAME_TIMEOUT_S + ABANDONED_SLACK_S

    async def _expire_if_lost(self, db: AsyncSession, job: VideoIngestJob) -> None:
        """Fail a job whose runner is gone, rather than leave it "reading".

        A row still claiming to run, past the point at which the work would
        have finished, with no task behind it in this process, means the
        gateway restarted mid-job. Somebody must be told; the alternative is a
        bar frozen at forty percent and a family who keeps checking it.
        """
        if job.status not in (TrainingStatus.QUEUED, TrainingStatus.RUNNING):
            return
        if job.id in self._jobs:
            return
        if _elapsed_since(job.started_at or job.queued_at) < self._deadline_s(job):
            return

        job.status = TrainingStatus.FAILED
        job.finished_at = datetime.now(UTC)
        job.error = (
            "reading the clip stopped reporting before it finished, so it was "
            "not completed. Nothing was left running; upload it again."
        )
        await db.commit()
        logger.warning(f"video job {job.id} was abandoned and has been failed")

    def _progress(self, job: VideoIngestJob) -> float:
        """How far along, counted rather than guessed.

        Unlike a splat build, this job knows exactly how much work it has and
        how much of it is done, so nothing here is an estimate from elapsed
        time.
        """
        if job.status is TrainingStatus.SUCCEEDED:
            return 1.0
        if job.status in (TrainingStatus.FAILED, TrainingStatus.CANCELLED):
            return 0.0
        if job.frames_planned <= 0:
            # The clip is still being read; there is no denominator yet.
            return MIN_PROGRESS
        share = job.frames_examined / job.frames_planned
        return round(max(MIN_PROGRESS, min(MAX_RUNNING_PROGRESS, share)), 3)

    async def wait(self, job_id: str, timeout_s: float = 60.0) -> None:
        """Block until a job finishes. Used by tests, never by a handler."""
        task = self._jobs.get(job_id)
        if task is None:
            return
        await asyncio.wait_for(asyncio.shield(task), timeout_s)

    async def shutdown(self) -> None:
        """Stop every running job. A gateway going down must not leave one."""
        for job_id in list(self._jobs):
            task = self._jobs.pop(job_id, None)
            if task is not None and not task.done():
                task.cancel()


def _elapsed_since(moment: datetime | None) -> float:
    """Seconds since a stored timestamp, tolerating a naive one.

    SQLite hands back what it was given without a timezone, and subtracting a
    naive datetime from an aware one raises. That would turn a progress read
    into a 500 on exactly the deployment the development database uses.
    """
    if moment is None:
        return 0.0
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - aware).total_seconds())


__all__ = [
    "StartedIngest",
    "VideoBusy",
    "VideoIngestService",
]
