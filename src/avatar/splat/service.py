"""What the customer uploaded, turned into a splat of the person.

How a splat is built lives in avatar/splat/routes.py and avatar/splat/build.py
and is not touched here. This module is the part that faces the product: it
reads what a family actually managed to upload, hands it to the route selector
in the shape that selector demands, runs the build as a job with a row, and
keeps what must be said about the result attached to the avatar rather than to
a request that ended minutes earlier.

Four decisions are worth stating, because each of them is a place this could
have gone wrong quietly.

*Frame counts are counted, never assumed.* Intake refuses to accept a video
without them, deliberately, so that a thirty-second clip of a shoulder cannot
be reconstructed on the strength of its duration. The way to honour that is to
count the frames this set actually took from the clip and how many of them a
face was found in, which the ingest checks already recorded one row at a time.

*A build is a job, not a function call.* The same posture as identity
training: a row, a status, an error column. A family who closes the tab can be
told what happened, and a gateway that dies mid-build leaves a row that says so
rather than an avatar stuck forever in "building".

*Refusal costs nothing and leaves nothing.* The route is decided before a job
row exists, so a set that cannot produce a likeness never creates a build to
explain later. Deciding it twice - once here, once inside the builder - is
free, because deciding is pure.

*The disclosure is written down.* The quality report computes both the
plain-English sentence and the measured fraction behind it, and both are
stored on the avatar the moment the build finishes. A disclosure that exists
only in the response to the request that started the build is a disclosure
nobody reads.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.config import Settings
from avatar.gateway.models import Avatar, Photo, PhotoSet, TrainingJob, TrainingStatus
from avatar.gateway.tenancy import TenantError
from avatar.ingest.video import probe_duration
from avatar.splat.build import (
    FAKE_ITERATIONS_PER_SECOND,
    Quality,
    SplatBackend,
    SplatBuilder,
    SplatError,
    SplatRefused,
    SplatResult,
    plan,
)
from avatar.splat.routes import Intake, RouteDecision
from avatar.storage.base import BlobStore
from avatar.storage.keys import source_clip_key

# How a frame taken from the uploaded clip is named by the video endpoint.
# Matched rather than guessed at, because the count of these rows is the
# evidence that somebody looked at the video - which is exactly what Intake
# refuses to proceed without.
_FRAME_NAME = re.compile(r"^frame-\d{4}\.jpe?g$", re.IGNORECASE)

# Marks a build job row as a splat build rather than an identity-training run.
# Both share the table, because they are the same kind of thing - a long,
# expensive job whose customer has left the page - and a second table would
# have meant a second status page saying the same words.
PROVIDER_PREFIX = "splat"


def expected_seconds(quality: Quality) -> float:
    """Roughly how long a build of this quality takes.

    An estimate, and only ever used to move a progress bar. Neither backend
    emits an iteration count, so the alternative was a bar that sits at zero
    for five minutes, which every customer reads as "it has hung".

    The rate is the one build.py already states for the development backend
    and describes as roughly what a mid-range card optimises at, so the bar
    behaves the same way whichever backend is behind it.
    """
    return quality.iterations / FAKE_ITERATIONS_PER_SECOND


# A progress bar never reaches the end before the splat exists. The only
# honest 1.0 is an artefact written to storage.
MAX_ESTIMATED_PROGRESS = 0.95

# Visible the instant the button is pressed, so a click clearly landed.
MIN_ESTIMATED_PROGRESS = 0.02

# How long past its own timeout a build may sit before it is declared lost.
# Reaching this means the process running the build went away without writing
# a failure, which is the case the row exists to catch.
ABANDONED_SLACK_S = 60.0


class SplatUnavailable(SplatError):
    """The set cannot be built from for a reason that is not the material.

    Distinct from SplatRefused: a refusal names what the customer must upload,
    and this names something they cannot act on - an unattached photo set, a
    build already running.
    """


SessionSource = Callable[[], AsyncIterator[AsyncSession]]


def build_splat_builder(cfg: Settings, store: BlobStore) -> SplatBuilder:
    """Backend selection, in one place, exactly as training and storage do it."""
    if cfg.splat_backend == "fake":
        from avatar.splat.build import FakeSplatBackend

        backend: SplatBackend = FakeSplatBackend(store)
    elif cfg.splat_backend == "runpod":
        from avatar.splat.build import RunPodSplatBackend

        backend = RunPodSplatBackend(cfg.runpod_api_key, cfg.runpod_endpoint_id)
    else:
        raise ValueError(
            f"unknown splat_backend {cfg.splat_backend!r}; expected 'fake' or 'runpod'"
        )
    return SplatBuilder(backend)


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


async def _read_clip(store: BlobStore, owner_id: str, photo_set_id: str) -> bytes | None:
    """The customer's own footage, or None when they uploaded photographs."""
    try:
        return await store.get(owner_id, source_clip_key(owner_id, photo_set_id))
    except Exception:  # noqa: BLE001 - no clip is the ordinary case, not an error
        return None


async def intake_for(
    db: AsyncSession, store: BlobStore, photo_set: PhotoSet, owner_id: str
) -> Intake:
    """What this family uploaded, in the shape the route selector reads.

    Only accepted images are counted. An image the ingest checks rejected is
    not evidence of a face, and counting it here would let a set refused at
    upload be built from anyway.

    Frames taken from a clip are counted as photographs as well as frames.
    A frame that passed the same sharpness, face and framing checks as an
    uploaded photograph is a photograph of that person by every measure this
    pipeline applies, and excluding it would refuse builds that can be made -
    a family whose only material is a six-second clip has usable stills.
    """
    photos = (
        (
            await db.execute(
                select(Photo)
                .where(Photo.photo_set_id == photo_set.id, Photo.owner_id == owner_id)
                .order_by(Photo.uploaded_at, Photo.filename)
            )
        )
        .scalars()
        .all()
    )

    frames = [p for p in photos if _FRAME_NAME.match(p.filename)]
    accepted = [p for p in photos if p.accepted]

    clip = await _read_clip(store, owner_id, photo_set.id)
    # ffprobe on a temporary file: off the event loop, because the gateway is
    # serving other families' requests while this one is being answered.
    seconds = await asyncio.to_thread(probe_duration, clip) if clip else 0.0

    # A clip with no frames examined is not offered to the router. Intake
    # refuses to default the counts on purpose, and inventing one here to get
    # past that would be the exact failure it exists to prevent: a video
    # nobody looked at, judged usable. Without frames the photographs decide,
    # and the refusal - if it comes - names the real counts.
    usable_clip = bool(clip) and bool(frames)
    if clip and not frames:
        logger.warning(
            f"photo set {photo_set.id} has a stored clip but no frames taken from it; "
            "the build will be decided on the photographs"
        )

    return Intake(
        tenant_id=owner_id,
        photo_set_id=photo_set.id,
        photo_keys=tuple(p.blob_key for p in accepted),
        video_key=source_clip_key(owner_id, photo_set.id) if usable_clip else None,
        video_seconds=seconds if usable_clip else 0.0,
        video_frames=len(frames) if usable_clip else 0,
        video_frames_with_face=sum(1 for p in frames if p.accepted) if usable_clip else 0,
        # The short edge is not recorded per image, and a guess here would end
        # up quoted in a quality report as a measurement. The ingest checks
        # already refused anything under 512px; the report simply does not
        # mention what it was not told.
        source_short_edge_px=0,
    )


@dataclass(frozen=True)
class Started:
    """A build that is now running, and why it is being built that way."""

    job_id: str
    avatar_id: str
    decision: RouteDecision


class SplatService:
    """Starts splat builds, reports on them, and leaves nothing running.

    Holds the running builds in memory and their outcomes in the database,
    which is the split that matters: a restart loses the tasks and keeps the
    rows, and a row whose task is gone is reported as a lost build rather than
    left claiming to be in progress.
    """

    def __init__(
        self,
        builder: SplatBuilder,
        store: BlobStore,
        sessions: SessionSource,
        *,
        quality: Quality = Quality.STANDARD,
    ):
        self._builder = builder
        self._store = store
        self._sessions = sessions
        self._quality = quality
        self._builds: dict[str, asyncio.Task] = {}

    @property
    def quality(self) -> Quality:
        return self._quality

    @property
    def running(self) -> tuple[str, ...]:
        """Job ids this process is currently building. Empty between builds."""
        return tuple(self._builds)

    @asynccontextmanager
    async def _own_session(self) -> AsyncIterator[AsyncSession]:
        """A session for work that outlives the request that started it.

        The request's own session is closed the moment the response is sent,
        and a build finishing twenty minutes later has to write its result
        somewhere.
        """
        source = self._sessions()
        session = await source.__anext__()
        try:
            yield session
        finally:
            await source.aclose()

    async def start(
        self, db: AsyncSession, photo_set_id: str, owner_id: str
    ) -> Started:
        """Decide the route, refuse if it cannot be built, otherwise build.

        Raises SplatRefused when the material is not enough - carrying the
        itemised list of what is missing - and SplatUnavailable when the set
        is not in a state a build can start from.
        """
        photo_set = (
            await db.execute(
                select(PhotoSet).where(
                    PhotoSet.id == photo_set_id, PhotoSet.owner_id == owner_id
                )
            )
        ).scalar_one_or_none()
        if photo_set is None:
            # Identical refusal for another tenant's set and for one that does
            # not exist. See gateway/tenancy.py.
            raise TenantError("no such photo set")

        if not photo_set.avatar_id:
            raise SplatUnavailable(
                "attach this photo set to an avatar before building a likeness"
            )

        running = (
            await db.execute(
                select(TrainingJob).where(
                    TrainingJob.photo_set_id == photo_set_id,
                    TrainingJob.owner_id == owner_id,
                    TrainingJob.provider.startswith(PROVIDER_PREFIX),
                    TrainingJob.status.in_(
                        (TrainingStatus.QUEUED, TrainingStatus.RUNNING)
                    ),
                )
            )
        ).scalars().first()
        if running is not None:
            # A splat build is minutes of GPU on one person. Two at once for
            # the same set is the same likeness bought twice.
            raise SplatUnavailable(
                f"a build for this photo set is already running (job {running.id})"
            )

        intake = await intake_for(db, self._store, photo_set, owner_id)

        # Planned here as well as inside the builder, so a refusal never
        # creates a job row somebody has to explain, and a key that has
        # somehow escaped its tenant prefix is caught before a GPU is asked
        # for anything. plan() is pure, so deciding twice costs nothing but
        # the reading.
        decision = plan(intake, photo_set.avatar_id, quality=self._quality).decision

        job = TrainingJob(
            owner_id=owner_id,
            photo_set_id=photo_set_id,
            status=TrainingStatus.RUNNING,
            provider=f"{PROVIDER_PREFIX}:{self._builder.backend_name}",
            started_at=datetime.now(UTC),
        )
        db.add(job)
        await db.commit()

        avatar_id = photo_set.avatar_id
        task = asyncio.create_task(self._run(job.id, intake, avatar_id))
        self._builds[job.id] = task
        task.add_done_callback(lambda _t, jid=job.id: self._builds.pop(jid, None))

        logger.info(
            f"splat build {job.id} started for avatar {avatar_id} via "
            f"{decision.route.value} on {self._builder.backend_name}"
        )
        return Started(job_id=job.id, avatar_id=avatar_id, decision=decision)

    async def _run(self, job_id: str, intake: Intake, avatar_id: str) -> None:
        """The build itself. Every exit writes something a customer can read."""
        try:
            result = await self._builder.build(intake, avatar_id, quality=self._quality)
        except asyncio.CancelledError:
            # The row was marked cancelled by whoever asked for it, before the
            # task was touched. Writing it again from a cancelled task is how
            # a cancel ends up hanging on a database call.
            raise
        except BaseException as exc:  # noqa: BLE001 - every failure must leave a row
            await self._record_failure(job_id, exc)
            return
        await self._record_success(job_id, avatar_id, result)

    async def _record_failure(self, job_id: str, exc: BaseException) -> None:
        async with self._own_session() as db:
            job = await db.get(TrainingJob, job_id)
            if job is None:
                return
            job.status = TrainingStatus.FAILED
            job.finished_at = datetime.now(UTC)
            # The exception text, not a generic apology. A family who is told
            # "something went wrong" has been told nothing, and support has
            # been told less.
            job.error = str(exc) or exc.__class__.__name__
            await db.commit()
        logger.error(f"splat build {job_id} failed: {exc}")

    async def _record_success(
        self, job_id: str, avatar_id: str, result: SplatResult
    ) -> None:
        async with self._own_session() as db:
            job = await db.get(TrainingJob, job_id)
            if job is None:
                return
            job.status = TrainingStatus.SUCCEEDED
            job.finished_at = datetime.now(UTC)
            job.output_key = result.splat_key
            job.error = None

            avatar = await db.get(Avatar, avatar_id)
            if avatar is not None:
                # Written together, in one commit. The splat and the sentence
                # that discloses how much of it was invented must never exist
                # apart: an avatar with a likeness and no disclosure is the
                # exact state this product cannot ship.
                avatar.splat_key = result.splat_key
                avatar.splat_route = result.route.value
                avatar.splat_reasoning = result.reasoning
                avatar.splat_disclosure = result.report.disclosure
                avatar.splat_measured_fraction = result.report.measured_fraction
                avatar.splat_concerns = json.dumps(list(result.warnings))
                avatar.splat_gaussians = result.gaussian_count
                avatar.splat_size_bytes = result.size_bytes
                avatar.splat_backend = result.backend
                avatar.splat_built_at = datetime.now(UTC)
            await db.commit()

        logger.info(
            f"splat build {job_id}: {result.gaussian_count} gaussians, "
            f"{result.size_bytes // 1024}KB, {result.report.measured_fraction:.0%} "
            f"measured, ~${result.cost_usd:.4f}"
        )

    async def read(self, db: AsyncSession, job_id: str, owner_id: str) -> dict:
        """What to tell the customer about this build, at any point in its life."""
        job = await self._owned_job(db, job_id, owner_id)
        await self._expire_if_lost(db, job)

        avatar_id: str | None = None
        photo_set = await db.get(PhotoSet, job.photo_set_id)
        if photo_set is not None:
            avatar_id = photo_set.avatar_id

        avatar = (
            await db.get(Avatar, avatar_id, populate_existing=True) if avatar_id else None
        )
        built = job.status is TrainingStatus.SUCCEEDED and avatar is not None

        payload: dict = {
            "id": job.id,
            "status": job.status.value,
            "backend": job.provider.removeprefix(f"{PROVIDER_PREFIX}:"),
            "progress": self._progress(job),
            "error": job.error,
            "avatar_id": avatar_id,
            "splat_key": job.output_key,
            # Present on every payload, so a caller cannot read a finished
            # build without also having been handed what to say about it.
            "route": avatar.splat_route if built else None,
            "reasoning": avatar.splat_reasoning if built else None,
            "disclosure": avatar.splat_disclosure if built else None,
            "measured_fraction": avatar.splat_measured_fraction if built else None,
            "generated_fraction": (
                round(1.0 - avatar.splat_measured_fraction, 2)
                if built and avatar.splat_measured_fraction is not None
                else None
            ),
            "concerns": json.loads(avatar.splat_concerns)
            if built and avatar.splat_concerns
            else [],
            "gaussians": avatar.splat_gaussians if built else 0,
            "size_bytes": avatar.splat_size_bytes if built else 0,
        }
        return payload

    async def cancel(self, db: AsyncSession, job_id: str, owner_id: str) -> dict:
        """Stop a build. Safe on one that has already finished.

        The row is written before the task is touched, so that the cancelled
        task has nothing left to record and cannot be interrupted while
        recording it.
        """
        job = await self._owned_job(db, job_id, owner_id)
        if job.status in (TrainingStatus.QUEUED, TrainingStatus.RUNNING):
            job.status = TrainingStatus.CANCELLED
            job.finished_at = datetime.now(UTC)
            job.error = "the build was cancelled"
            await db.commit()

        task = self._builds.pop(job_id, None)
        if task is not None and not task.done():
            # SplatBuilder cancels the backend job on its way out of any
            # exception, cancellation included, so nothing is left optimising
            # on a GPU nobody is watching.
            task.cancel()

        return {"id": job.id, "status": job.status.value}

    async def _owned_job(
        self, db: AsyncSession, job_id: str, owner_id: str
    ) -> TrainingJob:
        """This tenant's splat build, or an indistinguishable refusal.

        Ownership is part of the query rather than a check on the result, so
        there is no window in which another family's row exists in memory.
        """
        job = (
            await db.execute(
                select(TrainingJob)
                .where(
                    TrainingJob.id == job_id,
                    TrainingJob.owner_id == owner_id,
                    TrainingJob.provider.startswith(PROVIDER_PREFIX),
                )
                # The build writes its outcome from its own session, so a read
                # must take the database's answer rather than whatever this one
                # loaded earlier. Without this a status page can keep reporting
                # a finished build as still running.
                .execution_options(populate_existing=True)
            )
        ).scalar_one_or_none()
        if job is None:
            raise TenantError("no such build")
        return job

    async def _expire_if_lost(self, db: AsyncSession, job: TrainingJob) -> None:
        """Fail a build whose runner is gone, rather than leave it "building".

        A row still claiming to run, past the point at which the build would
        have timed out itself, with no task behind it in this process, means
        the gateway restarted mid-build. Somebody must be told; the alternative
        is a progress bar that never finishes and a family who keeps checking.
        """
        if job.status not in (TrainingStatus.QUEUED, TrainingStatus.RUNNING):
            return
        if job.id in self._builds:
            return
        if _elapsed_since(job.started_at or job.queued_at) < (
            self._quality.wait_s + ABANDONED_SLACK_S
        ):
            return

        job.status = TrainingStatus.FAILED
        job.finished_at = datetime.now(UTC)
        job.error = (
            "the build stopped reporting before it finished, so it was not "
            "completed. Nothing was left running; start it again."
        )
        await db.commit()
        logger.warning(f"splat build {job.id} was abandoned and has been failed")

    def _progress(self, job: TrainingJob) -> float:
        """How far along, as far as anyone can honestly say.

        Estimated from elapsed time because neither backend reports iteration
        counts, and capped below the end until the artefact exists.
        """
        if job.status is TrainingStatus.SUCCEEDED:
            return 1.0
        if job.status in (TrainingStatus.FAILED, TrainingStatus.CANCELLED):
            return 0.0
        share = _elapsed_since(job.started_at or job.queued_at) / expected_seconds(
            self._quality
        )
        return round(
            max(MIN_ESTIMATED_PROGRESS, min(MAX_ESTIMATED_PROGRESS, share)), 3
        )

    async def wait(self, job_id: str, timeout_s: float = 30.0) -> None:
        """Block until a build finishes. Used by tests, never by a handler."""
        task = self._builds.get(job_id)
        if task is None:
            return
        await asyncio.wait_for(asyncio.shield(task), timeout_s)

    async def shutdown(self) -> None:
        """Stop every running build. A gateway going down must not leave one."""
        for job_id in list(self._builds):
            task = self._builds.pop(job_id, None)
            if task is not None and not task.done():
                task.cancel()


__all__ = [
    "PROVIDER_PREFIX",
    "SplatRefused",
    "SplatService",
    "SplatUnavailable",
    "Started",
    "build_splat_builder",
    "expected_seconds",
    "intake_for",
]
