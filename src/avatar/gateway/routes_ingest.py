"""Upload and training endpoints.

Every handler resolves the signed-in tenant first and passes it into the
service layer, which builds every storage key from it. There is no path here
that takes a tenant id from the request body.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.csrf import require_same_site_header
from avatar.gateway.models import Photo, PhotoSet, PhotoSetStatus, TrainingJob, TrainingStatus
from avatar.gateway.tenancy import TenantError
from avatar.ingest.finalise import FinaliseError, finalise_avatar
from avatar.ingest.service import (
    add_photo,
    create_photo_set,
    delete_photo_set,
    describe,
    evaluate_set,
    get_photo_set,
    revalidate_set,
)
from avatar.ingest.validate import (
    MIN_FOR_HALF_BODY,
    MIN_USABLE,
    RECOMMENDED_MAX,
    RECOMMENDED_MIN,
)
from avatar.ingest.video import is_video
from avatar.ingest.video_service import VideoBusy, VideoIngestService
from avatar.training.base import JobState, TrainingRequest

# The largest clip that will be accepted at all.
#
# Two minutes is already the ceiling on useful length (ingest/video.py), and
# two minutes of 1080p from a phone is roughly 300MB. This sits just under
# that, so an ordinary recording passes and a 4K clip is refused in a sentence
# rather than filling the volume or being killed halfway through.
#
# It is enforced while the upload streams, not after: the point of the limit
# is to stop before the bytes are on disk, and a check on a completed file has
# already paid the cost it was meant to avoid.
MAX_VIDEO_BYTES = 256 * 1024 * 1024

# How much of an upload is in memory at once on its way to disk. Everything
# above this is the operating system's problem rather than the process's.
UPLOAD_CHUNK_BYTES = 1024 * 1024

# Spooled uploads older than this are somebody's crashed job, not work in
# progress. A two-minute clip is processed in minutes; an hour is not a job.
SPOOL_STALE_S = 3600.0


class UploadTooLarge(ValueError):
    pass


def spool_dir(settings) -> Path:
    """Where a clip waits between arriving and being read.

    On the deployment this is the mounted volume rather than container-local
    disk, which matters twice: the root filesystem is small, and a job that
    outlives a deploy would otherwise lose the file it was working on.
    """
    return Path(settings.assets_dir) / "uploads"


def sweep_spool(settings, *, older_than_s: float = SPOOL_STALE_S) -> int:
    """Delete abandoned uploads. Returns how many went.

    A process killed mid-job leaves its spooled clip behind, and 256MB of
    those accumulate into a full volume with no error pointing at the cause.
    Called at startup, which is exactly when a crash has just happened.
    """
    directory = spool_dir(settings)
    if not directory.is_dir():
        return 0
    removed = 0
    cutoff = time.time() - older_than_s
    for path in directory.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        # One unreadable file must not stop the sweep.
        except OSError:
            continue
    if removed:
        logger.info(f"swept {removed} abandoned upload(s) from {directory}")
    return removed


async def _spool_upload(file: UploadFile, directory: Path) -> tuple[Path, int]:
    """Stream the upload to disk without ever holding it in memory.

    The whole reason this is not `await file.read()`. A 72MB clip read into a
    bytes object, on a 4GB machine that also holds a speech model, is a real
    out-of-memory risk - and an out-of-memory kill presents as the machine
    vanishing, with nothing in the log to say why.

    Starlette has already spooled the multipart body to its own temporary
    file, so this is a disk-to-disk copy in 1MB pieces. The writes go through
    a thread because the event loop must not do file I/O either.
    """
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed in the finally below
        dir=directory, suffix=".upload", delete=False
    )
    path = Path(handle.name)
    size = 0
    try:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                raise UploadTooLarge(
                    f"that clip is larger than {MAX_VIDEO_BYTES // (1024 * 1024)}MB"
                )
            await asyncio.to_thread(handle.write, chunk)
    except BaseException:
        handle.close()
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        raise
    handle.close()
    return path, size


_STATE_TO_STATUS = {
    JobState.QUEUED: TrainingStatus.QUEUED,
    JobState.RUNNING: TrainingStatus.RUNNING,
    JobState.SUCCEEDED: TrainingStatus.SUCCEEDED,
    JobState.FAILED: TrainingStatus.FAILED,
    JobState.CANCELLED: TrainingStatus.CANCELLED,
}


def build_router(
    settings, current_user, get_db, store, runner, video_jobs: VideoIngestService
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/photo-sets/requirements")
    async def requirements():
        """What the upload page tells people before they start.

        Served rather than duplicated in the frontend, so the numbers the
        customer sees and the numbers enforced at upload cannot drift apart.
        """
        return {
            "minimum": MIN_USABLE,
            "recommended_min": RECOMMENDED_MIN,
            "recommended_max": RECOMMENDED_MAX,
            "half_body_threshold": MIN_FOR_HALF_BODY,
            "shots": [
                {"label": "Head-on, neutral, looking at the camera", "count": "4-5"},
                {"label": "Three-quarter turn, both sides", "count": "4-6"},
                {"label": "Profile, both sides", "count": "2"},
                {"label": "Talking or smiling, mouth open", "count": "4-6"},
                {
                    "label": "Chest and shoulders in frame (optional)",
                    "count": "3+",
                },
            ],
            "rules": [
                "A phone photograph is fine; anything above 512 pixels works",
                "The face in focus - a blurred background is not a problem",
                "One person per photograph",
                "No sunglasses or heavy shadow across the face",
                "Different days, outfits and lighting where possible",
            ],
            "note": (
                "Use whatever photographs exist. With three or more showing the "
                "chest, the avatar includes the torso; otherwise it is framed at "
                "head and shoulders."
            ),
        }

    @router.post(
        "/api/photo-sets", status_code=201, dependencies=[Depends(require_same_site_header)]
    )
    async def create(
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        photo_set = await create_photo_set(db, user_id)
        return {"id": photo_set.id, "status": photo_set.status.value}

    @router.get("/api/photo-sets/{photo_set_id}")
    async def read(
        photo_set_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        try:
            photo_set = await get_photo_set(db, photo_set_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        photos = (
            (await db.execute(select(Photo).where(Photo.photo_set_id == photo_set_id)))
            .scalars()
            .all()
        )
        return describe(photo_set, list(photos))

    @router.post("/api/photo-sets/{photo_set_id}/photos", status_code=201)
    async def upload(
        photo_set_id: str,
        file: UploadFile = File(...),  # noqa: B008
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        data = await file.read()
        try:
            photo = await add_photo(
                db,
                store,
                photo_set_id,
                user_id,
                file.filename or "photo.jpg",
                file.content_type or "application/octet-stream",
                data,
            )
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "id": photo.id,
            "filename": photo.filename,
            "accepted": photo.accepted,
            "reasons": json.loads(photo.rejection_reasons) if photo.rejection_reasons else [],
            "half_body": 0.0 < photo.face_height_fraction < 0.33,
        }

    @router.post("/api/photo-sets/{photo_set_id}/video", status_code=202)
    async def upload_video(
        photo_set_id: str,
        file: UploadFile = File(...),  # noqa: B008
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        """Accept a clip and return a job to watch. Nothing else.

        This handler used to read the whole upload into memory, decode sixty
        frames, and run face detection, sharpness analysis and an object store
        upload on every one of them, synchronously, on the event loop. That
        stopped every other request in the process for the duration - long
        enough for the platform health check on this same port to fail and for
        everybody on the site, not only the person uploading, to get a 502.

        So the work is a job now. What happens here is bounded by the size of
        the upload rather than by the length of the video: the bytes go to
        disk, a row is written, and the id of that row comes back. Frames are
        taken in the background, and ingest/video_service.py reports on it.
        """
        # Decided from the declared type and the filename, before a single
        # byte is spooled. Writing 256MB to the volume and then refusing it is
        # work done for nothing.
        if not is_video(file.content_type or "", file.filename or ""):
            raise HTTPException(status_code=400, detail="that is not a video file")

        try:
            path, size = await _spool_upload(file, spool_dir(settings))
        except UploadTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

        try:
            started = await video_jobs.start(
                db,
                photo_set_id,
                user_id,
                clip_path=path,
                filename=file.filename or "clip.mp4",
                content_type=file.content_type or "video/mp4",
                size_bytes=size,
            )
        except BaseException as exc:
            # Nothing is going to read this file now. Refusals here are
            # ordinary - another tenant's set, a set that has already passed
            # validation, a clip already being read - and each of them must
            # leave the volume as it found it.
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            if isinstance(exc, TenantError):
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if isinstance(exc, VideoBusy):
                # Theirs and real; what is wrong is its state, which they can
                # act on but not by uploading anything else.
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if isinstance(exc, ValueError):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise

        return {
            "job_id": started.job_id,
            "photo_set_id": started.photo_set_id,
            "status": "running",
            "size_bytes": size,
        }

    @router.get("/api/video-jobs/{job_id}")
    async def video_job_status(
        job_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        """Frames taken so far, of how many, and how many were usable."""
        try:
            return await video_jobs.read(db, job_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/video-jobs/{job_id}/cancel")
    async def cancel_video_job(
        job_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        try:
            return await video_jobs.cancel(db, job_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post(
        "/api/photo-sets/{photo_set_id}/evaluate",
        dependencies=[Depends(require_same_site_header)],
    )
    async def evaluate(
        photo_set_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        try:
            photo_set = await evaluate_set(db, photo_set_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        photos = (
            (await db.execute(select(Photo).where(Photo.photo_set_id == photo_set_id)))
            .scalars()
            .all()
        )
        return describe(photo_set, list(photos))

    @router.post(
        "/api/photo-sets/{photo_set_id}/revalidate",
        dependencies=[Depends(require_same_site_header)],
    )
    async def revalidate(
        photo_set_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        """Re-check stored images against the current rules.

        Exists so a validator fix reaches photographs already uploaded, rather
        than asking a family to gather them again.
        """
        try:
            photo_set = await revalidate_set(db, store, photo_set_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        photos = (
            (await db.execute(select(Photo).where(Photo.photo_set_id == photo_set_id)))
            .scalars()
            .all()
        )
        return describe(photo_set, list(photos))

    @router.delete("/api/photo-sets/{photo_set_id}")
    async def remove(
        photo_set_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        try:
            removed = await delete_photo_set(db, store, photo_set_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"deleted_images": removed}

    @router.post(
        "/api/photo-sets/{photo_set_id}/train",
        status_code=202,
        dependencies=[Depends(require_same_site_header)],
    )
    async def train(
        photo_set_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        try:
            photo_set = await get_photo_set(db, photo_set_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if photo_set.status is not PhotoSetStatus.READY:
            # Training is the expensive step. A set that has not passed
            # validation must not reach it.
            raise HTTPException(
                status_code=409,
                detail=f"photo set is {photo_set.status.value}; it must pass validation first",
            )

        accepted = (
            (
                await db.execute(
                    select(Photo).where(
                        Photo.photo_set_id == photo_set_id, Photo.accepted.is_(True)
                    )
                )
            )
            .scalars()
            .all()
        )

        if not photo_set.avatar_id:
            raise HTTPException(
                status_code=409,
                detail="attach this photo set to an avatar before building",
            )

        result = await runner.start(
            TrainingRequest(
                tenant_id=user_id,
                photo_set_id=photo_set_id,
                image_keys=[p.blob_key for p in accepted],
                subject_name=photo_set.id,
            )
        )

        job = TrainingJob(
            owner_id=user_id,
            photo_set_id=photo_set_id,
            status=_STATE_TO_STATUS[result.state],
            provider=runner.name,
            external_id=result.external_id,
            error=result.error,
        )
        db.add(job)
        photo_set.status = PhotoSetStatus.TRAINING
        await db.commit()

        return {"job_id": job.id, "status": job.status.value}

    @router.get("/api/training-jobs/{job_id}")
    async def job_status(
        job_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        job = (
            await db.execute(
                select(TrainingJob).where(
                    TrainingJob.id == job_id, TrainingJob.owner_id == user_id
                )
            )
        ).scalar_one_or_none()
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")

        progress = 1.0 if job.status is TrainingStatus.SUCCEEDED else 0.0
        avatar_id: str | None = None

        if job.status in (TrainingStatus.QUEUED, TrainingStatus.RUNNING) and job.external_id:
            result = await runner.poll(job.external_id)
            job.status = _STATE_TO_STATUS[result.state]
            job.output_key = result.output_key or job.output_key
            job.error = result.error or job.error
            progress = result.progress

            if job.status is TrainingStatus.SUCCEEDED:
                photo_set = await db.get(PhotoSet, job.photo_set_id)
                if photo_set is not None:
                    photo_set.status = PhotoSetStatus.TRAINED
                await db.commit()

                # Training on its own changes nothing the customer can see.
                # Building the renderable assets is what makes the avatar
                # callable, so it happens here rather than being a later step
                # somebody has to know about.
                try:
                    avatar = await finalise_avatar(
                        db, store, settings, job.photo_set_id, user_id
                    )
                    avatar_id = avatar.id
                    progress = 1.0
                except FinaliseError as exc:
                    job.status = TrainingStatus.FAILED
                    job.error = f"could not build the avatar: {exc}"
            elif job.status is TrainingStatus.FAILED:
                photo_set = await db.get(PhotoSet, job.photo_set_id)
                if photo_set is not None:
                    photo_set.status = PhotoSetStatus.FAILED

            await db.commit()

        if avatar_id is None:
            photo_set = await db.get(PhotoSet, job.photo_set_id)
            avatar_id = photo_set.avatar_id if photo_set else None

        return {
            "id": job.id,
            "status": job.status.value,
            "provider": job.provider,
            "error": job.error,
            "progress": round(max(0.0, min(1.0, progress)), 3),
            "avatar_id": avatar_id,
        }

    return router
