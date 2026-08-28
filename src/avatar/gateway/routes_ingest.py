"""Upload and training endpoints.

Every handler resolves the signed-in tenant first and passes it into the
service layer, which builds every storage key from it. There is no path here
that takes a tenant id from the request body.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
from avatar.ingest.video import VideoError, extract_frames, is_video
from avatar.training.base import JobState, TrainingRequest

_STATE_TO_STATUS = {
    JobState.QUEUED: TrainingStatus.QUEUED,
    JobState.RUNNING: TrainingStatus.RUNNING,
    JobState.SUCCEEDED: TrainingStatus.SUCCEEDED,
    JobState.FAILED: TrainingStatus.FAILED,
    JobState.CANCELLED: TrainingStatus.CANCELLED,
}


def build_router(settings, current_user, get_db, store, runner) -> APIRouter:
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

    @router.post("/api/photo-sets", status_code=201)
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

    @router.post("/api/photo-sets/{photo_set_id}/video", status_code=201)
    async def upload_video(
        photo_set_id: str,
        file: UploadFile = File(...),  # noqa: B008
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        """A clip in, frames added to the set as if each had been uploaded.

        Every frame goes through the same checks as a photograph, so a clip and
        an album are held to one standard and the caller reads one shape of
        result either way.
        """
        data = await file.read()
        if not is_video(file.content_type or "", file.filename or ""):
            raise HTTPException(status_code=400, detail="that is not a video file")

        try:
            frames = extract_frames(data)
        except VideoError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        added = []
        for index, frame in enumerate(frames):
            try:
                photo = await add_photo(
                    db, store, photo_set_id, user_id,
                    f"frame-{index:04d}.jpg", "image/jpeg", frame,
                )
            except TenantError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError:
                # One unusable frame is not a failed upload. A clip is expected
                # to contain blinks, blur and turns away from camera; rejecting
                # the whole video for them would reject every real video.
                continue
            added.append(
                {
                    "id": photo.id,
                    "filename": photo.filename,
                    "accepted": photo.accepted,
                    "reasons": json.loads(photo.rejection_reasons)
                    if photo.rejection_reasons
                    else [],
                }
            )

        return {
            "frames_examined": len(frames),
            "photos": added,
            "accepted": sum(1 for p in added if p["accepted"]),
        }

    @router.post("/api/photo-sets/{photo_set_id}/evaluate")
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

    @router.post("/api/photo-sets/{photo_set_id}/revalidate")
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

    @router.post("/api/photo-sets/{photo_set_id}/train", status_code=202)
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
