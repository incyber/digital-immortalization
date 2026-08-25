"""Photo set lifecycle: create, upload, validate, hand to training.

Every function takes an owner_id and every storage key is built from it, so a
photo set cannot be created, read, added to, or trained under another tenant.
"""

from __future__ import annotations

import json

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.models import Photo, PhotoSet, PhotoSetStatus
from avatar.gateway.tenancy import TenantError
from avatar.ingest.validate import (
    MAX_ACCEPTED,
    PhotoVerdict,
    Verdict,
    inspect_photo,
    inspect_set,
)
from avatar.storage.base import BlobStore
from avatar.storage.keys import photo_key, photo_set_prefix

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Twenty-five phone photographs at full resolution. Above this something is
# wrong - a video, a raw file, or an attempt to fill the bucket.
MAX_PHOTO_BYTES = 25 * 1024 * 1024

# A hard cap on how many files may be attached to one set, independent of how
# many pass validation. Without it, rejected uploads are an unbounded write.
MAX_UPLOADS = MAX_ACCEPTED * 3


async def create_photo_set(db: AsyncSession, owner_id: str) -> PhotoSet:
    photo_set = PhotoSet(owner_id=owner_id, status=PhotoSetStatus.UPLOADING)
    db.add(photo_set)
    await db.commit()
    return photo_set


async def get_photo_set(db: AsyncSession, photo_set_id: str, owner_id: str) -> PhotoSet:
    """Fetch a set, scoped to its owner. Refuses identically for missing and
    for another tenant's, so ids cannot be probed."""
    result = await db.execute(
        select(PhotoSet).where(PhotoSet.id == photo_set_id, PhotoSet.owner_id == owner_id)
    )
    photo_set = result.scalar_one_or_none()
    if photo_set is None:
        raise TenantError("no such photo set")
    return photo_set


async def add_photo(
    db: AsyncSession,
    store: BlobStore,
    photo_set_id: str,
    owner_id: str,
    filename: str,
    content_type: str,
    data: bytes,
) -> Photo:
    """Store one image and record what validation made of it.

    Validation happens here rather than at training time so the customer sees
    a per-image verdict while they are still on the page and can retake.
    """
    photo_set = await get_photo_set(db, photo_set_id, owner_id)

    if photo_set.status not in (PhotoSetStatus.UPLOADING, PhotoSetStatus.REJECTED):
        raise ValueError(f"photo set is {photo_set.status.value} and no longer accepts uploads")

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(f"unsupported image type {content_type!r}")

    if len(data) > MAX_PHOTO_BYTES:
        raise ValueError(f"image is larger than {MAX_PHOTO_BYTES // (1024 * 1024)}MB")

    count = (
        await db.execute(
            select(func.count()).select_from(Photo).where(Photo.photo_set_id == photo_set_id)
        )
    ).scalar_one()
    if count >= MAX_UPLOADS:
        raise ValueError(f"a photo set accepts at most {MAX_UPLOADS} files")

    verdict = inspect_photo(filename, data)

    key = photo_key(owner_id, photo_set_id, _safe_name(filename, count))
    stored = await store.put(owner_id, key, data, content_type)

    photo = Photo(
        photo_set_id=photo_set_id,
        owner_id=owner_id,
        blob_key=stored.key,
        filename=filename,
        content_type=content_type,
        size_bytes=stored.size,
        accepted=verdict.verdict is Verdict.OK,
        rejection_reasons=json.dumps([r.value for r in verdict.reasons]) or None,
        face_height_fraction=verdict.face_height_fraction,
    )
    db.add(photo)
    await db.commit()
    return photo


def _safe_name(filename: str, index: int) -> str:
    """A key-safe name derived from the upload.

    The customer's filename is kept in the database for display; the storage
    key uses a generated one, so nothing user-controlled reaches a key even if
    key validation were later relaxed.
    """
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    if suffix not in {"jpg", "jpeg", "png", "webp"}:
        suffix = "jpg"
    return f"photo-{index:03d}.{suffix}"


async def revalidate_set(
    db: AsyncSession, store: BlobStore, photo_set_id: str, owner_id: str
) -> PhotoSet:
    """Re-inspect every stored image with the current rules.

    Validation improves; customers should not have to re-upload photographs of
    somebody who has died because the checks changed. The images are already in
    storage, so this re-reads them and rewrites the verdicts in place.
    """
    # Ownership first; the row itself is re-read by evaluate_set at the end.
    await get_photo_set(db, photo_set_id, owner_id)

    photos = (
        (await db.execute(select(Photo).where(Photo.photo_set_id == photo_set_id)))
        .scalars()
        .all()
    )

    for photo in photos:
        try:
            data = await store.get(owner_id, photo.blob_key)
        except Exception:  # noqa: BLE001 - a missing object is simply unusable
            photo.accepted = False
            photo.rejection_reasons = json.dumps(["stored image could not be read"])
            continue

        verdict = inspect_photo(photo.filename, data)
        photo.accepted = verdict.verdict is Verdict.OK
        photo.rejection_reasons = json.dumps([r.value for r in verdict.reasons]) or None
        photo.face_height_fraction = verdict.face_height_fraction

    await db.commit()
    return await evaluate_set(db, photo_set_id, owner_id)


async def evaluate_set(db: AsyncSession, photo_set_id: str, owner_id: str) -> PhotoSet:
    """Judge the whole set and move it to ready or rejected."""
    photo_set = await get_photo_set(db, photo_set_id, owner_id)

    photos = (
        (await db.execute(select(Photo).where(Photo.photo_set_id == photo_set_id)))
        .scalars()
        .all()
    )

    verdicts = [
        PhotoVerdict(
            filename=p.filename,
            verdict=Verdict.OK if p.accepted else Verdict.REJECTED,
            reasons=[],
            face_height_fraction=p.face_height_fraction,
        )
        for p in photos
    ]
    result = inspect_set(verdicts)

    photo_set.usable_count = len(result.usable)
    photo_set.half_body_count = result.half_body_count
    photo_set.problems = json.dumps(result.problems) if result.problems else None
    photo_set.status = PhotoSetStatus.READY if result.acceptable else PhotoSetStatus.REJECTED
    await db.commit()
    return photo_set


async def delete_photo_set(
    db: AsyncSession, store: BlobStore, photo_set_id: str, owner_id: str
) -> int:
    """Remove a set and every image in it.

    Storage first, then rows. If it failed the other way round, an interrupted
    delete would leave photographs in the bucket with nothing pointing at them
    - unreachable through the product and still present on disk, which is the
    worst outcome for material somebody asked to have removed.
    """
    await get_photo_set(db, photo_set_id, owner_id)

    prefix = photo_set_prefix(owner_id, photo_set_id)
    removed = 0
    for obj in await store.list(owner_id, prefix):
        await store.delete(owner_id, obj.key)
        removed += 1

    photos = (
        (await db.execute(select(Photo).where(Photo.photo_set_id == photo_set_id)))
        .scalars()
        .all()
    )
    for photo in photos:
        await db.delete(photo)

    photo_set = await get_photo_set(db, photo_set_id, owner_id)
    await db.delete(photo_set)
    await db.commit()
    return removed


def describe(photo_set: PhotoSet, photos: list[Photo]) -> dict:
    """The shape the upload page renders."""
    return {
        "id": photo_set.id,
        "status": photo_set.status.value,
        "usable_count": photo_set.usable_count,
        "half_body_count": photo_set.half_body_count,
        "problems": json.loads(photo_set.problems) if photo_set.problems else [],
        "photos": [
            {
                "id": p.id,
                "filename": p.filename,
                "accepted": p.accepted,
                "reasons": json.loads(p.rejection_reasons) if p.rejection_reasons else [],
                "half_body": 0.0 < p.face_height_fraction < 0.33,
            }
            for p in photos
        ],
    }


__all__ = [
    "ALLOWED_CONTENT_TYPES",
    "MAX_PHOTO_BYTES",
    "MAX_UPLOADS",
    "add_photo",

    "create_photo_set",
    "delete_photo_set",
    "describe",
    "evaluate_set",
    "get_photo_set",
]
