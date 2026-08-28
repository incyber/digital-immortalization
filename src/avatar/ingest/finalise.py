"""The step that makes an avatar callable.

Training produces an identity; this turns it into the frames the renderer
needs and marks the avatar ready. Without it a training run finishes and
nothing changes on screen, which is what "build the avatar" appearing to do
nothing looked like.

Assets are written where the agent dispatcher already looks for them, so a
finished build is picked up on the next call with no further wiring.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.config import Settings
from avatar.gateway.models import Avatar, Photo, PhotoSet
from avatar.ingest.assets import build_avatar_assets
from avatar.ingest.gpu_assets import attach_base_clip
from avatar.ingest.validate import Framing
from avatar.storage.base import BlobStore


class FinaliseError(RuntimeError):
    pass


def assets_dir_for(cfg: Settings, avatar_id: str) -> Path:
    """Where the dispatcher expects to find an avatar's assets."""
    return Path(cfg.assets_dir) / "avatars" / avatar_id


async def finalise_avatar(
    db: AsyncSession,
    store: BlobStore,
    cfg: Settings,
    photo_set_id: str,
    owner_id: str,
) -> Avatar:
    """Build renderable assets from the accepted photographs and attach them.

    Reads the images back from storage rather than keeping them in memory
    across a training run that may have taken half an hour.
    """
    photo_set = (
        await db.execute(
            select(PhotoSet).where(
                PhotoSet.id == photo_set_id, PhotoSet.owner_id == owner_id
            )
        )
    ).scalar_one_or_none()
    if photo_set is None:
        raise FinaliseError("no such photo set")

    if not photo_set.avatar_id:
        raise FinaliseError("this photo set is not attached to an avatar")

    avatar = (
        await db.execute(
            select(Avatar).where(
                Avatar.id == photo_set.avatar_id, Avatar.owner_id == owner_id
            )
        )
    ).scalar_one_or_none()
    if avatar is None:
        raise FinaliseError("no such avatar")

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
    if not accepted:
        raise FinaliseError("this photo set has no usable photographs")

    images: list[bytes] = []
    for photo in accepted:
        try:
            images.append(await store.get(owner_id, photo.blob_key))
        except Exception as exc:  # noqa: BLE001 - a missing object is simply skipped
            logger.warning(f"could not read {photo.blob_key}: {exc}")

    if not images:
        raise FinaliseError("none of the stored photographs could be read")

    framing = Framing(photo_set.framing) if photo_set.framing else Framing.HEAD
    assets = build_avatar_assets(images, framing=framing)

    destination = assets_dir_for(cfg, avatar.id)
    assets.save(destination)

    # After the plate assets are saved, so a GPU outage costs the avatar its
    # head motion rather than its existence. Returns None when no endpoint is
    # configured, which is every developer machine.
    if assets.base_rgb is not None:
        attach_base_clip(cfg, destination, assets.base_rgb)

    avatar.assets_key = str(destination)
    avatar.framing = framing.value
    await db.commit()

    logger.info(f"avatar {avatar.id} is callable: {framing.value} framing at {destination}")
    return avatar
