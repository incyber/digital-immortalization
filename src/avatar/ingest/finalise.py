"""The step that makes an avatar callable.

Training produces an identity; this turns it into the frames the renderer
needs and marks the avatar ready. Without it a training run finishes and
nothing changes on screen, which is what "build the avatar" appearing to do
nothing looked like.

Assets are written where the agent dispatcher already looks for them, so a
finished build is picked up on the next call with no further wiring.
"""

from __future__ import annotations

import asyncio
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
from avatar.storage.keys import source_clip_key


class FinaliseError(RuntimeError):
    pass


def assets_dir_for(cfg: Settings, avatar_id: str) -> Path:
    """Where the dispatcher expects to find an avatar's assets."""
    return Path(cfg.assets_dir) / "avatars" / avatar_id


async def _copy_source_clip(
    store: BlobStore, owner_id: str, photo_set_id: str, destination: Path
) -> Path | None:
    """Put the uploaded footage where the renderer looks for its base.

    Copied rather than referenced. The renderer reads assets off local disk
    while a call is running, and a network fetch on that path would be a
    timeout waiting for somebody's first sentence.
    """
    try:
        data = await store.get(owner_id, source_clip_key(owner_id, photo_set_id))
    except Exception:  # noqa: BLE001 - no clip is the ordinary case
        return None

    destination.mkdir(parents=True, exist_ok=True)
    clip = destination / "base.mp4"
    clip.write_bytes(data)
    logger.info(f"base clip is the customer's own footage: {len(data) // 1024}KB")
    return clip


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

    # Off the event loop, both of them. Building the assets runs face
    # detection, sharpness scoring and mouth-plate generation over every
    # accepted photograph, and saving them writes a few hundred frames to
    # disk. This function is reached from the training-job status endpoint -
    # the request whose entire purpose is to answer "how is it going" - so
    # inline it froze the gateway at exactly the moment somebody was watching
    # it. Same failure as the video endpoint, one floor up.
    assets = await asyncio.to_thread(build_avatar_assets, images, framing=framing)

    destination = assets_dir_for(cfg, avatar.id)
    await asyncio.to_thread(assets.save, destination)

    # The customer's own footage is the base whenever it exists. This is the
    # difference between a face that moves and a face that does not: the lip
    # sync renderer animates a mouth onto a clip, and a clip of the real person
    # already contains their real head motion, blinks and posture. Nothing has
    # to generate any of it.
    #
    # Falling back to synthesising motion onto a still is strictly worse and is
    # only what happens when somebody uploaded photographs instead.
    clip = await _copy_source_clip(store, owner_id, photo_set_id, destination)
    if clip is None and assets.base_rgb is not None:
        await attach_base_clip(cfg, destination, assets.base_rgb)

    avatar.assets_key = str(destination)
    avatar.framing = framing.value
    await db.commit()

    logger.info(f"avatar {avatar.id} is callable: {framing.value} framing at {destination}")
    return avatar
