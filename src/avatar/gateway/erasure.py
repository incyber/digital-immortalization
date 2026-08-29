"""Deleting a person, and meaning it.

The product's own storage layer documents `delete_tenant` as satisfying a legal
deletion obligation, and an audit found it called from nowhere. There was no
way, through the product, to remove a voice clone, a built likeness, an avatar
or an account - only a photo set. "Delete my father's photographs" was
partially true, which for this product is worse than most bugs.

Two rules shape the code below.

**Blobs first, rows last.** A row deleted before its blobs leaves objects in a
bucket that nothing references and nobody can find, which is the one outcome
that cannot be corrected later. If the process dies halfway, the customer can
press delete again and the second attempt finishes the job; a missing object is
treated as already gone rather than as an error.

**A count is returned, not a status.** Somebody asking for their father's
photographs to be erased is owed a number, and support is owed something to
check. "It worked" is not a report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.models import (
    Avatar,
    ConsentRecord,
    Photo,
    PhotoSet,
    SceneObservation,
    Session,
    TrainingJob,
    User,
)
from avatar.gateway.tenancy import TenantError, assert_owned
from avatar.storage.base import BlobStore


@dataclass
class Erasure:
    """What was actually removed. Returned to the customer and to support."""

    avatars: int = 0
    photo_sets: int = 0
    photos: int = 0
    blobs: int = 0
    sessions: int = 0
    consent_records: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """False if anything refused to go. Never reported as done when not."""
        return not self.failures


async def _drop_blob(store: BlobStore, owner_id: str, key: str | None, out: Erasure) -> None:
    """Remove one object. Already gone counts as removed.

    A second attempt at a partly-finished erasure must be able to complete
    rather than failing on the objects the first attempt got to.
    """
    if not key:
        return
    try:
        await store.delete(owner_id, key)
        out.blobs += 1
    except FileNotFoundError:
        out.blobs += 1
    except Exception as exc:  # noqa: BLE001
        # Recorded, not raised. One unreachable object must not stop the rest
        # of a family's data being removed, and the caller is told.
        out.failures.append(f"{key}: {exc}")
        logger.warning(f"could not delete {key}: {exc}")


async def erase_avatar(
    db: AsyncSession, store: BlobStore, avatar_id: str, owner_id: str
) -> Erasure:
    """Remove one recreated person: their images, voice, likeness and consent.

    The consent record goes too. It exists to authorise a recreation, and once
    there is no recreation, keeping the name and relationship of a grieving
    family serves nobody.
    """
    await assert_owned(db, avatar_id, owner_id)
    out = Erasure()

    avatar = (
        await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    ).scalar_one()

    photo_sets = (
        (await db.execute(select(PhotoSet).where(PhotoSet.avatar_id == avatar_id)))
        .scalars()
        .all()
    )
    # A training job's output is a stored artefact too. It hangs off the photo
    # set rather than the avatar, so it has to be found through the set or it
    # is left behind - which is exactly the kind of orphan this module exists
    # to stop.
    for photo_set in photo_sets:
        jobs = (
            (
                await db.execute(
                    select(TrainingJob).where(TrainingJob.photo_set_id == photo_set.id)
                )
            )
            .scalars()
            .all()
        )
        for job in jobs:
            await _drop_blob(store, owner_id, job.output_key, out)
        await db.execute(
            sql_delete(TrainingJob).where(TrainingJob.photo_set_id == photo_set.id)
        )

        photos = (
            (await db.execute(select(Photo).where(Photo.photo_set_id == photo_set.id)))
            .scalars()
            .all()
        )
        for photo in photos:
            await _drop_blob(store, owner_id, photo.blob_key, out)
            out.photos += 1
        await db.execute(sql_delete(Photo).where(Photo.photo_set_id == photo_set.id))
        out.photo_sets += 1

    await _drop_blob(store, owner_id, avatar.voice_key, out)
    await _drop_blob(store, owner_id, avatar.assets_key, out)

    # What the camera saw during a call. Keyed by session rather than by
    # avatar, so the sessions have to be found first and removed after - a
    # grieving person's living room is not a thing to leave in a database once
    # they have asked for the recreation to be gone.
    session_ids = (
        (await db.execute(select(Session.id).where(Session.avatar_id == avatar_id)))
        .scalars()
        .all()
    )
    if session_ids:
        await db.execute(
            sql_delete(SceneObservation).where(
                SceneObservation.session_id.in_(session_ids)
            )
        )

    sessions = await db.execute(sql_delete(Session).where(Session.avatar_id == avatar_id))
    out.sessions = sessions.rowcount or 0

    consent = await db.execute(
        sql_delete(ConsentRecord).where(ConsentRecord.avatar_id == avatar_id)
    )
    out.consent_records = consent.rowcount or 0

    await db.execute(sql_delete(PhotoSet).where(PhotoSet.avatar_id == avatar_id))
    await db.execute(sql_delete(Avatar).where(Avatar.id == avatar_id))
    out.avatars = 1

    await db.commit()
    logger.info(f"erased avatar {avatar_id}: {out}")
    return out


async def erase_account(db: AsyncSession, store: BlobStore, owner_id: str) -> Erasure:
    """Remove everything belonging to one customer, then the customer.

    Every avatar goes through the same path as a single deletion rather than a
    faster bulk one, so there is only one place where the order of operations
    can be wrong. Then the tenant's whole storage prefix is swept, which
    catches anything orphaned by an interrupted earlier attempt.
    """
    out = Erasure()

    avatars = (
        (await db.execute(select(Avatar).where(Avatar.owner_id == owner_id)))
        .scalars()
        .all()
    )
    for avatar in avatars:
        try:
            one = await erase_avatar(db, store, avatar.id, owner_id)
        except TenantError:  # pragma: no cover - it was just read as theirs
            continue
        out.avatars += one.avatars
        out.photo_sets += one.photo_sets
        out.photos += one.photos
        out.blobs += one.blobs
        out.sessions += one.sessions
        out.consent_records += one.consent_records
        out.failures.extend(one.failures)

    try:
        out.blobs += await store.delete_tenant(owner_id)
    except Exception as exc:  # noqa: BLE001
        out.failures.append(f"storage sweep: {exc}")
        logger.warning(f"tenant sweep failed for {owner_id}: {exc}")

    await db.execute(sql_delete(User).where(User.id == owner_id))
    await db.commit()

    logger.info(f"erased account {owner_id}: {out}")
    return out
