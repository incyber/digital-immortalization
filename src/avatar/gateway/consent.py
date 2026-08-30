"""The gate.

One function, one rule, called from exactly one place. Keeping it here rather
than inlining the check at the call site is deliberate: a legal control that
appears in two places eventually differs between them.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.models import Avatar, ConsentRecord, ConsentStatus


class ConsentError(PermissionError):
    """Raised whenever a session must not start. Carries the reason so the
    customer is told what to fix rather than merely refused."""

    def __init__(self, reason: str, status: str | None = None):
        self.status = status
        super().__init__(reason)


# Both open a session. VERIFIED means a reviewer read the evidence;
# SELF_ATTESTED means the account holder claimed to be the subject and no
# third party's rights are engaged. Callers that only need "may this call
# proceed" should use this; callers that care whether a human reviewed the
# claim should read record.status directly instead of calling this a second
# time - see cli/consent.py for the operator side of that distinction.
_OPEN_STATUSES = (ConsentStatus.VERIFIED, ConsentStatus.SELF_ATTESTED)


async def assert_consented(db: AsyncSession, avatar_id: str) -> ConsentRecord:
    """Return the consent record, or raise.

    There is no permissive branch. An avatar with no record, a record awaiting
    review, a rejected record and a revoked record all refuse, each with its
    own message.
    """
    avatar = await db.get(Avatar, avatar_id)
    if avatar is None:
        raise ConsentError(f"no avatar {avatar_id}")

    record = (
        await db.execute(select(ConsentRecord).where(ConsentRecord.avatar_id == avatar_id))
    ).scalar_one_or_none()

    if record is None:
        raise ConsentError(
            "no consent record on file for this avatar; a documented "
            "rights-holder is required before any session can start"
        )

    if record.status not in _OPEN_STATUSES:
        raise ConsentError(
            f"consent status is {record.status.value}, not verified", status=record.status.value
        )

    return record


async def revoke(db: AsyncSession, avatar_id: str, note: str) -> None:
    """Withdraw consent with immediate effect.

    Takes effect on the next session request. Ending calls already in progress
    is the realtime agent's responsibility, which watches for this state.
    """
    record = (
        await db.execute(select(ConsentRecord).where(ConsentRecord.avatar_id == avatar_id))
    ).scalar_one_or_none()
    if record is None:
        raise ConsentError(f"no consent record for avatar {avatar_id}")

    record.status = ConsentStatus.REVOKED
    record.notes = f"{record.notes or ''}\nrevoked {datetime.now(UTC).isoformat()}: {note}"
    await db.commit()


class NoLikeness(RuntimeError):
    """There is nothing built for this avatar, so there is nothing to show.

    Separate from a consent failure because it means something different to
    the person reading it: consent is a permission they may not be able to
    obtain, and this is a step they have not finished yet.
    """


async def assert_has_likeness(db: AsyncSession, avatar_id: str) -> None:
    """Refuse a call until a real likeness exists.

    There was a generated stand-in behind this once - a still photograph with
    a mouth warped into it - so that a call could run before anybody had
    uploaded anything. It reached a customer, who saw it and reasonably
    concluded the product did not work.

    Somebody opening this is looking for a person who has died. An
    approximation of them is not a lesser version of the product, it is a
    different and worse thing, and no amount of "it is only a placeholder"
    survives contact with the moment they see it. So: something real, or
    nothing.
    """
    from avatar.gateway.models import Avatar

    avatar = (
        await db.execute(select(Avatar).where(Avatar.id == avatar_id))
    ).scalar_one_or_none()

    if avatar is None or not (avatar.splat_key or avatar.assets_key):
        raise NoLikeness(
            "there is no likeness for this person yet - upload a video or "
            "photographs and build one first"
        )
