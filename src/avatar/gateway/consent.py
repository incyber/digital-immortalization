"""The gate.

One function, one rule, called from exactly one place. Keeping it here rather
than inlining the check at the call site is deliberate: a legal control that
appears in two places eventually differs between them.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.models import Avatar, ConsentRecord, ConsentStatus


class ConsentError(PermissionError):
    """Raised whenever a session must not start. Carries the reason so the
    customer is told what to fix rather than merely refused."""

    def __init__(self, reason: str, status: str | None = None):
        self.status = status
        super().__init__(reason)


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

    if record.status is not ConsentStatus.VERIFIED:
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
    record.notes = f"{record.notes or ''}\nrevoked {datetime.now(timezone.utc).isoformat()}: {note}"
    await db.commit()
