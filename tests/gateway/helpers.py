"""Shared helpers for gateway tests."""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.models import ConsentRecord, ConsentStatus


async def set_status(db: AsyncSession, avatar, status: str) -> ConsentRecord:
    record = (
        await db.execute(select(ConsentRecord).where(ConsentRecord.avatar_id == avatar.id))
    ).scalar_one()
    record.status = ConsentStatus(status)
    await db.commit()
    return record
