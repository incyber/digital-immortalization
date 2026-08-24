"""Create the demo avatar with a verified consent record.

Cristóbal Colón is used deliberately: he died in 1506, so no post-mortem
personality right subsists anywhere, and the consent record below is a
demonstration of the mechanism rather than a claim about a living family. A
real avatar of a recently deceased person requires an actual rights-holder and
an actual evidence document, which is what the status column enforces.
"""

import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from avatar.config import get_settings
from avatar.gateway.db import create_all, init_engine
from avatar.gateway.models import Avatar, ConsentRecord, ConsentStatus, User

PROFILE_PATH = "src/avatar/profiles/colon.json"


async def main() -> None:
    cfg = get_settings()
    await create_all(cfg)

    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(init_engine(cfg), expire_on_commit=False)
    async with factory() as db:
        user = (
            await db.execute(select(User).where(User.email == "demo@example.com"))
        ).scalar_one_or_none()
        if user is None:
            user = User(email="demo@example.com", password_hash="!unusable")
            db.add(user)
            await db.flush()

        avatar = (await db.execute(select(Avatar).where(Avatar.id == "colon"))).scalar_one_or_none()
        if avatar is None:
            avatar = Avatar(
                id="colon",
                owner_id=user.id,
                display_name="Cristóbal Colón",
                locale="es",
                profile_path=PROFILE_PATH,
            )
            db.add(avatar)
            await db.flush()

        record = (
            await db.execute(select(ConsentRecord).where(ConsentRecord.avatar_id == "colon"))
        ).scalar_one_or_none()
        if record is None:
            record = ConsentRecord(avatar_id="colon")
            db.add(record)

        record.rights_holder_name = "Public domain — died 1506"
        record.relationship_to_subject = "none required"
        record.jurisdiction = "n/a"
        record.status = ConsentStatus.VERIFIED
        record.verified_at = datetime.now(timezone.utc)
        record.verified_by = "seed script"
        record.notes = (
            "No post-mortem personality right subsists. Present so the "
            "consent gate has something to pass; not a template for a real one."
        )
        await db.commit()

    print("seeded avatar 'colon' with a verified consent record")


if __name__ == "__main__":
    asyncio.run(main())
