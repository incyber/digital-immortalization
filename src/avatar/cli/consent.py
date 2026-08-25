"""Operator review of consent records.

The only route to a verified consent record for somebody else's likeness. It
is a command rather than an endpoint on purpose: verification means a person
read an evidence document, and anything reachable from the product would
eventually be automated by whoever is in a hurry.

    python -m avatar.cli.consent list
    python -m avatar.cli.consent verify <avatar-id> --reviewer "name" --note "what you saw"
    python -m avatar.cli.consent reject <avatar-id> --reviewer "name" --note "why"
    python -m avatar.cli.consent revoke <avatar-id> --note "who asked"
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from avatar.config import get_settings
from avatar.gateway.db import init_engine
from avatar.gateway.models import Avatar, ConsentRecord, ConsentStatus


def _factory():
    return async_sessionmaker(init_engine(get_settings()), expire_on_commit=False)


async def _record(db, avatar_id: str) -> ConsentRecord:
    record = (
        await db.execute(select(ConsentRecord).where(ConsentRecord.avatar_id == avatar_id))
    ).scalar_one_or_none()
    if record is None:
        raise SystemExit(f"no consent record for avatar {avatar_id}")
    return record


async def cmd_list() -> None:
    async with _factory()() as db:
        rows = (await db.execute(select(ConsentRecord))).scalars().all()
        if not rows:
            print("no consent records")
            return
        for record in rows:
            avatar = await db.get(Avatar, record.avatar_id)
            name = avatar.display_name if avatar else "(deleted avatar)"
            print(
                f"{record.avatar_id}  {record.status.value:<9} {name}\n"
                f"    rights holder: {record.rights_holder_name} "
                f"({record.relationship_to_subject}, {record.jurisdiction})\n"
                f"    evidence: {record.evidence_s3_key or 'none on file'}"
            )


async def cmd_set(avatar_id: str, status: ConsentStatus, reviewer: str, note: str) -> None:
    async with _factory()() as db:
        record = await _record(db, avatar_id)

        if status is ConsentStatus.VERIFIED and not record.evidence_s3_key:
            # A verified record with nothing behind it is the failure this
            # command exists to prevent.
            print(
                "warning: no evidence document is on file for this avatar.\n"
                "         Verify only if you have seen the authorisation elsewhere."
            )

        record.status = status
        record.verified_at = datetime.now(UTC)
        record.verified_by = reviewer
        stamp = datetime.now(UTC).isoformat()
        record.notes = f"{record.notes or ''}\n{stamp} {status.value} by {reviewer}: {note}".strip()
        await db.commit()
        print(f"{avatar_id} is now {status.value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="every consent record and its status")

    for name, help_text in (
        ("verify", "mark a record verified after reading the evidence"),
        ("reject", "mark a record rejected"),
        ("revoke", "withdraw consent; takes effect on the next session"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("avatar_id")
        p.add_argument("--reviewer", default="operator", help="who reviewed it")
        p.add_argument("--note", default="", help="what was seen, or why not")

    args = parser.parse_args()

    if args.command == "list":
        asyncio.run(cmd_list())
        return

    status = {
        "verify": ConsentStatus.VERIFIED,
        "reject": ConsentStatus.REJECTED,
        "revoke": ConsentStatus.REVOKED,
    }[args.command]
    asyncio.run(cmd_set(args.avatar_id, status, args.reviewer, args.note))


if __name__ == "__main__":
    main()
