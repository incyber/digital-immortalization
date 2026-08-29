"""Verifying somebody else's likeness with nothing on file must be refused,
not merely warned about - a warning still leaves a record indistinguishable
from a properly evidenced one."""
import pytest
from sqlalchemy import select

from avatar.cli.consent import cmd_set
from avatar.gateway.models import ConsentRecord, ConsentStatus


async def _read(db_factory, avatar_id):
    async with db_factory() as db:
        return (
            await db.execute(select(ConsentRecord).where(ConsentRecord.avatar_id == avatar_id))
        ).scalar_one()


async def test_verify_without_evidence_is_refused_by_default(db_factory, avatar_without_evidence):
    with pytest.raises(SystemExit):
        await cmd_set(avatar_without_evidence, ConsentStatus.VERIFIED, "operator", "looks fine")

    record = await _read(db_factory, avatar_without_evidence)
    assert record.status is ConsentStatus.PENDING, "a refused verify must not change the record"


async def test_verify_without_evidence_succeeds_with_the_override_flag(
    db_factory, avatar_without_evidence
):
    await cmd_set(
        avatar_without_evidence,
        ConsentStatus.VERIFIED,
        "operator",
        "seen over a call with the family",
        force=True,
    )

    record = await _read(db_factory, avatar_without_evidence)
    assert record.status is ConsentStatus.VERIFIED


async def test_the_override_is_recorded_in_the_notes(db_factory, avatar_without_evidence):
    await cmd_set(
        avatar_without_evidence,
        ConsentStatus.VERIFIED,
        "operator",
        "seen over a call with the family",
        force=True,
    )

    record = await _read(db_factory, avatar_without_evidence)
    assert "force" in record.notes.lower()
    assert "seen over a call with the family" in record.notes


async def test_verify_with_evidence_on_file_needs_no_override(db_factory, avatar_with_evidence):
    # The refusal is specifically about evidence being absent; it must not
    # become a second gate on every verification.
    await cmd_set(avatar_with_evidence, ConsentStatus.VERIFIED, "operator", "evidence checked out")

    record = await _read(db_factory, avatar_with_evidence)
    assert record.status is ConsentStatus.VERIFIED
    assert "force" not in record.notes.lower()


async def test_reject_does_not_require_evidence_or_the_override_flag(
    db_factory, avatar_without_evidence
):
    # The evidence check only ever gates a move to VERIFIED.
    await cmd_set(avatar_without_evidence, ConsentStatus.REJECTED, "operator", "not convincing")

    record = await _read(db_factory, avatar_without_evidence)
    assert record.status is ConsentStatus.REJECTED
