"""The gate has no permissive branch. Every status is tested, not just the
happy one, because the failure that matters is a status nobody considered
turning into an open door."""
import pytest

from avatar.gateway.consent import ConsentError, assert_consented, revoke
from avatar.gateway.models import ConsentStatus
from tests.gateway.helpers import set_status


@pytest.mark.parametrize("status", ["pending", "rejected", "revoked"])
async def test_non_verified_status_is_refused(db, avatar, status):
    await set_status(db, avatar, status)
    with pytest.raises(ConsentError) as exc:
        await assert_consented(db, avatar.id)
    assert status in str(exc.value)


async def test_verified_status_passes(db, avatar):
    await set_status(db, avatar, "verified")
    record = await assert_consented(db, avatar.id)
    assert record.status is ConsentStatus.VERIFIED


async def test_self_attested_status_also_passes(db, avatar):
    # A self-attestation is a real, callable basis for a session even though
    # it is not a reviewed verification - see routes_avatars.py.
    await set_status(db, avatar, "self_attested")
    record = await assert_consented(db, avatar.id)
    assert record.status is ConsentStatus.SELF_ATTESTED


async def test_self_attested_stays_distinguishable_from_verified(db, avatar):
    # The gate treats the two statuses alike; nothing downstream is allowed
    # to collapse them into the same value.
    await set_status(db, avatar, "self_attested")
    record = await assert_consented(db, avatar.id)
    assert record.status is not ConsentStatus.VERIFIED


async def test_missing_record_is_refused(db, avatar_without_consent):
    with pytest.raises(ConsentError, match="no consent record"):
        await assert_consented(db, avatar_without_consent.id)


async def test_unknown_avatar_is_refused(db):
    with pytest.raises(ConsentError, match="no avatar"):
        await assert_consented(db, "does-not-exist")


async def test_revocation_takes_effect_immediately(db, avatar):
    await set_status(db, avatar, "verified")
    await assert_consented(db, avatar.id)          # callable now
    await revoke(db, avatar.id, note="family withdrew permission")
    with pytest.raises(ConsentError, match="revoked"):
        await assert_consented(db, avatar.id)      # and not a moment later
