"""Tenant isolation.

This system holds photographs and voice of dead people, uploaded by their
families. A cross-tenant read is not a bug report, it is the end of the
company. So isolation is asserted here as its own concern, separately from
consent, and the tests are written from the attacker's side: what happens when
a signed-in user asks for something that is not theirs.
"""

import pytest

from avatar.gateway.tenancy import TenantError, assert_owned, owned_query
from tests.gateway.helpers import set_status


async def test_owner_can_reach_their_own_avatar(db, owner, avatar):
    assert (await assert_owned(db, avatar.id, owner.id)).id == avatar.id


async def test_another_tenant_cannot_reach_it(db, other_owner, avatar):
    with pytest.raises(TenantError):
        await assert_owned(db, avatar.id, other_owner.id)


async def test_the_error_does_not_confirm_the_avatar_exists(db, other_owner, avatar):
    # A distinguishable "exists but not yours" versus "does not exist" lets an
    # outsider enumerate which avatars are real.
    with pytest.raises(TenantError) as found:
        await assert_owned(db, avatar.id, other_owner.id)
    with pytest.raises(TenantError) as missing:
        await assert_owned(db, "no-such-avatar", other_owner.id)
    assert str(found.value) == str(missing.value)


async def test_unknown_avatar_is_refused(db, owner):
    with pytest.raises(TenantError):
        await assert_owned(db, "no-such-avatar", owner.id)


async def test_owned_query_never_returns_another_tenants_rows(db, owner, other_owner, avatar):
    from avatar.gateway.models import Avatar

    mine = (await db.execute(owned_query(Avatar, owner.id))).scalars().all()
    theirs = (await db.execute(owned_query(Avatar, other_owner.id))).scalars().all()
    assert [a.id for a in mine] == [avatar.id]
    assert theirs == []


async def test_consent_alone_does_not_grant_access(db, other_owner, avatar):
    # Consent says the recreation is permitted. It says nothing about who may
    # place the call. Conflating the two is how a verified avatar becomes
    # reachable by every account on the platform.
    await set_status(db, avatar, "verified")
    with pytest.raises(TenantError):
        await assert_owned(db, avatar.id, other_owner.id)
