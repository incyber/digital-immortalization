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


async def test_every_tenant_scoped_model_can_actually_be_scoped(db, owner):
    # owned_query raises for a model with neither owner_id nor avatar_id.
    # Running it over every table is how a future model that forgets tenancy
    # gets caught here rather than in production.
    from avatar.gateway.models import (
        Avatar,
        ConsentRecord,
        Photo,
        PhotoSet,
        SafetyEvent,
        Session,
        TrainingJob,
    )

    for model in (Avatar, ConsentRecord, PhotoSet, Photo, TrainingJob, Session, SafetyEvent):
        query = owned_query(model, owner.id)
        await db.execute(query)  # must build and run without error


async def test_no_table_holding_tenant_data_is_unscopable():
    """Every mapped table must be reachable by exactly one tenant.

    A table with neither owner_id nor avatar_id cannot be filtered, so any
    query over it returns every tenant's rows. Catching that here is the
    difference between a design rule and a comment nobody reads.
    """
    from avatar.gateway.models import Base, User

    # Users are the tenant, so they are scoped by identity rather than by a
    # foreign key to themselves.
    exempt = {User.__tablename__}

    unscopable = [
        mapper.class_.__name__
        for mapper in Base.registry.mappers
        if mapper.class_.__tablename__ not in exempt
        and not hasattr(mapper.class_, "owner_id")
        and not hasattr(mapper.class_, "avatar_id")
    ]
    assert unscopable == [], (
        f"these tables hold tenant data but cannot be scoped to a tenant: {unscopable}"
    )
