"""The shared demo account.

One account, created on first use, that every visitor is signed into when
DEMO_MODE is on. It exists so somebody can try the product without inventing a
password on the worst week of their life, and so an operator can put a link in
front of a reviewer.

Three things about it are not negotiable, and each is enforced somewhere rather
than described here:

  It is one tenant, not one per visitor. Tenancy is what keeps a family's
  photographs away from strangers, and a demo that quietly minted a tenant per
  browser would be a different product with the same screens. ensure() returns
  the same row every time, and DEMO_EMAIL is the uniqueness constraint that
  makes that true even under two simultaneous first visits.

  Therefore everything uploaded into it is visible to everyone holding the
  link. The interface says so on every screen (components/DemoBanner.tsx). This
  product holds photographs of dead people and recordings of their voices;
  nobody may discover that property by accident.

  It may not coexist with real accounts. sessions.assert_demo_mode_safe refuses
  to start the process if this database holds any account other than this one.

The password hash is deliberately not a valid Argon2 digest. verify_password
fails closed on a malformed digest, so there is no password that signs into
this account through the normal route - the only way in is the demo path, and
turning DEMO_MODE off closes it completely rather than leaving a shared
credential behind.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.models import User

# The identity of the one shared tenant. Everything else in this module is
# arranged around the uniqueness of this address.
DEMO_EMAIL = "demo@shared.invalid"

# Not an Argon2 digest, on purpose. See the module docstring.
_UNUSABLE_PASSWORD_HASH = "!demo-account-has-no-password"

# Two visitors can arrive at an empty database in the same millisecond. The
# unique index on email is the real guarantee; this only saves the second one
# an exception round trip in the single-process deployment this runs in.
_creating = asyncio.Lock()


async def ensure(db: AsyncSession) -> User:
    """The demo account, created if this is the first visit.

    Returns the same row for every caller. A concurrent creation loses the
    insert on the unique index and re-reads rather than producing a second
    tenant, which is the failure this function exists to prevent.
    """
    existing = await _read(db)
    if existing is not None:
        return existing

    async with _creating:
        # Another request may have created it while this one waited.
        existing = await _read(db)
        if existing is not None:
            return existing

        user = User(email=DEMO_EMAIL, password_hash=_UNUSABLE_PASSWORD_HASH)
        db.add(user)
        try:
            await db.commit()
        except IntegrityError:
            # Lost the race against a different process on the same database.
            # The row it wrote is the account; ours never existed.
            await db.rollback()
            found = await _read(db)
            if found is None:  # pragma: no cover - the index says this cannot happen
                raise
            return found
        return user


async def _read(db: AsyncSession) -> User | None:
    return (
        await db.execute(select(User).where(User.email == DEMO_EMAIL))
    ).scalar_one_or_none()


async def real_account_emails(db: AsyncSession) -> list[str]:
    """Every account that is not the demo account.

    Read by the startup check. Named for what it means rather than what it
    queries: a non-empty answer is the statement "a real person has an account
    here", which is the condition under which demo mode must not run.
    """
    rows = (await db.execute(select(User.email).where(User.email != DEMO_EMAIL))).scalars()
    return sorted(rows)
