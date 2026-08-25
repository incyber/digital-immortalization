"""Tenant isolation.

Every row in this system belongs to exactly one account. The data is
photographs, voice and conversation transcripts of dead people, uploaded by
their families, so a cross-tenant read is categorically worse than an outage.

The rule is enforced in one module rather than repeated at call sites, for the
same reason the consent gate is: a control that appears in twelve places
eventually differs in one of them.

Two things deliberately look unhelpful:

  Ownership is checked separately from consent. Consent says a recreation is
  permitted at all; ownership says who may reach it. An avatar can be fully
  consented and still none of your business.

  Refusals do not distinguish "not yours" from "does not exist". Telling them
  apart lets an outsider enumerate which avatar ids are real.
"""

from __future__ import annotations

from typing import TypeVar

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.models import Avatar

T = TypeVar("T")

# One message for every refusal. See the module docstring.
_REFUSED = "no such avatar"


class TenantError(PermissionError):
    """Raised when a tenant asks for something that is not theirs."""


async def assert_owned(db: AsyncSession, avatar_id: str, owner_id: str) -> Avatar:
    """Return the avatar only if this tenant owns it.

    The ownership predicate is part of the query rather than a check on the
    result, so there is no window in which another tenant's row exists in
    memory and could be returned by a later edit to this function.
    """
    result = await db.execute(
        select(Avatar).where(Avatar.id == avatar_id, Avatar.owner_id == owner_id)
    )
    avatar = result.scalar_one_or_none()
    if avatar is None:
        raise TenantError(_REFUSED)
    return avatar


def owned_query(model: type[T], owner_id: str) -> Select:
    """A SELECT already narrowed to one tenant.

    Use this instead of select(Model) anywhere a tenant is browsing their own
    records. Models reached through an avatar carry owner_id indirectly; those
    are joined here so callers cannot forget to.
    """
    if model is Avatar:
        return select(Avatar).where(Avatar.owner_id == owner_id)

    if hasattr(model, "owner_id"):
        return select(model).where(model.owner_id == owner_id)  # type: ignore[attr-defined]

    if hasattr(model, "avatar_id"):
        return (
            select(model)
            .join(Avatar, Avatar.id == model.avatar_id)  # type: ignore[attr-defined]
            .where(Avatar.owner_id == owner_id)
        )

    raise TypeError(
        f"{model.__name__} has no owner_id or avatar_id, so it cannot be "
        "scoped to a tenant; give it one rather than querying it unscoped"
    )
