"""Avatar creation and management.

An account starts with nothing. Every character is described and named by its
owner; the application ships with none.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.models import Avatar, ConsentRecord, ConsentStatus, PhotoSet
from avatar.gateway.tenancy import TenantError, assert_owned, owned_query
from avatar.persona import InvalidProfile, persona_from_avatar
from avatar.safety.crisis_lines import UnsupportedCountry, parse_attested, selectable


class AvatarInput(BaseModel):
    """What the customer supplies. Notably absent: the disclosure text and the
    crisis line, neither of which they get to write."""

    display_name: str = Field(min_length=1, max_length=255)
    locale: str = Field(default="en", max_length=8)
    country: str = Field(min_length=2, max_length=2)
    biography: str = Field(min_length=1)
    voice_description: str = ""
    boundaries: str = ""


class ConsentInput(BaseModel):
    rights_holder_name: str = Field(min_length=1, max_length=255)
    relationship_to_subject: str = Field(min_length=1, max_length=128)
    jurisdiction: str = Field(min_length=1, max_length=64)
    evidence_key: str | None = None


# Recreating yourself is the one case where the account holder is
# unambiguously the rights holder. Nobody needs to review a person's
# permission to be themselves, and requiring it would make the gate
# ceremony rather than protection.
SELF_RELATIONSHIP = "self"


def _describe(avatar: Avatar, attested: frozenset[str]) -> dict:
    try:
        persona = persona_from_avatar(avatar, attested)
        disclosure = persona.disclosure
        crisis = {"name": persona.crisis_line.name, "number": persona.crisis_line.number}
        usable = True
    except (InvalidProfile, UnsupportedCountry) as exc:
        disclosure, crisis, usable = str(exc), None, False

    return {
        "id": avatar.id,
        "display_name": avatar.display_name,
        "locale": avatar.locale,
        "country": avatar.country,
        "biography": avatar.biography,
        "voice_description": avatar.voice_description,
        "boundaries": avatar.boundaries,
        "photo_set_id": avatar.photo_set_id,
        "has_assets": bool(avatar.assets_key),
        "disclosure": disclosure,
        "crisis_line": crisis,
        "callable": usable and bool(avatar.assets_key),
    }


def build_router(settings, current_user, get_db) -> APIRouter:
    router = APIRouter()
    attested = parse_attested(settings.crisis_lines_verified)

    @router.get("/api/countries")
    async def countries():
        """Countries an avatar can currently be created for.

        Empty until the operator attests a crisis line, which is deliberate:
        the product should not run somewhere it cannot direct a distressed
        person to real help.
        """
        return {
            "countries": [
                {
                    "code": line.country,
                    "name": line.country_name,
                    "locale": line.locale,
                    "crisis_line": f"{line.name} ({line.number})",
                }
                for line in selectable(attested)
            ]
        }

    @router.post("/api/avatars", status_code=201)
    async def create(
        body: AvatarInput,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        avatar = Avatar(
            owner_id=user_id,
            display_name=body.display_name.strip(),
            locale=body.locale.strip() or "en",
            country=body.country.strip().upper(),
            biography=body.biography.strip(),
            voice_description=body.voice_description.strip(),
            boundaries=body.boundaries.strip(),
        )

        # Validated before it is stored, so an avatar that could never be
        # spoken to is never created in the first place.
        try:
            persona_from_avatar(avatar, attested)
        except (InvalidProfile, UnsupportedCountry) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        db.add(avatar)
        await db.commit()
        return _describe(avatar, attested)

    @router.get("/api/avatars")
    async def list_avatars(
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        rows = (await db.execute(owned_query(Avatar, user_id))).scalars().all()
        return {"avatars": [_describe(a, attested) for a in rows]}

    @router.get("/api/avatars/{avatar_id}")
    async def read(
        avatar_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        try:
            avatar = await assert_owned(db, avatar_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _describe(avatar, attested)

    @router.patch("/api/avatars/{avatar_id}")
    async def update(
        avatar_id: str,
        body: AvatarInput,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        try:
            avatar = await assert_owned(db, avatar_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        avatar.display_name = body.display_name.strip()
        avatar.locale = body.locale.strip() or "en"
        avatar.country = body.country.strip().upper()
        avatar.biography = body.biography.strip()
        avatar.voice_description = body.voice_description.strip()
        avatar.boundaries = body.boundaries.strip()

        try:
            persona_from_avatar(avatar, attested)
        except (InvalidProfile, UnsupportedCountry) as exc:
            await db.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await db.commit()
        return _describe(avatar, attested)

    @router.post("/api/avatars/{avatar_id}/photo-set/{photo_set_id}")
    async def attach_photo_set(
        avatar_id: str,
        photo_set_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        """Bind a trained photo set to an avatar.

        Both sides are checked against the same owner, so a photo set cannot be
        attached to somebody else's avatar or vice versa.
        """
        try:
            avatar = await assert_owned(db, avatar_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        photo_set = (
            await db.execute(
                select(PhotoSet).where(
                    PhotoSet.id == photo_set_id, PhotoSet.owner_id == user_id
                )
            )
        ).scalar_one_or_none()
        if photo_set is None:
            raise HTTPException(status_code=404, detail="no such photo set")

        avatar.photo_set_id = photo_set.id
        photo_set.avatar_id = avatar.id
        await db.commit()
        return _describe(avatar, attested)

    @router.post("/api/avatars/{avatar_id}/consent", status_code=201)
    async def record_consent(
        avatar_id: str,
        body: ConsentInput,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        """Record who authorised this recreation.

        Recreating yourself is verified immediately: the account holder is the
        rights holder and no third party's rights are engaged. Anybody else's
        likeness is recorded as pending and needs a human to read the evidence
        document, because a self-service route to 'verified' would make the
        gate decorative.
        """
        try:
            await assert_owned(db, avatar_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        existing = (
            await db.execute(
                select(ConsentRecord).where(ConsentRecord.avatar_id == avatar_id)
            )
        ).scalar_one_or_none()
        record = existing or ConsentRecord(avatar_id=avatar_id)

        relationship = body.relationship_to_subject.strip().lower()

        record.rights_holder_name = body.rights_holder_name.strip()
        record.relationship_to_subject = relationship
        record.jurisdiction = body.jurisdiction.strip()
        record.evidence_s3_key = body.evidence_key

        if relationship == SELF_RELATIONSHIP:
            # The account holder attesting that the subject is themselves. No
            # third party's rights are engaged, so there is nothing for a
            # reviewer to check.
            record.status = ConsentStatus.VERIFIED
            record.verified_at = datetime.now(UTC)
            record.verified_by = f"self-attested by {user_id}"
        else:
            # Somebody else's likeness. This needs a human to read the evidence
            # document; there is deliberately no self-service route to verified.
            record.status = ConsentStatus.PENDING

        if existing is None:
            db.add(record)
        await db.commit()

        return {
            "status": record.status.value,
            "avatar_id": avatar_id,
            "needs_review": record.status is ConsentStatus.PENDING,
        }

    return router
