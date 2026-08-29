"""Avatar creation and management.

An account starts with nothing. Every character is described and named by its
owner; the application ships with none.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.models import (
    MAX_HEIGHT_CM,
    MIN_HEIGHT_CM,
    Avatar,
    BodyBuild,
    ConsentRecord,
    ConsentStatus,
    PhotoSet,
    Posture,
    Shoulders,
    body_shape,
)
from avatar.gateway.tenancy import TenantError, assert_owned, owned_query
from avatar.ingest.voice import ACCEPTED_TYPES, inspect_voice, normalise
from avatar.persona import InvalidProfile, persona_from_avatar
from avatar.safety.crisis_lines import UnsupportedCountry, parse_attested, selectable
from avatar.services.voices import UnsupportedLocale, supported
from avatar.storage.keys import voice_key

# Two minutes of audio at any sane bitrate. Past this something other than a
# voice reference is being uploaded.
MAX_VOICE_BYTES = 40 * 1024 * 1024


class AvatarInput(BaseModel):
    """What the customer supplies. Notably absent: the disclosure text and the
    crisis line, neither of which they get to write."""

    display_name: str = Field(min_length=1, max_length=255)
    locale: str = Field(default="en", max_length=8)
    country: str = Field(min_length=2, max_length=2)
    biography: str = Field(min_length=1)
    voice_description: str = ""
    boundaries: str = ""

    # What the person's body was like, in the family's own words. Nothing here
    # is required: somebody who does not know, or does not want to be asked
    # this today, still gets an avatar. An unanswered field stays unanswered
    # in the database and is resolved to a neutral value at build time, which
    # is a different thing from a machine deciding an answer for them.
    #
    # Anything outside these words, or a height no person has had, is refused
    # rather than nudged into range: silently rewriting a family's answer is
    # the one behaviour that would defeat the point of asking.
    height_cm: int | None = Field(default=None, ge=MIN_HEIGHT_CM, le=MAX_HEIGHT_CM)
    build: BodyBuild | None = None
    shoulders: Shoulders | None = None
    posture: Posture | None = None


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


_BODY_FIELDS = ("height_cm", "build", "shoulders", "posture")


def _apply_body(avatar: Avatar, body: AvatarInput) -> None:
    """Copy across only the body attributes this request actually carried.

    A field the request left out leaves what is stored alone, so a client that
    predates these questions cannot wipe out what a family took the trouble to
    tell us. Clearing one means sending it as null, which is somebody deciding
    rather than somebody omitting.
    """
    for field in _BODY_FIELDS:
        if field in body.model_fields_set:
            setattr(avatar, field, getattr(body, field))


def _describe(avatar: Avatar, attested: frozenset[str]) -> dict:
    try:
        persona = persona_from_avatar(avatar, attested)
        disclosure = persona.disclosure
        crisis = {"name": persona.crisis_line.name, "number": persona.crisis_line.number}
        usable = True
    except (InvalidProfile, UnsupportedCountry, UnsupportedLocale) as exc:
        disclosure, crisis, usable = str(exc), None, False

    return {
        "id": avatar.id,
        "display_name": avatar.display_name,
        "locale": avatar.locale,
        "country": avatar.country,
        "biography": avatar.biography,
        "voice_description": avatar.voice_description,
        "boundaries": avatar.boundaries,
        "body": body_shape(avatar),
        "photo_set_id": avatar.photo_set_id,
        "has_assets": bool(avatar.assets_key),
        "has_voice": bool(avatar.voice_key),
        "voice_quality": avatar.voice_quality,
        "disclosure": disclosure,
        "crisis_line": crisis,
        "callable": usable and bool(avatar.assets_key),
    }


def build_router(settings, current_user, get_db, store) -> APIRouter:
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

    @router.get("/api/languages")
    async def languages():
        """Languages an avatar can speak.

        Served rather than typed. A free-text language field is how an avatar
        ends up recorded as "SPANISH", falling back to English prompts while a
        Spanish voice reads them aloud.
        """
        return {
            "languages": [
                {"code": v.locale, "name": v.name, "voice": v.piper_voice}
                for v in supported()
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
        _apply_body(avatar, body)

        # Validated before it is stored, so an avatar that could never be
        # spoken to is never created in the first place.
        try:
            persona_from_avatar(avatar, attested)
        except (InvalidProfile, UnsupportedCountry, UnsupportedLocale) as exc:
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
        _apply_body(avatar, body)

        try:
            persona_from_avatar(avatar, attested)
        except (InvalidProfile, UnsupportedCountry, UnsupportedLocale) as exc:
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

    @router.post("/api/avatars/{avatar_id}/voice", status_code=201)
    async def upload_voice(
        avatar_id: str,
        file: UploadFile = File(...),  # noqa: B008
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        """Attach a recording of the person, for cloning their voice.

        Checked and normalised here rather than at synthesis time, so a
        customer learns their voicemail is too quiet while they can still look
        for a better file.
        """
        try:
            avatar = await assert_owned(db, avatar_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if file.content_type not in ACCEPTED_TYPES:
            raise HTTPException(
                status_code=400, detail=f"unsupported audio type {file.content_type!r}"
            )

        data = await file.read()
        if len(data) > MAX_VOICE_BYTES:
            raise HTTPException(status_code=400, detail="that recording is too large")

        verdict = inspect_voice(data)
        if not verdict.usable:
            raise HTTPException(
                status_code=400,
                detail="; ".join(p.value for p in verdict.problems),
            )

        stored = await store.put(
            user_id, voice_key(user_id, avatar_id), normalise(data), "audio/wav"
        )

        avatar.voice_key = stored.key
        avatar.voice_seconds = verdict.duration_s
        avatar.voice_quality = verdict.quality
        await db.commit()

        return {
            "avatar_id": avatar_id,
            "seconds": round(verdict.duration_s, 1),
            "sample_rate": verdict.sample_rate,
            "quality": verdict.quality,
        }

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
