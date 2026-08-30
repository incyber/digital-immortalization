"""Avatar creation and management.

An account starts with nothing. Every character is described and named by its
owner; the application ships with none.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.gateway.csrf import require_same_site_header
from avatar.gateway.defaults import defaults_payload
from avatar.gateway.erasure import erase_account, erase_avatar
from avatar.gateway.models import (
    MAX_HEIGHT_CM,
    MIN_HEIGHT_CM,
    Avatar,
    BodyBuild,
    ConsentRecord,
    ConsentStatus,
    Directness,
    Humour,
    PhotoSet,
    Posture,
    Shoulders,
    SpeechPace,
    body_shape,
)
from avatar.gateway.tenancy import TenantError, assert_owned, owned_query
from avatar.ingest.voice import ACCEPTED_TYPES, inspect_voice, normalise
from avatar.persona import (
    MAX_PHRASES,
    InvalidProfile,
    decode_phrases,
    encode_phrases,
    persona_from_avatar,
)
from avatar.safety.crisis_lines import UnsupportedCountry, parse_attested, selectable
from avatar.services.voices import UnsupportedLocale, supported
from avatar.storage.keys import belongs_to, voice_key

# Two minutes of audio at any sane bitrate. Past this something other than a
# voice reference is being uploaded.
MAX_VOICE_BYTES = 40 * 1024 * 1024

# Long enough for anything anybody actually said out loud, short enough that
# five of them together cannot crowd out the rest of the prompt.
MAX_PHRASE_CHARS = 160


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

    # How the person came across. Optional in exactly the way the body
    # questions are: a family that answers none of it gets the same avatar it
    # would have got before any of these existed.
    #
    # The lengths are limits on the prompt, not opinions about how much
    # somebody may say about their father. A small model handed a long prompt
    # stops speaking in character and starts reporting its instructions - see
    # persona.py - so the budget is real. It is enforced here, where an
    # over-long answer is refused while the person can still shorten it,
    # rather than by truncating their words somewhere they will never see.
    characteristic_phrases: list[str] | None = Field(default=None, max_length=MAX_PHRASES)
    mannerisms: str | None = Field(default=None, max_length=600)
    topics_loved: str | None = Field(default=None, max_length=400)
    topics_to_avoid: str | None = Field(default=None, max_length=400)
    caller_relationship: str | None = Field(default=None, max_length=120)
    speech_pace: SpeechPace | None = None
    speech_humour: Humour | None = None
    speech_directness: Directness | None = None

    @field_validator("characteristic_phrases")
    @classmethod
    def _phrases_are_sayings_not_paragraphs(cls, value: list[str] | None) -> list[str] | None:
        """Refuse a phrase that is really a monologue.

        A saying is short by definition, and something long pasted in here is
        a biography in the wrong box - where it would be quoted back verbatim
        as a line the person used to say.
        """
        if value is None:
            return None
        cleaned = [phrase.strip() for phrase in value if phrase and phrase.strip()]
        for phrase in cleaned:
            if len(phrase) > MAX_PHRASE_CHARS:
                raise ValueError(
                    "a characteristic phrase is something they said, not a "
                    f"paragraph; keep each one under {MAX_PHRASE_CHARS} characters"
                )
        return cleaned


class ConsentInput(BaseModel):
    rights_holder_name: str = Field(min_length=1, max_length=255)
    relationship_to_subject: str = Field(min_length=1, max_length=128)
    jurisdiction: str = Field(min_length=1, max_length=64)
    evidence_key: str | None = None


# Recreating yourself is a real, reasonable case: nobody needs a reviewer's
# permission to be themselves. But "I am the subject" is a claim the account
# holder makes about themselves, not something a third party attested to and
# a human read evidence for - see ConsentStatus.SELF_ATTESTED, the status
# this relationship produces. Anything else is arbitrary client text and gets
# no special treatment at all: the only thing that ever bypasses review is
# this exact word.
SELF_RELATIONSHIP = "self"


_BODY_FIELDS = ("height_cm", "build", "shoulders", "posture")


# Stored as text, so an omitted field leaves what is there and an explicit
# null clears it back to unanswered.
_MANNER_TEXT_FIELDS = (
    "mannerisms",
    "topics_loved",
    "topics_to_avoid",
    "caller_relationship",
)

# Stored as enums, where null is itself the unanswered state.
_MANNER_CHOICE_FIELDS = ("speech_pace", "speech_humour", "speech_directness")


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


def _apply_manner(avatar: Avatar, body: AvatarInput) -> None:
    """Copy across only the manner this request actually carried.

    Same rule as _apply_body, and it matters more here: these are the fields a
    family fills in slowly, weeks apart, as they remember things. A client
    that does not know about a field must not be able to erase it by not
    sending it.
    """
    for name in _MANNER_TEXT_FIELDS:
        if name in body.model_fields_set:
            value = getattr(body, name)
            setattr(avatar, name, (value or "").strip())

    for name in _MANNER_CHOICE_FIELDS:
        if name in body.model_fields_set:
            setattr(avatar, name, getattr(body, name))

    if "characteristic_phrases" in body.model_fields_set:
        avatar.characteristic_phrases = encode_phrases(body.characteristic_phrases)


def _manner(avatar: Avatar) -> dict:
    """What the family said about how the person came across.

    Read back in the shape it was sent, so a form can be reopened on it. The
    dials report null for unanswered, which stays distinct from humour "none"
    - "nobody told us" and "they did not joke" are different facts about
    somebody's father.
    """
    return {
        "characteristic_phrases": list(decode_phrases(avatar.characteristic_phrases)),
        "mannerisms": avatar.mannerisms,
        "topics_loved": avatar.topics_loved,
        "topics_to_avoid": avatar.topics_to_avoid,
        "caller_relationship": avatar.caller_relationship,
        "speech_pace": avatar.speech_pace.value if avatar.speech_pace else None,
        "speech_humour": avatar.speech_humour.value if avatar.speech_humour else None,
        "speech_directness": (
            avatar.speech_directness.value if avatar.speech_directness else None
        ),
    }


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
        "manner": _manner(avatar),
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

    @router.get("/api/avatars/defaults")
    async def defaults(request: Request):
        """What the create form can be opened with.

        Declared before /api/avatars/{avatar_id} because routes are matched in
        order and "defaults" would otherwise be read as an avatar id.

        Unauthenticated, like /api/countries and /api/languages beside it: it
        describes the form rather than anybody's data, and it must be
        answerable before an account has anything in it.

        Everything here shapes tone or comes from the request. Nothing here is
        a fact about a person - see defaults.py, where that boundary is a set
        of field names checked in code rather than a line in the copy.
        """
        return defaults_payload(request.headers, attested)

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
        _apply_manner(avatar, body)

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
        _apply_manner(avatar, body)

        try:
            persona_from_avatar(avatar, attested)
        except (InvalidProfile, UnsupportedCountry, UnsupportedLocale) as exc:
            await db.rollback()
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        await db.commit()
        return _describe(avatar, attested)

    @router.post(
        "/api/avatars/{avatar_id}/photo-set/{photo_set_id}",
        dependencies=[Depends(require_same_site_header)],
    )
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

        # Both of these off the event loop. inspect_voice shells out to
        # ffprobe and to ffmpeg's volumedetect, and normalise runs a second
        # ffmpeg pass with loudnorm over the whole recording: seconds of
        # blocking on a voicemail, minutes on a five-minute file. Run inline
        # they stopped every other request in the process, which is the same
        # bug the video endpoint had.
        #
        # Deliberately still awaited by the request rather than turned into a
        # job. One recording is one ffmpeg pass, not sixty face detections,
        # and the customer needs the verdict - too quiet, clipped, too short -
        # while they can still look for a better file. What was wrong was
        # where it ran, not that the caller waits for it.
        verdict = await asyncio.to_thread(inspect_voice, data)
        if not verdict.usable:
            raise HTTPException(
                status_code=400,
                detail="; ".join(p.value for p in verdict.problems),
            )

        normalised = await asyncio.to_thread(normalise, data)
        stored = await store.put(
            user_id, voice_key(user_id, avatar_id), normalised, "audio/wav"
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

    @router.delete("/api/avatars/{avatar_id}")
    async def erase(
        avatar_id: str,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        """Remove a recreated person: images, voice, likeness, consent, calls.

        A count is returned rather than a status. Somebody asking for their
        father's photographs to be erased is owed a number, and support is owed
        something they can check against a bucket.
        """
        try:
            result = await erase_avatar(db, store, avatar_id, user_id)
        except TenantError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return {
            "avatar_id": avatar_id,
            "complete": result.complete,
            "photographs": result.photos,
            "files": result.blobs,
            "calls": result.sessions,
            # Present when something refused to go. Reporting a partial
            # erasure as done is the one outcome that cannot be corrected
            # afterwards, because nobody knows to look.
            "failures": result.failures,
        }

    @router.delete("/api/account")
    async def erase_everything(
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        """Remove the account and everything in it."""
        result = await erase_account(db, store, user_id)
        return {
            "complete": result.complete,
            "avatars": result.avatars,
            "photographs": result.photos,
            "files": result.blobs,
            "failures": result.failures,
        }

    @router.post("/api/avatars/{avatar_id}/consent", status_code=201)
    async def record_consent(
        avatar_id: str,
        body: ConsentInput,
        db: AsyncSession = Depends(get_db),  # noqa: B008
        user_id: str = Depends(current_user),
    ):
        """Record who authorised this recreation.

        Recreating yourself is callable immediately, recorded as a claim
        (SELF_ATTESTED) rather than as a reviewed verification: the account
        holder is asserting they are the subject, and nobody needs to review
        a person's permission to be themselves. Anybody else's likeness is
        recorded as pending and needs a human to read the evidence document,
        because a self-service route to VERIFIED would make the gate
        decorative.
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

        # evidence_key is client-supplied text; the only guarantee worth
        # anything is that it names an object inside this tenant's own prefix,
        # checked the same way every other tenant-scoped key is - see
        # storage/keys.py. Anything else is either a bug or an attempt to
        # attach somebody else's evidence document to this consent record.
        evidence_key = body.evidence_key
        if evidence_key is not None and not belongs_to(evidence_key, user_id):
            raise HTTPException(
                status_code=400, detail="evidence key does not belong to this tenant"
            )

        record.rights_holder_name = body.rights_holder_name.strip()
        record.relationship_to_subject = relationship
        record.jurisdiction = body.jurisdiction.strip()
        record.evidence_s3_key = evidence_key

        if relationship == SELF_RELATIONSHIP:
            # The account holder claiming to be the subject. Real and
            # callable, but a self-attestation is not a reviewed
            # verification - it stays a distinguishable status rather than
            # being folded into VERIFIED, so an operator auditing consent
            # records can always tell which is which.
            record.status = ConsentStatus.SELF_ATTESTED
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
