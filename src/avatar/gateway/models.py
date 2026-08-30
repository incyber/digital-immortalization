"""Persistence.

consent_records is the table the product turns on. Everything else here is
ordinary bookkeeping; that one is a legal control, and its status column is
read on every session request.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ConsentStatus(str, enum.Enum):
    """VERIFIED and SELF_ATTESTED both open a session; nothing else does.

    REVOKED exists as a distinct state from REJECTED because a rights-holder
    withdrawing permission is a supported operation with its own audit trail,
    not a review that went badly.

    SELF_ATTESTED exists as a distinct state from VERIFIED for the same
    reason: an account holder's claim to be the subject is a real, callable
    basis for a session, but it is not the thing a reviewer read and signed
    off on, and collapsing the two into one value would erase that
    difference from every audit trail that reads status back.
    """

    PENDING = "pending"
    VERIFIED = "verified"
    SELF_ATTESTED = "self_attested"
    REJECTED = "rejected"
    REVOKED = "revoked"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    avatars: Mapped[list[Avatar]] = relationship(back_populates="owner")


class BodyBuild(str, enum.Enum):
    """How heavily built the person was.

    Words rather than a number: a family can answer "solid" and cannot answer
    0.63, and an answer somebody can actually give is worth more than a scale
    they would have to guess at.
    """

    SLIGHT = "slight"
    AVERAGE = "average"
    SOLID = "solid"
    HEAVY = "heavy"


class Shoulders(str, enum.Enum):
    """Frame width, which build does not imply: a slight person can be broad."""

    NARROW = "narrow"
    AVERAGE = "average"
    BROAD = "broad"


class Posture(str, enum.Enum):
    """How they carried themselves, which relatives recognise before anything
    a tape measure would report."""

    UPRIGHT = "upright"
    RELAXED = "relaxed"
    STOOPED = "stooped"


class SpeechPace(str, enum.Enum):
    """How fast they talked.

    Reaches the prompt as an instruction about sentence length, which is the
    only place it can land: nothing in the speech pipeline takes a rate, so a
    dial claiming to slow the voice down would be a dial that does nothing.
    """

    SLOW = "slow"
    MEASURED = "measured"
    QUICK = "quick"


class Humour(str, enum.Enum):
    """Whether they joked, and how.

    NONE is a real answer and is why this is an enum rather than free text:
    "they were not a joker" has to be sayable, and it is a different state
    from an unanswered column. A recreation that quips at somebody's father's
    expense when he never did is the complaint that gets made out loud.
    """

    NONE = "none"
    DRY = "dry"
    WARM = "warm"
    PLAYFUL = "playful"


class Directness(str, enum.Enum):
    """Whether they said the thing or came at it sideways."""

    BLUNT = "blunt"
    PLAIN = "plain"
    GENTLE = "gentle"


# Volume is deliberately not a column here, although families describe it
# readily. Loudness comes from the voice recording, which is cloned; with no
# recording the stock voice has one loudness and nothing in synthesis takes a
# gain. A dial that changed neither the words nor the sound would be a
# question asked for nothing, and every question asked for nothing is one more
# field between a family and their father.


# Deliberately wide. The job is to refuse nonsense -- a zero, a typo with an
# extra digit -- not to second-guess a family, and the person being recreated
# is not always an adult. Turning away somebody's true number would be worse
# than accepting an unusual one.
MIN_HEIGHT_CM = 50
MAX_HEIGHT_CM = 250

# What the build falls back on for anything left blank. Spelled out here
# because "we were not told" and "they were average" must stay different
# states: a NULL column is an unanswered question, and this is the only place
# that turns one into a value. A zero standing in for either would quietly
# become a claim about how somebody looked.
NEUTRAL_HEIGHT_CM = 170
NEUTRAL_BUILD = BodyBuild.AVERAGE
NEUTRAL_SHOULDERS = Shoulders.AVERAGE
NEUTRAL_POSTURE = Posture.RELAXED


class Avatar(Base):
    """One recreated person, described entirely by the customer.

    Everything the persona needs lives here rather than in a file shipped with
    the application. There is no built-in character: an account with no avatars
    can do nothing until it creates one, which is the correct state for a
    product whose entire subject matter is supplied by its users.

    Two fields are deliberately not customer-editable and so are not columns:
    the synthetic-media disclosure, which is generated from display_name so it
    cannot be weakened or removed, and the crisis line, which is resolved from
    a verified registry by country rather than typed.
    """

    __tablename__ = "avatars"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)

    # Who this is. The customer's words, used verbatim in the prompt.
    display_name: Mapped[str] = mapped_column(String(255))
    locale: Mapped[str] = mapped_column(String(8), default="en")
    # Country decides which crisis line the guardrail speaks.
    country: Mapped[str] = mapped_column(String(2), default="US")
    biography: Mapped[str] = mapped_column(Text, default="")
    voice_description: Mapped[str] = mapped_column(Text, default="")
    # Optional. What the recreation should decline to do or claim.
    boundaries: Mapped[str] = mapped_column(Text, default="")

    # How the person came across, in the family's words. Every column below is
    # optional and empty by default, and an avatar with none of them answered
    # is built and spoken to exactly as it was before they existed.
    #
    # They are separate columns rather than more biography because each is
    # rendered differently into the prompt - see persona.py. A phrase has to
    # be quoted as an example, a subject to avoid has to be a trailing
    # instruction, and a pace has to become an instruction about sentence
    # length. None of that can be recovered from prose by a small model.
    #
    # Nothing here is ever pre-filled by the product. See gateway/defaults.py:
    # a starting tone is a default, a fact about a specific dead person is
    # not, and the boundary between them is enforced there rather than left to
    # whoever writes the form copy.

    # Phrases they actually used, as a JSON list. The single strongest field
    # of the set: a family recognises a saying instantly, and it is the one
    # thing a model can imitate literally. Also the most dangerous, which is
    # why the prompt frames it as examples - recited, it becomes a parody.
    characteristic_phrases: Mapped[str] = mapped_column(Text, default="")

    # Habits, verbal and physical. Shapes speech only: a described facial
    # mannerism does not reach the face. See MANNERISM_MOTION_LIMIT in
    # persona.py, which spells out exactly why and what it would take.
    mannerisms: Mapped[str] = mapped_column(Text, default="")

    # What they always steered back to. Changes what the recreation talks
    # about unprompted, which is most of what makes a conversation feel like
    # a person rather than a question-answering service.
    topics_loved: Mapped[str] = mapped_column(Text, default="")

    # What must not be raised. Kept apart from boundaries on purpose: that
    # column governs what the recreation may claim about itself, this one
    # governs what a grieving family cannot bear to hear, and they are written
    # by different people at different times.
    topics_to_avoid: Mapped[str] = mapped_column(Text, default="")

    # Who is on the other end - "his daughter", "her youngest son". Changes
    # the register of every sentence, and cannot be guessed: the account
    # holder is not always the person who will sit down in front of it.
    caller_relationship: Mapped[str] = mapped_column(String(255), default="")

    # The three dials. Nullable rather than empty-defaulted, because for these
    # "not stated" and a stated value are both meaningful and must not
    # collapse: humour NONE is "they did not joke", NULL is "nobody told us".
    speech_pace: Mapped[SpeechPace | None] = mapped_column(
        Enum(SpeechPace, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=True,
    )
    speech_humour: Mapped[Humour | None] = mapped_column(
        Enum(Humour, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=True,
    )
    speech_directness: Mapped[Directness | None] = mapped_column(
        Enum(Directness, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=True,
    )

    # Set once the photo set has trained and assets have been built.
    photo_set_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    assets_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    framing: Mapped[str] = mapped_column(String(16), default="head")

    # A recording of the person, used to clone their voice. Without it the
    # recreation speaks in a stock voice, which is the wrong person rather
    # than a lesser version of the right one.
    voice_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    voice_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    voice_quality: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Body shape, stated by the family and never estimated from the
    # photographs. No published method recovers a body from a head-and-
    # shoulders crop, which is what family photographs overwhelmingly are;
    # every shape estimator needs a visible torso. Somebody who knew the
    # person beats a guess from a head crop, and nothing about how they looked
    # is then settled anywhere except by a person who remembers them.
    #
    # Four fields, because each moves the silhouette in a way the others
    # cannot: height sets scale, build sets mass, shoulders set frame width,
    # and posture sets how they stood. Weight is not asked for -- it is
    # guessed badly and says nothing build does not.
    #
    # All four are nullable, and NULL means "not stated": never a zero, never
    # a neutral value posing as an answer. body_shape() below is the single
    # place a missing answer becomes a usable one.
    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    build: Mapped[BodyBuild | None] = mapped_column(
        Enum(BodyBuild, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=True,
    )
    shoulders: Mapped[Shoulders | None] = mapped_column(
        Enum(Shoulders, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=True,
    )
    posture: Mapped[Posture | None] = mapped_column(
        Enum(Posture, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=True,
    )

    # The Gaussian splat of this person, and what must be said about it.
    #
    # These live on the avatar rather than on the build job because the job is
    # over in minutes and the disclosure is forever. A family reads "about 40%
    # of this likeness was generated" on a page they open weeks later, not in
    # the response to the request that started the build, so the sentence and
    # the number that backs it are stored with the person they describe.
    #
    # splat_measured_fraction is nullable and never defaulted to zero: NULL is
    # "no splat has been built", and 0.0 would be a claim that none of the
    # likeness was measured. The two must not be the same value.
    splat_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # "reconstruct" or "generate". See avatar/splat/routes.py.
    splat_route: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Why that route, in the customer's language.
    splat_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The sentence the customer is shown. Computed by the quality report from
    # the route, never written by hand, so it cannot be softened.
    splat_disclosure: Mapped[str | None] = mapped_column(Text, nullable=True)
    splat_measured_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    # What is weak about this build, as a JSON list. Support reads it when a
    # family says it does not look like him.
    splat_concerns: Mapped[str | None] = mapped_column(Text, nullable=True)
    splat_gaussians: Mapped[int] = mapped_column(Integer, default=0)
    # The customer's download: the splat renders in their own browser, so a
    # file a phone will not fetch is a likeness nobody sees.
    splat_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    splat_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    splat_built_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    owner: Mapped[User] = relationship(back_populates="avatars")
    consent: Mapped[ConsentRecord | None] = relationship(
        back_populates="avatar", uselist=False
    )


def body_shape(avatar: Avatar) -> dict:
    """What the family stated, alongside what the build will actually use.

    Both halves are reported rather than one merged answer, because they
    settle different questions: what we were told, and what gets made from it.
    Merged, a neutral default reads exactly like an answer -- and being able
    to say which is which is the whole reason a person is asked instead of
    something guessing from a photograph.
    """
    stated = {
        "height_cm": avatar.height_cm,
        "build": avatar.build.value if avatar.build is not None else None,
        "shoulders": avatar.shoulders.value if avatar.shoulders is not None else None,
        "posture": avatar.posture.value if avatar.posture is not None else None,
    }
    return {
        "stated": stated,
        "in_use": {
            "height_cm": (
                avatar.height_cm if avatar.height_cm is not None else NEUTRAL_HEIGHT_CM
            ),
            "build": (avatar.build or NEUTRAL_BUILD).value,
            "shoulders": (avatar.shoulders or NEUTRAL_SHOULDERS).value,
            "posture": (avatar.posture or NEUTRAL_POSTURE).value,
        },
    }


class ConsentRecord(Base):
    """Documented permission from whoever holds the rights.

    evidence_s3_key is not optional in practice: a checkbox is not evidence,
    and the statutes that make this necessary contemplate a rights-holder who
    can later be asked to confirm.
    """

    __tablename__ = "consent_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(ForeignKey("avatars.id"), unique=True, index=True)
    rights_holder_name: Mapped[str] = mapped_column(String(255))
    relationship_to_subject: Mapped[str] = mapped_column(String(128))
    jurisdiction: Mapped[str] = mapped_column(String(64))
    evidence_s3_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[ConsentStatus] = mapped_column(
        Enum(ConsentStatus, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        default=ConsentStatus.PENDING,
        index=True,
    )
    signed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    avatar: Mapped[Avatar] = relationship(back_populates="consent")


class PhotoSetStatus(str, enum.Enum):
    """A photo set's progress from upload to a trained likeness."""

    UPLOADING = "uploading"
    READY = "ready"          # passed validation, awaiting training
    REJECTED = "rejected"    # failed validation; the customer must add more
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class PhotoSet(Base):
    """One batch of photographs uploaded to build one avatar.

    Carries owner_id directly rather than only through the avatar, because it
    exists before an avatar does and must still be tenant-scoped while it is
    being uploaded.
    """

    __tablename__ = "photo_sets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    avatar_id: Mapped[str | None] = mapped_column(
        ForeignKey("avatars.id"), nullable=True, index=True
    )
    status: Mapped[PhotoSetStatus] = mapped_column(
        Enum(PhotoSetStatus, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        default=PhotoSetStatus.UPLOADING,
        index=True,
    )
    usable_count: Mapped[int] = mapped_column(Integer, default=0)
    half_body_count: Mapped[int] = mapped_column(Integer, default=0)
    # "head" or "half_body", decided by what the photographs show rather than
    # demanded of the customer. See ingest/validate.py.
    framing: Mapped[str] = mapped_column(String(16), default="head")
    problems: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Per-requirement progress, so the upload page can show which condition is
    # unmet and by how much rather than only a pass or fail.
    requirements_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Photo(Base):
    """One uploaded image and what validation made of it."""

    __tablename__ = "photos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    photo_set_id: Mapped[str] = mapped_column(ForeignKey("photo_sets.id"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    # Storage key, always inside the owner's prefix. See storage/keys.py.
    blob_key: Mapped[str] = mapped_column(String(512))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    accepted: Mapped[bool] = mapped_column(default=False)
    rejection_reasons: Mapped[str | None] = mapped_column(Text, nullable=True)
    face_height_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TrainingStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(Base):
    """One identity-training run over a photo set.

    Kept as a row rather than only in the job runner so that a customer can be
    told what is happening after they close the tab, and so a crashed runner
    leaves evidence rather than a silently stuck avatar.
    """

    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    photo_set_id: Mapped[str] = mapped_column(ForeignKey("photo_sets.id"), index=True)
    status: Mapped[TrainingStatus] = mapped_column(
        Enum(TrainingStatus, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        default=TrainingStatus.QUEUED,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(64), default="local")
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    output_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class VideoIngestJob(Base):
    """One uploaded clip being turned into frames.

    A row rather than only a task in memory, for the same two reasons the
    training and splat jobs are rows: a family who closes the tab can still be
    told what happened, and a gateway that dies mid-job leaves evidence rather
    than a bar stuck at forty percent forever.

    A separate table rather than three more columns on training_jobs. Schema
    is created with metadata.create_all, which adds missing tables and does
    not add missing columns, so widening a table that already exists in the
    deployed database would reach every fresh checkout and no live one - and
    the first read of the new column would be a 500 on the running product.
    A new table is applied by exactly the same call.

    The counts are the point of it. "Forty of sixty frames, thirty-one usable"
    is something a person can watch; a spinner is not, and a spinner is what
    the customer complained about.
    """

    __tablename__ = "video_ingest_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    photo_set_id: Mapped[str] = mapped_column(ForeignKey("photo_sets.id"), index=True)
    status: Mapped[TrainingStatus] = mapped_column(
        Enum(TrainingStatus, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        default=TrainingStatus.QUEUED,
        index=True,
    )
    # The customer's own filename, for display. The storage key is generated.
    filename: Mapped[str] = mapped_column(String(255), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    # How many frames the clip will yield, known from container metadata
    # before any decoding, so the bar has a denominator from the start. Zero
    # means the clip has not been read yet.
    frames_planned: Mapped[int] = mapped_column(Integer, default=0)
    # How many have been checked so far, and how many of those a face was
    # found in. Both are written as the work proceeds, not at the end.
    frames_examined: Mapped[int] = mapped_column(Integer, default=0)
    frames_usable: Mapped[int] = mapped_column(Integer, default=0)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    avatar_id: Mapped[str] = mapped_column(ForeignKey("avatars.id"), index=True)
    room_name: Mapped[str] = mapped_column(String(128), unique=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SceneObservation(Base):
    """What the camera showed, and what it cost."""

    __tablename__ = "scene_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    description: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(128))
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)


class SafetyEvent(Base):
    """Every crisis-guardrail trigger. Append-only in practice."""

    __tablename__ = "safety_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    matched_term: Mapped[str] = mapped_column(String(128))
    locale: Mapped[str] = mapped_column(String(8))
    transcript_excerpt: Mapped[str] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
