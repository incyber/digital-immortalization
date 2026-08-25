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
    """Only VERIFIED opens a session.

    REVOKED exists as a distinct state from REJECTED because a rights-holder
    withdrawing permission is a supported operation with its own audit trail,
    not a review that went badly.
    """

    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    REVOKED = "revoked"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    avatars: Mapped[list[Avatar]] = relationship(back_populates="owner")


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

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    owner: Mapped[User] = relationship(back_populates="avatars")
    consent: Mapped[ConsentRecord | None] = relationship(
        back_populates="avatar", uselist=False
    )


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
