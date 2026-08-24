"""Persistence.

consent_records is the table the product turns on. Everything else here is
ordinary bookkeeping; that one is a legal control, and its status column is
read on every session request.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


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

    avatars: Mapped[list["Avatar"]] = relationship(back_populates="owner")


class Avatar(Base):
    """One recreated person."""

    __tablename__ = "avatars"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    locale: Mapped[str] = mapped_column(String(8), default="en")
    profile_path: Mapped[str] = mapped_column(String(512))
    assets_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    owner: Mapped[User] = relationship(back_populates="avatars")
    consent: Mapped["ConsentRecord | None"] = relationship(
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
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    description: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(128))
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)


class SafetyEvent(Base):
    """Every crisis-guardrail trigger. Append-only in practice."""

    __tablename__ = "safety_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    matched_term: Mapped[str] = mapped_column(String(128))
    locale: Mapped[str] = mapped_column(String(8))
    transcript_excerpt: Mapped[str] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
