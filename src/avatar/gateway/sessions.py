"""Session issuance.

A session is a LiveKit room plus a token scoped to it. The token is minted only
after the consent gate passes, which is why this module imports the gate rather
than the other way round.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

from livekit import api
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.config import Settings
from avatar.gateway.consent import assert_consented
from avatar.gateway.dispatch import AgentDispatcher
from avatar.gateway.models import Session
from avatar.gateway.tenancy import assert_owned

# Long enough for a real conversation, short enough that a leaked token is not
# a standing invitation.
TOKEN_TTL = timedelta(hours=2)


def mint_token(
    cfg: Settings,
    room: str,
    identity: str,
    name: str,
    can_update_metadata: bool = False,
) -> str:
    """A JWT granting join rights to exactly one room.

    Scoped per room rather than per project: a token that leaked would open one
    conversation, not every conversation.

    can_update_metadata is granted only to the agent. It is what allows the
    synthetic-content declaration to be attached to the published stream, and
    a visitor has no reason to be able to rewrite it.
    """
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        can_update_own_metadata=can_update_metadata,
    )
    return (
        api.AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
        .with_identity(identity)
        .with_name(name)
        .with_grants(grants)
        .with_ttl(TOKEN_TTL)
        .to_jwt()
    )


async def open_session(
    db: AsyncSession,
    cfg: Settings,
    avatar_id: str,
    owner_id: str,
    dispatcher: AgentDispatcher | None = None,
) -> dict[str, str]:
    """Create a room, put an agent in it, and return joining details.

    Two independent gates, in this order:

      Ownership - is this avatar this tenant's business at all? Checked first,
      so a stranger probing avatar ids cannot learn which exist or what their
      consent status is.

      Consent - is a recreation permitted at all? Checked second.

    Neither implies the other. An avatar can be fully consented and belong to
    somebody else entirely. Nothing is written, no token exists, and no agent
    is dispatched unless both pass.
    """
    avatar = await assert_owned(db, avatar_id, owner_id)
    consent = await assert_consented(db, avatar_id)

    room = f"call-{uuid.uuid4().hex[:12]}"
    session = Session(avatar_id=avatar_id, room_name=room)
    db.add(session)
    await db.commit()

    if dispatcher is not None:
        await dispatcher.dispatch(
            room,
            avatar_id,
            avatar.profile_path,
            consent_record_id=consent.id,
            rights_holder=consent.rights_holder_name,
        )

    return {
        "session_id": session.id,
        "room": room,
        "url": cfg.livekit_url,
        "token": mint_token(cfg, room, identity=f"human-{session.id[:8]}", name="Visitor"),
        "avatar_id": avatar_id,
    }


# The LiveKit --dev server issues this pair. Convenient locally, a published
# credential anywhere else.
DEV_CREDENTIALS = {("devkey", "secret")}
MIN_SECRET_BYTES = 32


def assert_production_ready(cfg: Settings) -> None:
    """Refuse to start with development credentials.

    Called from the deployment entry point, not from create_app, so tests and
    local runs are unaffected while a cloud start with a leftover .env fails
    loudly instead of serving signed tokens anyone can forge.
    """
    if (cfg.livekit_api_key, cfg.livekit_api_secret) in DEV_CREDENTIALS:
        raise ValueError(
            "LIVEKIT_API_KEY/SECRET are the LiveKit development secret; "
            "tokens signed with it can be forged by anyone"
        )
    if len(cfg.livekit_api_secret.encode()) < MIN_SECRET_BYTES:
        raise ValueError(
            f"LIVEKIT_API_SECRET is shorter than {MIN_SECRET_BYTES} bytes, "
            "which is below the HS256 minimum"
        )
    if cfg.session_secret == Settings.model_fields["session_secret"].default:
        raise ValueError(
            "SESSION_SECRET is still the development default; anyone could "
            "forge a session cookie for any account"
        )
    if not cfg.cookies_secure:
        raise ValueError(
            "COOKIES_SECURE is false; session cookies would be sent over "
            "plaintext HTTP"
        )
