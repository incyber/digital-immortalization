"""Session issuance.

A session is a LiveKit room plus a token scoped to it. The token is minted only
after the consent gate passes, which is why this module imports the gate rather
than the other way round.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import Iterable
from datetime import timedelta

from livekit import api
from sqlalchemy.ext.asyncio import AsyncSession

from avatar.config import Settings
from avatar.gateway.consent import assert_consented
from avatar.gateway.demo import DEMO_EMAIL
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
    # Called for the check, not the row: the agent loads the character
    # itself from the database.
    await assert_owned(db, avatar_id, owner_id)
    consent = await assert_consented(db, avatar_id)

    room = f"call-{uuid.uuid4().hex[:12]}"
    session = Session(avatar_id=avatar_id, room_name=room)
    db.add(session)
    await db.commit()

    if dispatcher is not None:
        await dispatcher.dispatch(
            room,
            avatar_id,
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


def assert_production_ready(cfg: Settings, account_emails: Iterable[str] | None = None) -> None:
    """Refuse to start with development credentials.

    Called from create_app when PRODUCTION is set, and only then - so tests and
    local runs are unaffected while a cloud start with a leftover .env fails
    loudly instead of serving signed tokens anyone can forge.

    It used to be called from nowhere at all. It had passing unit tests, which
    proved its logic was right and said nothing about whether it ran, and an
    audit found it dead. That is the second time in this project a safety
    check has been described as protection while never executing, so the test
    that matters now is the one asserting create_app calls it.

    account_emails is the database's answer to "is anybody real here", passed
    through to assert_demo_mode_safe. It is optional because this function runs
    at construction, before there is a database to ask; the lifespan asks again
    with it filled in.
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
    assert_demo_mode_safe(cfg, account_emails)


# Printed rather than logged. Logging configuration is a deployment's business
# and can be turned down; this must be visible in whatever a container's
# console is, on every boot, for as long as the flag is on.
_DEMO_BANNER = """
================================================================================
DEMO_MODE IS ON.

Every visitor is signed into ONE SHARED ACCOUNT with no password. Everything
uploaded here - photographs, recordings, biographies - is readable by anyone
who has the link. This is not a private deployment and it must never be put in
front of a real family.

Turn it off by unsetting DEMO_MODE.
================================================================================
"""


def assert_demo_mode_safe(cfg: Settings, account_emails: Iterable[str] | None = None) -> None:
    """Refuse to run demo mode over anybody's real data.

    Two halves, because the dangerous condition is not visible in one place.

    What configuration alone can say is only that the flag is on, so that half
    is a banner on stderr - loud, on every boot, unmissable in a container's
    console. A flag that silently changes who can read a stranger's
    photographs must not be quiet.

    What actually decides safety is the database, so the caller that has read
    it passes the account list and this raises. "Any real customer data could
    exist" is, precisely, "an account exists that is not the demo account":
    every photograph, recording and consent record in this system hangs off an
    account, so no account means nothing of anyone's to expose. That check runs
    at startup (see create_app's lifespan) before the first request is served,
    and it runs whether or not PRODUCTION is set - a laptop holding a real
    family's uploads is not less serious for being a laptop.

    Passing no account list checks the configuration half only. That is what
    assert_production_ready does, because it is called before the database
    exists.
    """
    if not cfg.demo_mode:
        return

    print(_DEMO_BANNER, file=sys.stderr, flush=True)

    if account_emails is None:
        return

    real = sorted({address for address in account_emails if address != DEMO_EMAIL})
    if real:
        raise ValueError(
            f"DEMO_MODE is on and this database holds {len(real)} account(s) that "
            f"are not the demo account ({', '.join(real[:3])}"
            f"{', ...' if len(real) > 3 else ''}). Demo mode signs every visitor "
            "into one shared tenant, so starting here would hand a stranger's "
            "photographs and recordings to anyone with the link. Unset DEMO_MODE, "
            "or point this deployment at an empty database."
        )
