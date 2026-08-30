import pytest

from avatar.gateway.consent import ConsentError
from avatar.gateway.sessions import mint_token, open_session


async def test_open_session_refuses_without_consent(db, cfg, avatar, owner):
    with pytest.raises(ConsentError):
        await open_session(db, cfg, avatar.id, owner.id)


async def test_open_session_writes_nothing_when_refused(db, cfg, avatar, owner):
    from sqlalchemy import func, select

    from avatar.gateway.models import Session

    with pytest.raises(ConsentError):
        await open_session(db, cfg, avatar.id, owner.id)
    count = (await db.execute(select(func.count()).select_from(Session))).scalar_one()
    assert count == 0, "a refused request must leave no session row behind"


async def test_open_session_returns_joinable_details(db, cfg, callable_avatar, owner):
    out = await open_session(db, cfg, callable_avatar.id, owner.id)
    assert out["url"].startswith("ws")
    assert out["room"].startswith("call-")
    assert len(out["token"].split(".")) == 3


async def test_token_is_scoped_to_one_room(cfg):
    import jwt

    token = mint_token(cfg, room="call-abc", identity="human-1", name="Visitor")
    claims = jwt.decode(token, cfg.livekit_api_secret, algorithms=["HS256"])
    assert claims["video"]["room"] == "call-abc"
    assert claims["video"]["roomJoin"] is True


async def test_each_session_gets_a_distinct_room(db, cfg, callable_avatar, owner):
    a = await open_session(db, cfg, callable_avatar.id, owner.id)
    b = await open_session(db, cfg, callable_avatar.id, owner.id)
    assert a["room"] != b["room"]


async def test_dev_secret_is_rejected_outside_development(cfg):
    # The LiveKit dev server ships a 6-byte secret. It is fine on localhost and
    # a real weakness anywhere else, so shipping it to cloud must be a failure
    # rather than a warning nobody reads.
    import pytest

    from avatar.gateway.sessions import assert_production_ready

    with pytest.raises(ValueError, match="development secret"):
        assert_production_ready(cfg)


async def test_a_self_attested_avatar_is_callable(db, cfg, avatar, owner):
    # The honest case - recreating yourself - must not dead-end behind a
    # review that a self-attestation deliberately skips.
    from tests.gateway.helpers import set_status

    await set_status(db, avatar, "self_attested")
    avatar.splat_key = f"tenants/t/avatars/{avatar.id}/avatar.splat"
    await db.commit()
    out = await open_session(db, cfg, avatar.id, owner.id)
    assert out["room"].startswith("call-")


async def test_agent_is_dispatched_for_a_self_attested_avatar(db, cfg, avatar, owner):
    from avatar.gateway.dispatch import NullDispatcher
    from tests.gateway.helpers import set_status

    await set_status(db, avatar, "self_attested")
    avatar.splat_key = f"tenants/t/avatars/{avatar.id}/avatar.splat"
    await db.commit()
    dispatcher = NullDispatcher()
    out = await open_session(db, cfg, avatar.id, owner.id, dispatcher)
    assert dispatcher.calls == [(out["room"], avatar.id)]


async def test_agent_is_dispatched_into_the_room(db, cfg, callable_avatar, owner):
    from avatar.gateway.dispatch import NullDispatcher

    dispatcher = NullDispatcher()
    out = await open_session(db, cfg, callable_avatar.id, owner.id, dispatcher)
    assert dispatcher.calls == [(out["room"], callable_avatar.id)]


async def test_no_agent_is_dispatched_when_consent_is_refused(db, cfg, avatar, owner):
    from avatar.gateway.dispatch import NullDispatcher

    dispatcher = NullDispatcher()
    with pytest.raises(ConsentError):
        await open_session(db, cfg, avatar.id, owner.id, dispatcher)
    assert dispatcher.calls == [], "a refused call must never put an agent in a room"


async def test_visitor_tokens_cannot_rewrite_the_synthetic_declaration(cfg):
    # The declaration says the stream is AI-generated. A visitor able to edit
    # participant metadata could remove it.
    import jwt

    token = mint_token(cfg, room="call-abc", identity="human-1", name="Visitor")
    claims = jwt.decode(token, cfg.livekit_api_secret, algorithms=["HS256"])
    assert not claims["video"].get("canUpdateOwnMetadata", False)


async def test_agent_tokens_may_publish_the_declaration(cfg):
    import jwt

    token = mint_token(
        cfg, room="call-abc", identity="avatar-x", name="Avatar", can_update_metadata=True
    )
    claims = jwt.decode(token, cfg.livekit_api_secret, algorithms=["HS256"])
    assert claims["video"]["canUpdateOwnMetadata"] is True


async def test_a_stranger_cannot_open_a_session_on_someone_elses_avatar(
    db, cfg, callable_avatar, other_owner
):
    # The avatar is real and fully consented. It is still not theirs.
    from avatar.gateway.dispatch import NullDispatcher
    from avatar.gateway.tenancy import TenantError

    dispatcher = NullDispatcher()
    with pytest.raises(TenantError):
        await open_session(db, cfg, callable_avatar.id, other_owner.id, dispatcher)
    assert dispatcher.calls == [], "no agent may be dispatched for another tenant"


async def test_ownership_is_checked_before_consent(db, cfg, avatar, other_owner):
    # avatar is pending consent AND owned by someone else. The refusal must be
    # the ownership one, so consent status does not leak to a stranger.
    from avatar.gateway.tenancy import TenantError

    with pytest.raises(TenantError):
        await open_session(db, cfg, avatar.id, other_owner.id)


async def test_development_session_secret_is_rejected_in_production(cfg):
    from avatar.gateway.sessions import assert_production_ready

    cfg.livekit_api_key = "real-key"
    cfg.livekit_api_secret = "x" * 40
    with pytest.raises(ValueError, match="SESSION_SECRET"):
        assert_production_ready(cfg)


async def test_insecure_cookies_are_rejected_in_production(cfg):
    from avatar.gateway.sessions import assert_production_ready

    cfg.livekit_api_key = "real-key"
    cfg.livekit_api_secret = "x" * 40
    cfg.session_secret = "a-real-and-sufficiently-long-session-secret"
    cfg.cookies_secure = False
    with pytest.raises(ValueError, match="COOKIES_SECURE"):
        assert_production_ready(cfg)


def _register_and_read_the_cookie(cfg) -> str:
    from fastapi.testclient import TestClient

    from avatar.gateway.app import create_app

    with TestClient(create_app(cfg)) as client:
        response = client.post(
            "/api/auth/register",
            json={"email": "cross@example.com", "password": "correct-horse-battery"},
        )
        assert response.status_code in (200, 201), response.text
        return response.headers["set-cookie"].lower()


def test_the_deployed_session_cookie_is_first_party():
    """One origin serves the site and this API, so the cookie is first-party.

    This used to be samesite=none, because the site was on one domain and the
    API on another. Safari blocks third-party cookies outright and Chrome
    blocks them in common configurations, so sign-in returned 200, the cookie
    was discarded, and the app bounced back to the sign-in page forever. The
    fix was to serve both from this process; lax is the cookie that follows
    from it, and it is the one a browser will actually keep.
    """
    from avatar.config import Settings

    cfg = Settings(_env_file=None, cookies_secure=True, database_url="sqlite+aiosqlite:///:memory:")
    header = _register_and_read_the_cookie(cfg)

    assert "samesite=lax" in header
    assert "secure" in header


def test_a_split_origin_deployment_can_still_ask_for_a_cross_site_cookie():
    """The escape hatch is real, because somebody will need it again.

    A site hosted somewhere other than this process is cross-site by
    definition, and a lax cookie is simply never sent there.
    """
    from avatar.config import Settings

    cfg = Settings(
        _env_file=None,
        cookies_secure=True,
        cookie_samesite="none",
        database_url="sqlite+aiosqlite:///:memory:",
    )
    header = _register_and_read_the_cookie(cfg)

    assert "samesite=none" in header
    assert "secure" in header


def test_samesite_none_is_downgraded_when_the_cookie_is_not_secure():
    """Browsers discard SameSite=None without Secure, silently.

    Honouring the setting over plain http would produce no cookie at all and a
    failure that looks like the server. Lax at least works.
    """
    from avatar.config import Settings

    cfg = Settings(
        _env_file=None,
        cookies_secure=False,
        cookie_samesite="none",
        database_url="sqlite+aiosqlite:///:memory:",
    )
    header = _register_and_read_the_cookie(cfg)

    assert "samesite=lax" in header


def test_create_app_refuses_to_start_in_production_with_dev_credentials():
    """The check used to exist and never run.

    It had passing unit tests proving its logic, which said nothing about
    whether anything called it. An audit found it dead. This is the test that
    would have caught that, and it asserts the wiring rather than the logic.
    """
    from avatar.config import Settings
    from avatar.gateway.app import create_app

    cfg = Settings(
        _env_file=None,
        production=True,
        livekit_api_key="devkey",
        livekit_api_secret="secret",
    )

    with pytest.raises(ValueError, match="development secret"):
        create_app(cfg)


def test_create_app_starts_normally_when_not_in_production():
    """Local runs and tests must be unaffected, or the check gets removed."""
    from avatar.config import Settings
    from avatar.gateway.app import create_app

    assert create_app(Settings(_env_file=None, production=False)) is not None


def test_the_browser_may_use_the_methods_the_product_exposes():
    """Deleting a photo set is a DELETE, and editing an avatar is a PATCH.

    Neither is a CORS simple method, so a browser preflights them. Omitting
    them from allow_methods made both fail silently cross-origin, without ever
    reaching this process.
    """
    from avatar.config import Settings
    from avatar.gateway.app import create_app

    app = create_app(Settings(_env_file=None))
    cors = next(
        m for m in app.user_middleware if "CORSMiddleware" in str(m.cls)
    )
    allowed = set(cors.kwargs["allow_methods"])

    assert {"GET", "POST", "PATCH", "DELETE"} <= allowed


async def test_production_refuses_the_placeholder_splat_backend(cfg):
    """A placeholder likeness is worse in production than no product at all.

    The fake backend exists so the build pipeline can be tested without a GPU.
    It writes something splat-shaped that is not anybody, and the call gate
    cannot tell the difference - `splat_key` is set either way - so the
    refusal has to happen at startup instead.
    """
    from avatar.gateway.sessions import assert_production_ready

    production = cfg.model_copy(
        update={
            # Everything else this check looks at, set to something it accepts,
            # so the failure that surfaces is the one under test.
            "livekit_api_key": "APIrealkey1234",
            "livekit_api_secret": "a" * 40,
            "session_secret": "b" * 40,
            "cookies_secure": True,
            "splat_backend": "fake",
        }
    )
    with pytest.raises(ValueError, match="placeholder"):
        assert_production_ready(production)
