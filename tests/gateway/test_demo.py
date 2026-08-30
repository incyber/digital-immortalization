"""Demo mode: what it changes, and everything it must not change.

The flag skips authentication so somebody can try the product without
inventing a password. Four properties are load-bearing and each has a test
here, because every one of them, broken, is a way for a stranger to end up
holding a family's photographs:

  off is unchanged - the default must be the current product exactly
  on signs in without credentials
  on is ONE shared tenant, not one per visitor
  on does not touch the consent gate
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from avatar.config import Settings
from avatar.gateway.app import create_app
from avatar.gateway.demo import DEMO_EMAIL
from avatar.gateway.models import Avatar, Base, ConsentRecord, ConsentStatus, User


async def _client(cfg, tmp_path, name):
    """An app on its own database file, reachable over ASGI."""
    import avatar.gateway.db as db_module

    cfg.database_url = f"sqlite+aiosqlite:///{tmp_path}/{name}.db"
    engine = create_async_engine(cfg.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    db_module._engine = engine
    db_module._factory = async_sessionmaker(engine, expire_on_commit=False)

    app = create_app(cfg)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), engine


# An avatar cannot be created until an operator has attested to the country's
# crisis line (safety/crisis_lines.py). Attested here so these tests are about
# demo mode rather than about that gate, which has its own tests.
ATTESTED = "US"


@pytest_asyncio.fixture
async def demo_client(tmp_path):
    cfg = Settings(_env_file=None, demo_mode=True, crisis_lines_verified=ATTESTED)
    client, engine = await _client(cfg, tmp_path, "demo")
    async with client as c:
        yield c
    await engine.dispose()


@pytest_asyncio.fixture
async def normal_client(tmp_path):
    cfg = Settings(_env_file=None, demo_mode=False, crisis_lines_verified=ATTESTED)
    client, engine = await _client(cfg, tmp_path, "normal")
    async with client as c:
        yield c
    await engine.dispose()


# --- off is the current product, unchanged --------------------------------


async def test_off_still_requires_sign_in(normal_client):
    assert (await normal_client.get("/api/me")).status_code == 401
    assert (await normal_client.get("/api/avatars")).status_code == 401
    body = await normal_client.post("/api/sessions", json={"avatar_id": "anything"})
    assert body.status_code == 401


async def test_off_is_the_default():
    """Nobody has to remember to turn it off, which is the only safe default."""
    assert Settings(_env_file=None).demo_mode is False


async def test_off_reports_itself_as_off(normal_client):
    assert (await normal_client.get("/api/config")).json() == {"demo_mode": False}


async def test_off_still_takes_registrations(normal_client):
    response = await normal_client.post(
        "/api/auth/register",
        json={"email": "family@example.com", "password": "a-long-enough-password"},
    )
    assert response.status_code == 201


# --- on signs you in without credentials ----------------------------------


async def test_on_signs_you_in_with_no_credentials(demo_client):
    response = await demo_client.get("/api/me")
    assert response.status_code == 200
    assert response.json()["id"]


async def test_on_issues_a_first_party_session_cookie(demo_client):
    response = await demo_client.get("/api/me")
    header = response.headers["set-cookie"].lower()
    assert "avatar_session=" in header
    assert "samesite=lax" in header
    assert "httponly" in header


async def test_on_lets_an_anonymous_visitor_create_an_avatar(demo_client):
    response = await demo_client.post(
        "/api/avatars",
        json={"display_name": "Marguerite Chen", "country": "US", "biography": "A cellist."},
    )
    assert response.status_code == 201, response.text
    assert (await demo_client.get("/api/avatars")).json()["avatars"]


async def test_on_says_so_in_the_config_the_site_reads(demo_client):
    """The interface cannot warn anybody that this is shared unless it is told."""
    assert (await demo_client.get("/api/config")).json() == {"demo_mode": True}


# --- one shared tenant, not one per visitor -------------------------------


async def test_two_visitors_share_one_account(demo_client, tmp_path):
    """The property that makes it a demo rather than an open sign-up.

    Two clients with no cookies between them are two different browsers. If
    they came back with different ids, every visitor would be minting a tenant
    and the shared-demo warning on the screen would be a lie in the other
    direction.
    """
    first = (await demo_client.get("/api/me")).json()["id"]

    # A second browser: same app, no cookie jar in common.
    transport = demo_client._transport
    async with AsyncClient(transport=transport, base_url="http://test") as other:
        second = (await other.get("/api/me")).json()["id"]

    assert first == second


async def test_an_avatar_created_by_one_visitor_is_visible_to_the_next(demo_client):
    """Stated as a test because it is what the banner on the screen promises.

    It is not a bug being documented - it is the consequence of one shared
    tenant, and if it ever stopped being true the warning would be wrong.
    """
    await demo_client.post(
        "/api/avatars",
        json={"display_name": "Tomás Duarte", "country": "US", "biography": "A bookbinder."},
    )

    async with AsyncClient(
        transport=demo_client._transport, base_url="http://test"
    ) as other:
        avatars = (await other.get("/api/avatars")).json()["avatars"]

    assert [a["display_name"] for a in avatars] == ["Tomás Duarte"]


async def test_only_one_account_row_is_ever_created(demo_client, tmp_path):
    import avatar.gateway.db as db_module

    for _ in range(5):
        async with AsyncClient(
            transport=demo_client._transport, base_url="http://test"
        ) as visitor:
            await visitor.get("/api/me")

    async with db_module._factory() as db:
        from sqlalchemy import select

        emails = (await db.execute(select(User.email))).scalars().all()

    assert emails == [DEMO_EMAIL]


async def test_on_refuses_to_take_new_accounts(demo_client):
    """Registration would create the second tenant the startup check forbids."""
    response = await demo_client.post(
        "/api/auth/register",
        json={"email": "family@example.com", "password": "a-long-enough-password"},
    )
    assert response.status_code == 409

    login = await demo_client.post(
        "/api/auth/login",
        json={"email": "family@example.com", "password": "a-long-enough-password"},
    )
    assert login.status_code == 409


# --- consent is untouched -------------------------------------------------


async def test_the_consent_gate_still_blocks_a_call(demo_client):
    """Demo mode skips authentication. It does not skip the legal gate.

    A demo visitor reaches an avatar with no consent record on file and is
    refused for the same reason a customer would be, with the same message.
    """
    created = await demo_client.post(
        "/api/avatars",
        json={"display_name": "Marguerite Chen", "country": "US", "biography": "A cellist."},
    )
    avatar_id = created.json()["id"]

    response = await demo_client.post("/api/sessions", json={"avatar_id": avatar_id})
    assert response.status_code == 403
    assert "consent" in response.json()["detail"]


async def test_the_consent_gate_still_opens_for_a_verified_record(demo_client):
    """The complement, so the test above is proof of a gate and not of a wall."""
    import avatar.gateway.db as db_module

    created = await demo_client.post(
        "/api/avatars",
        json={"display_name": "Marguerite Chen", "country": "US", "biography": "A cellist."},
    )
    avatar_id = created.json()["id"]

    async with db_module._factory() as db:
        db.add(
            ConsentRecord(
                avatar_id=avatar_id,
                rights_holder_name="Executor",
                relationship_to_subject="daughter",
                jurisdiction="US-CA",
                status=ConsentStatus.VERIFIED,
            )
        )
        # And a likeness, because consent alone no longer opens a call - there
        # has to be something real to show.
        avatar = await db.get(Avatar, avatar_id)
        avatar.splat_key = f"tenants/demo/avatars/{avatar_id}/avatar.splat"
        await db.commit()

    response = await demo_client.post("/api/sessions", json={"avatar_id": avatar_id})
    assert response.status_code == 200, response.text
    assert response.json()["token"]


# --- the startup refusal --------------------------------------------------


def test_demo_mode_refuses_to_run_over_real_accounts():
    from avatar.gateway.sessions import assert_demo_mode_safe

    cfg = Settings(_env_file=None, demo_mode=True)

    with pytest.raises(ValueError, match="not the demo account"):
        assert_demo_mode_safe(cfg, [DEMO_EMAIL, "family@example.com"])


def test_demo_mode_is_content_with_only_its_own_account():
    from avatar.gateway.sessions import assert_demo_mode_safe

    cfg = Settings(_env_file=None, demo_mode=True)
    assert_demo_mode_safe(cfg, [DEMO_EMAIL])
    assert_demo_mode_safe(cfg, [])


def test_the_check_does_nothing_at_all_when_the_flag_is_off():
    """Off must be off. A guard that fires anyway is a guard that gets deleted."""
    from avatar.gateway.sessions import assert_demo_mode_safe

    cfg = Settings(_env_file=None, demo_mode=False)
    assert_demo_mode_safe(cfg, ["family@example.com", "another@example.com"])


def test_assert_production_ready_carries_the_demo_refusal():
    """Stated against the function the deployment actually calls.

    The last two safety controls in this project were correct and dead. This
    asserts the wiring: the production check is what a deployment runs, so the
    demo refusal has to be reachable through it.
    """
    from avatar.gateway.sessions import assert_production_ready

    cfg = Settings(
        _env_file=None,
        demo_mode=True,
        production=True,
        cookies_secure=True,
        livekit_api_key="a-real-key",
        livekit_api_secret="x" * 40,
        session_secret="a-real-and-sufficiently-long-session-secret",
        # Production also refuses the placeholder splat backend, and that
        # refusal would fire first and hide the one under test.
        splat_backend="runpod",
    )

    # Clean: no real accounts.
    assert_production_ready(cfg, [DEMO_EMAIL])

    with pytest.raises(ValueError, match="not the demo account"):
        assert_production_ready(cfg, ["family@example.com"])


async def test_the_gateway_refuses_to_start_over_real_accounts(tmp_path):
    """The whole boot, not the helper. Nothing is served after this raises."""
    import avatar.gateway.db as db_module
    from avatar.gateway.auth import hash_password

    database = f"sqlite+aiosqlite:///{tmp_path}/occupied.db"
    engine = create_async_engine(database)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as db:
        db.add(User(email="family@example.com", password_hash=hash_password("x" * 14)))
        await db.commit()
    await engine.dispose()

    cfg = Settings(_env_file=None, demo_mode=True, database_url=database)
    app = create_app(cfg)

    with pytest.raises(ValueError, match="not the demo account"):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover - the context manager raises on entry

    if db_module._engine is not None:
        await db_module._engine.dispose()


async def test_an_avatar_in_the_demo_account_still_gets_its_declaration(demo_client):
    """The synthetic-media disclosure is generated server-side, from the name.

    Demo mode must not be a route to a recreation that does not declare itself.
    """
    created = await demo_client.post(
        "/api/avatars",
        json={"display_name": "Marguerite Chen", "country": "US", "biography": "A cellist."},
    )
    assert created.json()["disclosure"]
    assert "Marguerite Chen" in created.json()["disclosure"]


async def test_demo_avatars_are_still_owned_by_a_real_tenant_row(demo_client):
    """Ownership is not bypassed, it is satisfied - by the demo account.

    An avatar with no owner would slip past every query in tenancy.py.
    """
    from sqlalchemy import select

    import avatar.gateway.db as db_module

    await demo_client.post(
        "/api/avatars",
        json={"display_name": "Marguerite Chen", "country": "US", "biography": "A cellist."},
    )

    async with db_module._factory() as db:
        avatar = (await db.execute(select(Avatar))).scalars().one()
        owner = await db.get(User, avatar.owner_id)

    assert owner is not None
    assert owner.email == DEMO_EMAIL
