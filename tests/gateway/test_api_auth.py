"""The HTTP surface, exercised the way an attacker would reach it."""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from avatar.gateway.app import create_app
from avatar.gateway.models import Avatar, Base, ConsentRecord, ConsentStatus


@pytest_asyncio.fixture
async def client(cfg, tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import avatar.gateway.db as db_module

    cfg.database_url = f"sqlite+aiosqlite:///{tmp_path}/api.db"
    engine = create_async_engine(cfg.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    db_module._engine = engine
    db_module._factory = async_sessionmaker(engine, expire_on_commit=False)

    app = create_app(cfg)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await engine.dispose()


CREDS = {"email": "family@example.com", "password": "a-long-enough-password"}


async def test_registration_signs_you_in(client):
    response = await client.post("/api/auth/register", json=CREDS)
    assert response.status_code == 201
    assert (await client.get("/api/me")).status_code == 200


async def test_sessions_require_sign_in(client):
    # The hole this closed: before authentication existed, this returned a
    # joinable room token to anybody who asked.
    response = await client.post("/api/sessions", json={"avatar_id": "colon"})
    assert response.status_code == 401


async def test_me_requires_sign_in(client):
    assert (await client.get("/api/me")).status_code == 401


async def test_login_with_a_wrong_password_is_401(client):
    await client.post("/api/auth/register", json=CREDS)
    await client.post("/api/auth/logout")
    response = await client.post(
        "/api/auth/login", json={**CREDS, "password": "not-the-password"}
    )
    assert response.status_code == 401


async def test_login_does_not_reveal_whether_the_account_exists(client):
    await client.post("/api/auth/register", json=CREDS)
    await client.post("/api/auth/logout")
    known = await client.post("/api/auth/login", json={**CREDS, "password": "wrong-one-here"})
    unknown = await client.post(
        "/api/auth/login",
        json={"email": "nobody@example.com", "password": "wrong-one-here"},
    )
    assert known.status_code == unknown.status_code
    assert known.json()["detail"] == unknown.json()["detail"]


async def test_session_cookie_is_not_readable_by_scripts(client):
    response = await client.post("/api/auth/register", json=CREDS)
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie


async def test_logout_clears_the_session(client):
    await client.post("/api/auth/register", json=CREDS)
    await client.post("/api/auth/logout")
    assert (await client.get("/api/me")).status_code == 401


async def test_another_tenants_avatar_is_a_404_not_a_403(client):
    """A 403 would confirm the avatar exists. Signed in, but not yours."""
    import avatar.gateway.db as db_module

    # Tenant A owns a fully consented avatar.
    await client.post("/api/auth/register", json=CREDS)
    me = (await client.get("/api/me")).json()
    async with db_module._factory() as db:
        a = Avatar(owner_id=me["id"], display_name="Theirs", profile_path="p.json")
        db.add(a)
        await db.flush()
        db.add(
            ConsentRecord(
                avatar_id=a.id,
                rights_holder_name="Executor",
                relationship_to_subject="son",
                jurisdiction="US-CA",
                status=ConsentStatus.VERIFIED,
            )
        )
        await db.commit()
        avatar_id = a.id

    # Tenant B signs in and asks for it.
    await client.post("/api/auth/logout")
    await client.post(
        "/api/auth/register",
        json={"email": "stranger@example.com", "password": "a-long-enough-password"},
    )
    response = await client.post("/api/sessions", json={"avatar_id": avatar_id})
    assert response.status_code == 404, "403 would confirm the avatar exists"


async def test_duplicate_registration_is_400(client):
    await client.post("/api/auth/register", json=CREDS)
    response = await client.post("/api/auth/register", json=CREDS)
    assert response.status_code == 400
