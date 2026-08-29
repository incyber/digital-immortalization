"""A POST that takes no JSON body is a CORS "simple" request: a browser sends
it cross-site with no preflight at all, so the CORS origin allowlist - which
only ever runs before a preflighted request - never gets a say. The defence
here is a required header a cross-site page cannot be made to send; see
gateway/csrf.py for why that is enough."""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from avatar.gateway.app import create_app
from avatar.gateway.csrf import REQUIRED_HEADER, REQUIRED_VALUE
from avatar.gateway.models import Base

AVATAR = {
    "display_name": "Marguerite Chen",
    "locale": "en",
    "country": "US",
    "biography": "A cellist from Vancouver who taught for thirty years.",
    "voice_description": "",
    "boundaries": "",
}

# What the web app's own shared request() helper would need to send on every
# call for these routes to keep working - see the report for the precise
# apps/web change this implies.
HEADERS = {REQUIRED_HEADER: REQUIRED_VALUE}


@pytest_asyncio.fixture
async def client(cfg, tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import avatar.gateway.db as db_module

    cfg.database_url = f"sqlite+aiosqlite:///{tmp_path}/api.db"
    cfg.storage_root = str(tmp_path / "blobs")
    cfg.crisis_lines_verified = "US,ES"

    engine = create_async_engine(cfg.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    db_module._engine = engine
    db_module._factory = async_sessionmaker(engine, expire_on_commit=False)

    async with AsyncClient(
        transport=ASGITransport(app=create_app(cfg)), base_url="http://test"
    ) as c:
        yield c
    await engine.dispose()


async def sign_in(client, email="a@example.com"):
    await client.post(
        "/api/auth/register", json={"email": email, "password": "a-long-enough-password"}
    )


async def test_creating_a_photo_set_without_the_header_is_refused(client):
    await sign_in(client)
    response = await client.post("/api/photo-sets")
    assert response.status_code == 403


async def test_creating_a_photo_set_with_the_header_succeeds(client):
    await sign_in(client)
    response = await client.post("/api/photo-sets", headers=HEADERS)
    assert response.status_code == 201


async def test_evaluate_without_the_header_is_refused(client):
    await sign_in(client)
    photo_set_id = (await client.post("/api/photo-sets", headers=HEADERS)).json()["id"]
    response = await client.post(f"/api/photo-sets/{photo_set_id}/evaluate")
    assert response.status_code == 403


async def test_evaluate_with_the_header_succeeds(client):
    await sign_in(client)
    photo_set_id = (await client.post("/api/photo-sets", headers=HEADERS)).json()["id"]
    response = await client.post(
        f"/api/photo-sets/{photo_set_id}/evaluate", headers=HEADERS
    )
    assert response.status_code == 200


async def test_revalidate_without_the_header_is_refused(client):
    await sign_in(client)
    photo_set_id = (await client.post("/api/photo-sets", headers=HEADERS)).json()["id"]
    response = await client.post(f"/api/photo-sets/{photo_set_id}/revalidate")
    assert response.status_code == 403


async def test_revalidate_with_the_header_succeeds(client):
    await sign_in(client)
    photo_set_id = (await client.post("/api/photo-sets", headers=HEADERS)).json()["id"]
    response = await client.post(
        f"/api/photo-sets/{photo_set_id}/revalidate", headers=HEADERS
    )
    assert response.status_code == 200


async def test_train_without_the_header_is_refused(client):
    await sign_in(client)
    photo_set_id = (await client.post("/api/photo-sets", headers=HEADERS)).json()["id"]
    response = await client.post(f"/api/photo-sets/{photo_set_id}/train")
    assert response.status_code == 403
    # Refused before any training-specific check runs, not a 409 or a 404.
    assert "client header" in response.json()["detail"]


async def test_attaching_a_photo_set_without_the_header_is_refused(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    photo_set_id = (await client.post("/api/photo-sets", headers=HEADERS)).json()["id"]
    response = await client.post(f"/api/avatars/{avatar_id}/photo-set/{photo_set_id}")
    assert response.status_code == 403


async def test_attaching_a_photo_set_with_the_header_succeeds(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    photo_set_id = (await client.post("/api/photo-sets", headers=HEADERS)).json()["id"]
    response = await client.post(
        f"/api/avatars/{avatar_id}/photo-set/{photo_set_id}", headers=HEADERS
    )
    assert response.status_code == 200


async def test_a_wrong_header_value_is_refused_the_same_as_a_missing_one(client):
    # Presence alone must not be enough - a cross-site attacker's fetch can
    # add arbitrary headers too, it just cannot add this specific value
    # without already knowing to.
    await sign_in(client)
    response = await client.post(
        "/api/photo-sets", headers={"x-avatar-client": "something-else"}
    )
    assert response.status_code == 403


async def test_json_bodied_routes_do_not_need_the_header(client):
    # POST /api/avatars already forces a preflight via
    # content-type: application/json; requiring this header there too would
    # be a second, redundant lock rather than closing a real gap.
    await sign_in(client)
    response = await client.post("/api/avatars", json=AVATAR)
    assert response.status_code == 201


async def test_starting_a_likeness_build_without_the_header_is_refused(client):
    """The splat routes have the same shape as the ones the audit named.

    A bodyless POST is a request a browser sends cross-origin without asking
    first, so the session cookie rides along on a form submitted by any other
    site. Starting or cancelling somebody's likeness build is exactly the kind
    of mutation that gap exists to protect. Refused before the set is even
    looked up, which is why a nonexistent id still gives 403 rather than 404.
    """
    await sign_in(client)

    response = await client.post("/api/photo-sets/does-not-exist/splat")

    assert response.status_code == 403


async def test_cancelling_a_likeness_build_without_the_header_is_refused(client):
    await sign_in(client)

    response = await client.post("/api/splat-jobs/does-not-exist/cancel")

    assert response.status_code == 403


async def test_the_splat_routes_are_reachable_with_the_header(client):
    """The guard must refuse the right thing, not everything.

    With the header these get past it and fail on their own merits - an
    unknown photo set is a 404 - which is what shows the check is a gate
    rather than a wall.
    """
    await sign_in(client)

    response = await client.post(
        "/api/photo-sets/does-not-exist/splat", headers=HEADERS
    )

    assert response.status_code != 403
