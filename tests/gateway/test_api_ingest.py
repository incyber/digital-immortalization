"""Upload and training over HTTP, including the cross-tenant attempts."""

import cv2
import numpy as np
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from avatar.gateway.app import create_app
from avatar.gateway.models import Base


def an_image(w=1024, h=1024) -> bytes:
    rng = np.random.default_rng(2)
    frame = (np.full((h, w, 3), 140, np.uint8) + rng.integers(-30, 30, (h, w, 3))).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


@pytest_asyncio.fixture
async def client(cfg, tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import avatar.gateway.db as db_module

    cfg.database_url = f"sqlite+aiosqlite:///{tmp_path}/api.db"
    cfg.storage_root = str(tmp_path / "blobs")
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


def a_file(name="p.jpg"):
    return {"file": (name, an_image(), "image/jpeg")}


async def test_requirements_are_served_not_hardcoded_in_the_frontend(client):
    await sign_in(client)
    body = (await client.get("/api/photo-sets/requirements")).json()
    assert body["recommended_min"] == 20
    assert body["recommended_max"] == 30
    assert body["minimum_half_body"] >= 5
    assert any("Half body" in s["label"] for s in body["shots"])


async def test_creating_a_set_requires_sign_in(client):
    assert (await client.post("/api/photo-sets")).status_code == 401


async def test_upload_returns_a_per_image_verdict(client):
    await sign_in(client)
    set_id = (await client.post("/api/photo-sets")).json()["id"]
    response = await client.post(f"/api/photo-sets/{set_id}/photos", files=a_file())
    assert response.status_code == 201
    assert "accepted" in response.json()
    assert "reasons" in response.json()


async def test_another_tenant_cannot_read_a_photo_set(client):
    await sign_in(client, "owner@example.com")
    set_id = (await client.post("/api/photo-sets")).json()["id"]
    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    assert (await client.get(f"/api/photo-sets/{set_id}")).status_code == 404


async def test_another_tenant_cannot_upload_into_a_photo_set(client):
    await sign_in(client, "owner@example.com")
    set_id = (await client.post("/api/photo-sets")).json()["id"]
    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    response = await client.post(f"/api/photo-sets/{set_id}/photos", files=a_file())
    assert response.status_code == 404


async def test_another_tenant_cannot_delete_a_photo_set(client):
    await sign_in(client, "owner@example.com")
    set_id = (await client.post("/api/photo-sets")).json()["id"]
    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    assert (await client.delete(f"/api/photo-sets/{set_id}")).status_code == 404


async def test_an_unvalidated_set_cannot_be_trained(client):
    # Training is the expensive step; it must be gated on validation.
    await sign_in(client)
    set_id = (await client.post("/api/photo-sets")).json()["id"]
    await client.post(f"/api/photo-sets/{set_id}/photos", files=a_file())

    response = await client.post(f"/api/photo-sets/{set_id}/train")
    assert response.status_code == 409


async def test_a_failing_set_is_rejected_with_reasons(client):
    await sign_in(client)
    set_id = (await client.post("/api/photo-sets")).json()["id"]
    for i in range(3):
        await client.post(f"/api/photo-sets/{set_id}/photos", files=a_file(f"p{i}.jpg"))

    body = (await client.post(f"/api/photo-sets/{set_id}/evaluate")).json()
    assert body["status"] == "rejected"
    assert body["problems"]


async def test_unsupported_file_types_are_refused(client):
    await sign_in(client)
    set_id = (await client.post("/api/photo-sets")).json()["id"]
    response = await client.post(
        f"/api/photo-sets/{set_id}/photos", files={"file": ("x.gif", b"GIF89a", "image/gif")}
    )
    assert response.status_code == 400


async def test_another_tenants_training_job_is_a_404(client):
    await sign_in(client, "owner@example.com")
    set_id = (await client.post("/api/photo-sets")).json()["id"]
    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    assert (await client.get(f"/api/training-jobs/{set_id}")).status_code == 404
