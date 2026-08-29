"""Building a likeness over HTTP, including the cross-tenant attempts.

The three behaviours that make this endpoint different from every other build
endpoint in the system are each defended here: a refusal comes back as
guidance rather than an error, a finished likeness cannot be read without the
sentence saying how much of it was invented, and a build belongs to exactly
one account.
"""

import asyncio
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from avatar.gateway.app import create_app
from avatar.gateway.models import Base

AVATAR = {
    "display_name": "Aurelio Sandoval",
    "locale": "es",
    "country": "ES",
    "biography": "A carpenter from Cádiz who whistled while he worked.",
    "voice_description": "Warm, slow, a little hoarse.",
    "boundaries": "",
}


@pytest_asyncio.fixture
async def client(cfg, tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import avatar.gateway.db as db_module

    cfg.database_url = f"sqlite+aiosqlite:///{tmp_path}/api.db"
    cfg.storage_root = str(tmp_path / "blobs")
    cfg.crisis_lines_verified = "US,ES"
    # No GPU in the room, and none needed: everything this suite asserts is
    # true of the real backend too, because the report and the route are
    # derived from the job rather than reported by the worker.
    cfg.splat_backend = "fake"

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


def a_clip(seconds: float, fps: int = 6) -> bytes:
    """A real, decodable video: its length is read back with ffprobe."""
    target = Path(tempfile.mkdtemp()) / "clip.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    rng = np.random.default_rng(3)
    for _ in range(int(seconds * fps)):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    return target.read_bytes()


@pytest.fixture
def make_set(client, cfg):
    """A photo set attached to an avatar, holding what a family uploaded.

    The rows are written directly and the clip is put straight into the same
    blob root the gateway reads from. Face detection has its own suite; what
    is under test here is what the counts decide once the checks have run.
    """

    async def build(*, photos: int = 0, frames: int = 0, clip_seconds: float = 0.0):
        from sqlalchemy import select

        import avatar.gateway.db as db_module
        from avatar.gateway.models import Photo, PhotoSet
        from avatar.storage.keys import photo_key, source_clip_key
        from avatar.storage.local import LocalBlobStore

        avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
        set_id = (await client.post("/api/photo-sets")).json()["id"]
        await client.post(f"/api/avatars/{avatar_id}/photo-set/{set_id}")

        async with db_module._factory() as db:
            photo_set = (
                await db.execute(select(PhotoSet).where(PhotoSet.id == set_id))
            ).scalar_one()
            owner_id = photo_set.owner_id
            names = [f"photo-{i:03d}.jpg" for i in range(photos)]
            names += [f"frame-{i:04d}.jpg" for i in range(frames)]
            for name in names:
                db.add(
                    Photo(
                        photo_set_id=set_id,
                        owner_id=owner_id,
                        blob_key=photo_key(owner_id, set_id, name),
                        filename=name,
                        content_type="image/jpeg",
                        size_bytes=1,
                        accepted=True,
                        face_height_fraction=0.4,
                    )
                )
            await db.commit()

        if clip_seconds:
            store = LocalBlobStore(cfg.storage_root)
            await store.put(
                owner_id, source_clip_key(owner_id, set_id), a_clip(clip_seconds), "video/mp4"
            )

        return avatar_id, set_id

    return build


async def until_settled(client, job_id, tries: int = 200) -> dict:
    """Poll the status endpoint the way the page does, until it stops moving."""
    body = {}
    for _ in range(tries):
        body = (await client.get(f"/api/splat-jobs/{job_id}")).json()
        if body["status"] in ("succeeded", "failed", "cancelled"):
            return body
        await asyncio.sleep(0.01)
    return body


# --- building -------------------------------------------------------------


async def test_a_set_with_a_usable_video_builds_by_reconstruction(client, make_set):
    await sign_in(client)
    _, set_id = await make_set(frames=20, clip_seconds=12.0)

    response = await client.post(f"/api/photo-sets/{set_id}/splat")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "building"
    assert body["route"] == "reconstruct"
    assert "your video" in body["reasoning"]


async def test_photographs_alone_build_by_generation(client, make_set):
    await sign_in(client)
    _, set_id = await make_set(photos=20)

    body = (await client.post(f"/api/photo-sets/{set_id}/splat")).json()

    assert body["route"] == "generate"
    assert "photographs" in body["reasoning"]


async def test_a_finished_build_reports_the_disclosure_and_the_measured_fraction(client, make_set):
    await sign_in(client)
    avatar_id, set_id = await make_set(photos=6)

    job_id = (await client.post(f"/api/photo-sets/{set_id}/splat")).json()["job_id"]
    body = await until_settled(client, job_id)

    assert body["status"] == "succeeded"
    # A quarter measured, three quarters invented, and said in a sentence.
    assert body["measured_fraction"] == 0.25
    assert body["generated_fraction"] == 0.75
    assert "generated rather than photographed" in body["disclosure"]
    assert body["avatar_id"] == avatar_id


async def test_the_disclosure_outlives_the_page_that_started_the_build(client, make_set):
    """A family who comes back later still sees how much was invented."""
    await sign_in(client)
    avatar_id, set_id = await make_set(photos=6)
    job_id = (await client.post(f"/api/photo-sets/{set_id}/splat")).json()["job_id"]
    await until_settled(client, job_id)

    body = (await client.get(f"/api/avatars/{avatar_id}/splat")).json()

    assert body["built"] is True
    assert "generated rather than photographed" in body["disclosure"]
    assert body["measured_fraction"] == 0.25
    assert body["gaussians"] > 0


async def test_progress_is_reportable_while_a_build_runs(client, make_set):
    await sign_in(client)
    _, set_id = await make_set(photos=12)
    job_id = (await client.post(f"/api/photo-sets/{set_id}/splat")).json()["job_id"]

    body = (await client.get(f"/api/splat-jobs/{job_id}")).json()

    assert 0.0 < body["progress"] <= 1.0
    assert body["status"] in ("running", "succeeded")


async def test_an_avatar_with_no_likeness_yet_claims_nothing_about_one(client, make_set):
    await sign_in(client)
    avatar_id, _ = await make_set(photos=2)

    body = (await client.get(f"/api/avatars/{avatar_id}/splat")).json()

    assert body["built"] is False
    assert body["disclosure"] is None
    # Not zero: NULL is "no likeness", and 0.0 would be a claim that none of
    # one was measured.
    assert body["measured_fraction"] is None


# --- refusal is guidance, not an error ------------------------------------


async def test_too_little_material_is_answered_with_guidance_not_an_error(client, make_set):
    await sign_in(client)
    _, set_id = await make_set(photos=2)

    response = await client.post(f"/api/photo-sets/{set_id}/splat")

    # Not 400. A family who uploaded two photographs has not made a bad
    # request; they have been told what else we need.
    assert response.status_code == 200
    body = response.json()
    assert body["buildable"] is False
    assert "at least 8 seconds" in body["guidance"]
    assert "at least 3 photographs" in body["guidance"]
    assert "we have 2" in body["guidance"]


async def test_a_refusal_names_what_was_weighed_for_support_to_read(client, make_set):
    await sign_in(client)
    _, set_id = await make_set(photos=1)

    body = (await client.post(f"/api/photo-sets/{set_id}/splat")).json()

    assert any("photographs accepted: 1" in line for line in body["considered"])


async def test_a_photo_set_not_attached_to_an_avatar_says_so(client):
    await sign_in(client)
    set_id = (await client.post("/api/photo-sets")).json()["id"]

    response = await client.post(f"/api/photo-sets/{set_id}/splat")

    assert response.status_code == 409
    assert "attach this photo set" in response.json()["detail"]


# --- tenant isolation -----------------------------------------------------


async def test_starting_a_build_requires_signing_in(client):
    assert (await client.post("/api/photo-sets/anything/splat")).status_code == 401


async def test_one_tenant_cannot_start_a_build_on_anothers_photo_set(client, make_set):
    await sign_in(client, "owner@example.com")
    _, set_id = await make_set(photos=20)
    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    response = await client.post(f"/api/photo-sets/{set_id}/splat")

    # 404, not 403: telling them apart would confirm the set exists.
    assert response.status_code == 404


async def test_one_tenant_cannot_read_or_cancel_anothers_build(client, make_set):
    await sign_in(client, "owner@example.com")
    avatar_id, set_id = await make_set(photos=20)
    job_id = (await client.post(f"/api/photo-sets/{set_id}/splat")).json()["job_id"]
    await until_settled(client, job_id)
    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    assert (await client.get(f"/api/splat-jobs/{job_id}")).status_code == 404
    assert (await client.post(f"/api/splat-jobs/{job_id}/cancel")).status_code == 404
    assert (await client.get(f"/api/avatars/{avatar_id}/splat")).status_code == 404


async def test_a_tenant_can_cancel_their_own_build(client, make_set):
    await sign_in(client)
    _, set_id = await make_set(photos=20)
    job_id = (await client.post(f"/api/photo-sets/{set_id}/splat")).json()["job_id"]

    response = await client.post(f"/api/splat-jobs/{job_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] in ("cancelled", "succeeded")
