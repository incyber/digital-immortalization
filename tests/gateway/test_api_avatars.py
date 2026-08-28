"""Avatar creation. Every character is the customer's; none ship with the app."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from avatar.gateway.app import create_app
from avatar.gateway.models import Base

AVATAR = {
    "display_name": "Marguerite Chen",
    "locale": "en",
    "country": "US",
    "biography": "A cellist from Vancouver who taught for thirty years.",
    "voice_description": "Dry, unhurried, fond of understatement.",
    "boundaries": "",
}


@pytest_asyncio.fixture
async def client(cfg, tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import avatar.gateway.db as db_module

    cfg.database_url = f"sqlite+aiosqlite:///{tmp_path}/api.db"
    cfg.storage_root = str(tmp_path / "blobs")
    # The operator has attested these two countries.
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


async def test_a_new_account_has_no_avatars(client):
    # There is no built-in character to fall back on.
    await sign_in(client)
    assert (await client.get("/api/avatars")).json()["avatars"] == []


async def test_creating_an_avatar(client):
    await sign_in(client)
    response = await client.post("/api/avatars", json=AVATAR)
    assert response.status_code == 201
    body = response.json()
    assert body["display_name"] == "Marguerite Chen"
    assert body["id"]


async def test_the_disclosure_names_the_customers_person(client):
    await sign_in(client)
    body = (await client.post("/api/avatars", json=AVATAR)).json()
    assert "Marguerite Chen" in body["disclosure"]
    assert "synthetic recreation" in body["disclosure"]


async def test_the_customer_cannot_supply_a_disclosure(client):
    # Extra keys are ignored by the model, so an attempt to override the
    # disclosure silently does nothing rather than taking effect.
    await sign_in(client)
    body = (
        await client.post(
            "/api/avatars", json={**AVATAR, "disclosure": "This is definitely a real person"}
        )
    ).json()
    assert "definitely a real person" not in body["disclosure"]


async def test_the_customer_cannot_supply_a_crisis_number(client):
    await sign_in(client)
    body = (
        await client.post(
            "/api/avatars", json={**AVATAR, "crisis_line_number": "555-0100"}
        )
    ).json()
    assert body["crisis_line"]["number"] == "988"


async def test_an_avatar_without_a_biography_is_refused(client):
    await sign_in(client)
    response = await client.post("/api/avatars", json={**AVATAR, "biography": ""})
    assert response.status_code == 422


async def test_an_unattested_country_is_refused(client):
    await sign_in(client)
    response = await client.post("/api/avatars", json={**AVATAR, "country": "MX"})
    assert response.status_code == 400
    assert "MX" in response.json()["detail"] or "Mexico" in response.json()["detail"]


async def test_countries_lists_only_what_the_operator_attested(client):
    await sign_in(client)
    codes = {c["code"] for c in (await client.get("/api/countries")).json()["countries"]}
    assert codes == {"US", "ES"}


async def test_avatars_are_private_to_their_owner(client):
    await sign_in(client, "owner@example.com")
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]

    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    assert (await client.get(f"/api/avatars/{avatar_id}")).status_code == 404
    assert (await client.get("/api/avatars")).json()["avatars"] == []


async def test_another_tenant_cannot_edit_an_avatar(client):
    await sign_in(client, "owner@example.com")
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]

    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    response = await client.patch(
        f"/api/avatars/{avatar_id}", json={**AVATAR, "display_name": "Hijacked"}
    )
    assert response.status_code == 404


async def test_an_avatar_without_assets_is_not_callable(client):
    await sign_in(client)
    body = (await client.post("/api/avatars", json=AVATAR)).json()
    assert body["callable"] is False


async def test_editing_an_avatar_updates_the_disclosure(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    body = (
        await client.patch(
            f"/api/avatars/{avatar_id}", json={**AVATAR, "display_name": "Tomás Duarte"}
        )
    ).json()
    assert "Tomás Duarte" in body["disclosure"]


async def test_consent_is_recorded_as_pending_not_verified(client):
    # A self-service route to "verified" would make the gate decorative.
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    response = await client.post(
        f"/api/avatars/{avatar_id}/consent",
        json={
            "rights_holder_name": "Ana Chen",
            "relationship_to_subject": "daughter",
            "jurisdiction": "US-WA",
        },
    )
    assert response.status_code == 201
    assert response.json()["status"] == "pending"


async def test_a_pending_consent_still_blocks_a_call(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    await client.post(
        f"/api/avatars/{avatar_id}/consent",
        json={
            "rights_holder_name": "Ana Chen",
            "relationship_to_subject": "daughter",
            "jurisdiction": "US-WA",
        },
    )
    response = await client.post("/api/sessions", json={"avatar_id": avatar_id})
    assert response.status_code == 403


@pytest.mark.parametrize("country", ["us", "Us", "US"])
async def test_country_is_case_insensitive(client, country):
    await sign_in(client)
    response = await client.post("/api/avatars", json={**AVATAR, "country": country})
    assert response.status_code == 201


async def test_recreating_yourself_is_verified_immediately(client):
    """Nobody needs permission to be themselves.

    Requiring a reviewer here would be ceremony, and it is the case that
    otherwise dead-ends: build an avatar of yourself and never be able to call
    it.
    """
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    response = await client.post(
        f"/api/avatars/{avatar_id}/consent",
        json={
            "rights_holder_name": "Myself",
            "relationship_to_subject": "self",
            "jurisdiction": "US-WA",
        },
    )
    assert response.status_code == 201
    assert response.json()["status"] == "verified"
    assert response.json()["needs_review"] is False


async def test_self_attestation_is_case_insensitive(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    response = await client.post(
        f"/api/avatars/{avatar_id}/consent",
        json={
            "rights_holder_name": "Myself",
            "relationship_to_subject": "Self",
            "jurisdiction": "US-WA",
        },
    )
    assert response.json()["status"] == "verified"


async def test_anybody_elses_likeness_still_needs_review(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    response = await client.post(
        f"/api/avatars/{avatar_id}/consent",
        json={
            "rights_holder_name": "Ana Chen",
            "relationship_to_subject": "daughter",
            "jurisdiction": "US-WA",
        },
    )
    assert response.json()["status"] == "pending"
    assert response.json()["needs_review"] is True


async def test_a_stranger_cannot_self_attest_on_your_avatar(client):
    await sign_in(client, "owner@example.com")
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    response = await client.post(
        f"/api/avatars/{avatar_id}/consent",
        json={
            "rights_holder_name": "Me",
            "relationship_to_subject": "self",
            "jurisdiction": "US-WA",
        },
    )
    assert response.status_code == 404


def a_recording(seconds=15.0, rate=24000, volume="0.5"):
    import subprocess

    return subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi",
            "-i", f"sine=frequency=180:duration={seconds}:sample_rate={rate}",
            "-af", f"volume={volume}", "-f", "wav", "-",
        ],
        capture_output=True, check=False,
    ).stdout


async def test_a_usable_recording_is_accepted(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]

    response = await client.post(
        f"/api/avatars/{avatar_id}/voice",
        files={"file": ("voice.wav", a_recording(), "audio/wav")},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["seconds"] > 10
    assert body["quality"] in {"usable", "good"}


async def test_the_avatar_reports_having_a_voice(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    assert (await client.get(f"/api/avatars/{avatar_id}")).json()["has_voice"] is False

    await client.post(
        f"/api/avatars/{avatar_id}/voice",
        files={"file": ("voice.wav", a_recording(), "audio/wav")},
    )
    assert (await client.get(f"/api/avatars/{avatar_id}")).json()["has_voice"] is True


async def test_a_too_short_recording_is_refused_with_the_reason(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]

    response = await client.post(
        f"/api/avatars/{avatar_id}/voice",
        files={"file": ("voice.wav", a_recording(seconds=2), "audio/wav")},
    )
    assert response.status_code == 400
    assert "four seconds" in response.json()["detail"]


async def test_an_unsupported_type_is_refused(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    response = await client.post(
        f"/api/avatars/{avatar_id}/voice",
        files={"file": ("voice.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


async def test_another_tenant_cannot_attach_a_voice(client):
    await sign_in(client, "owner@example.com")
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    response = await client.post(
        f"/api/avatars/{avatar_id}/voice",
        files={"file": ("voice.wav", a_recording(), "audio/wav")},
    )
    assert response.status_code == 404


async def test_the_stored_voice_lands_in_the_owners_prefix(client):
    from sqlalchemy import select

    import avatar.gateway.db as db_module
    from avatar.gateway.models import Avatar

    await sign_in(client)
    me = (await client.get("/api/me")).json()
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]
    await client.post(
        f"/api/avatars/{avatar_id}/voice",
        files={"file": ("voice.wav", a_recording(), "audio/wav")},
    )

    async with db_module._factory() as db:
        avatar = (
            await db.execute(select(Avatar).where(Avatar.id == avatar_id))
        ).scalar_one()
    assert avatar.voice_key.startswith(f"tenants/{me['id']}/")
