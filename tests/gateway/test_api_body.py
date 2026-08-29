"""What the person's body was like, in the family's own words.

Nothing here is estimated from the photographs. A head-and-shoulders picture
does not contain a body, so the family is asked instead, and these tests exist
to hold two lines: an answer is stored exactly as given or refused outright,
and no answer at all is a perfectly ordinary way to use the product.
"""

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

BODY = {
    "height_cm": 163,
    "build": "slight",
    "shoulders": "narrow",
    "posture": "stooped",
}


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


async def sign_back_in(client, email):
    await client.post(
        "/api/auth/login", json={"email": email, "password": "a-long-enough-password"}
    )


async def test_what_the_family_stated_is_stored_and_read_back_unchanged(client):
    await sign_in(client)
    created = (await client.post("/api/avatars", json={**AVATAR, **BODY})).json()

    fetched = (await client.get(f"/api/avatars/{created['id']}")).json()
    assert fetched["body"]["stated"] == BODY


async def test_an_avatar_can_be_created_without_answering_any_of_it(client):
    # Somebody who does not know, or cannot face the questions today, still
    # gets their avatar.
    await sign_in(client)
    response = await client.post("/api/avatars", json=AVATAR)
    assert response.status_code == 201
    assert response.json()["body"]["stated"] == {
        "height_cm": None,
        "build": None,
        "shoulders": None,
        "posture": None,
    }


async def test_an_avatar_with_nothing_stated_is_still_a_working_avatar(client):
    # Leaving the questions blank must not quietly cost a family anything
    # else: the disclosure, the crisis line and the rest of the avatar are
    # exactly as they would have been.
    await sign_in(client)
    body = (await client.post("/api/avatars", json=AVATAR)).json()
    assert "Marguerite Chen" in body["disclosure"]
    assert body["crisis_line"]["number"] == "988"


async def test_an_unanswered_question_is_built_from_a_neutral_default(client):
    # "We were not told" and "they were average" stay visibly different: one
    # is reported as stated, the other only as what the build will use.
    await sign_in(client)
    body = (await client.post("/api/avatars", json=AVATAR)).json()["body"]
    assert body["stated"]["build"] is None
    assert body["in_use"] == {
        "height_cm": 170,
        "build": "average",
        "shoulders": "average",
        "posture": "relaxed",
    }


async def test_a_stated_answer_is_the_one_the_build_uses(client):
    await sign_in(client)
    body = (await client.post("/api/avatars", json={**AVATAR, **BODY})).json()["body"]
    assert body["in_use"] == BODY


async def test_each_question_can_be_answered_on_its_own(client):
    # The questions are independent; answering one does not oblige a family to
    # invent the other three.
    await sign_in(client)
    for field, value in BODY.items():
        created = (await client.post("/api/avatars", json={**AVATAR, field: value})).json()
        stated = created["body"]["stated"]
        assert stated[field] == value
        assert [v for k, v in stated.items() if k != field] == [None, None, None]


async def test_a_build_that_is_not_one_of_the_words_offered_is_refused(client):
    await sign_in(client)
    response = await client.post("/api/avatars", json={**AVATAR, "build": "muscular"})
    assert response.status_code == 422


async def test_shoulders_and_posture_are_refused_the_same_way(client):
    await sign_in(client)
    assert (
        await client.post("/api/avatars", json={**AVATAR, "shoulders": "wide"})
    ).status_code == 422
    assert (
        await client.post("/api/avatars", json={**AVATAR, "posture": "slouching"})
    ).status_code == 422


async def test_a_height_no_person_has_had_is_refused(client):
    await sign_in(client)
    for height in (0, -5, 12, 400):
        response = await client.post("/api/avatars", json={**AVATAR, "height_cm": height})
        assert response.status_code == 422, height


async def test_a_height_that_is_not_a_number_is_refused_rather_than_guessed_at(client):
    await sign_in(client)
    response = await client.post("/api/avatars", json={**AVATAR, "height_cm": "tall"})
    assert response.status_code == 422


async def test_a_refused_answer_is_never_quietly_replaced_by_a_usable_one(client):
    # The failure this guards against is a bad value being clamped into range
    # and then presented back to the family as though they had said it.
    await sign_in(client)
    await client.post("/api/avatars", json={**AVATAR, "height_cm": 400})
    assert (await client.get("/api/avatars")).json()["avatars"] == []


async def test_the_family_can_add_the_answers_later(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]

    updated = await client.patch(f"/api/avatars/{avatar_id}", json={**AVATAR, **BODY})
    assert updated.status_code == 200
    assert updated.json()["body"]["stated"] == BODY


async def test_the_family_can_change_their_mind(client):
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json={**AVATAR, **BODY})).json()["id"]

    body = (
        await client.patch(
            f"/api/avatars/{avatar_id}", json={**AVATAR, **BODY, "build": "solid"}
        )
    ).json()["body"]
    assert body["stated"]["build"] == "solid"


async def test_the_family_can_take_an_answer_back(client):
    # Sent as null, which is a decision, and it returns to unanswered rather
    # than to a neutral value pretending to be one.
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json={**AVATAR, **BODY})).json()["id"]

    body = (
        await client.patch(
            f"/api/avatars/{avatar_id}", json={**AVATAR, **BODY, "posture": None}
        )
    ).json()["body"]
    assert body["stated"]["posture"] is None
    assert body["in_use"]["posture"] == "relaxed"


async def test_an_edit_that_does_not_mention_the_body_leaves_it_alone(client):
    # An older page saving a name change must not silently erase what somebody
    # who knew the person told us.
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json={**AVATAR, **BODY})).json()["id"]

    body = (
        await client.patch(
            f"/api/avatars/{avatar_id}", json={**AVATAR, "display_name": "Marguerite"}
        )
    ).json()["body"]
    assert body["stated"] == BODY


async def test_another_tenant_cannot_read_the_body_attributes(client):
    await sign_in(client, "owner@example.com")
    avatar_id = (await client.post("/api/avatars", json={**AVATAR, **BODY})).json()["id"]

    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    assert (await client.get(f"/api/avatars/{avatar_id}")).status_code == 404
    assert (await client.get("/api/avatars")).json()["avatars"] == []


async def test_another_tenant_cannot_change_the_body_attributes(client):
    await sign_in(client, "owner@example.com")
    avatar_id = (await client.post("/api/avatars", json={**AVATAR, **BODY})).json()["id"]

    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")
    hijack = await client.patch(
        f"/api/avatars/{avatar_id}", json={**AVATAR, "height_cm": 200, "build": "heavy"}
    )
    assert hijack.status_code == 404

    # The refusal has to be a refusal, not a 404 rendered after the write.
    await client.post("/api/auth/logout")
    await sign_back_in(client, "owner@example.com")
    assert (await client.get(f"/api/avatars/{avatar_id}")).json()["body"]["stated"] == BODY


async def test_the_refusal_does_not_reveal_that_the_avatar_exists(client):
    # Same answer for somebody else's avatar as for one that was never
    # created, so an outsider cannot use these questions to enumerate people.
    await sign_in(client, "owner@example.com")
    avatar_id = (await client.post("/api/avatars", json={**AVATAR, **BODY})).json()["id"]

    await client.post("/api/auth/logout")
    await sign_in(client, "stranger@example.com")

    theirs = await client.patch(f"/api/avatars/{avatar_id}", json={**AVATAR, **BODY})
    nobodys = await client.patch("/api/avatars/no-such-avatar", json={**AVATAR, **BODY})
    assert theirs.status_code == nobodys.status_code == 404
    assert theirs.json()["detail"] == nobodys.json()["detail"]
