"""How much of a person a family may describe, and what reaches the prompt.

Two lines are held here. The first is that all of it is optional: somebody who
answers nothing beyond a name and a biography gets the avatar they would have
got before any of these fields existed, and it is coherent. The second is that
anything they do answer changes the recreation - a field that did not reach the
prompt would be a question asked of a grieving family for nothing.

The phrases are the delicate one. They must arrive as examples of how somebody
talked, never as a list of lines to get through, and the wording that makes
that difference is asserted rather than assumed.
"""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from avatar.gateway.app import create_app
from avatar.gateway.defaults import PLACEHOLDERS
from avatar.gateway.models import Avatar, Base
from avatar.motion.gesture import GESTURES
from avatar.persona import (
    MANNERISM_MOTION_LIMIT,
    MAX_PHRASES,
    build_system_prompt,
    decode_phrases,
    encode_phrases,
    persona_from_avatar,
)

ATTESTED = frozenset({"US", "ES"})

AVATAR = {
    "display_name": "Marguerite Chen",
    "locale": "en",
    "country": "US",
    "biography": "A cellist from Vancouver who taught for thirty years.",
    "voice_description": "Dry, unhurried, fond of understatement.",
    "boundaries": "",
}

MANNER = {
    "characteristic_phrases": ["Well now, let's see about that", "Play it like you mean it"],
    "mannerisms": "Tapped the table twice before answering anything difficult.",
    "topics_loved": "The orchestra, her students, the ferry to Victoria",
    "topics_to_avoid": "The last year in hospital",
    "caller_relationship": "her daughter Ana",
    "speech_pace": "slow",
    "speech_humour": "dry",
    "speech_directness": "gentle",
}

UNANSWERED = {
    "characteristic_phrases": [],
    "mannerisms": "",
    "topics_loved": "",
    "topics_to_avoid": "",
    "caller_relationship": "",
    "speech_pace": None,
    "speech_humour": None,
    "speech_directness": None,
}


def described(**manner) -> Avatar:
    """An avatar record, unsaved, described however a test needs it."""
    phrases = manner.pop("characteristic_phrases", None)
    return Avatar(
        id="av-1",
        display_name="Marguerite Chen",
        locale=manner.pop("locale", "en"),
        country=manner.pop("country", "US"),
        biography="A cellist from Vancouver who taught for thirty years.",
        voice_description="Dry, unhurried, fond of understatement.",
        boundaries="",
        characteristic_phrases=encode_phrases(phrases),
        **manner,
    )


def prompt_for(**manner) -> str:
    return build_system_prompt(persona_from_avatar(described(**manner), ATTESTED))


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


# ---------------------------------------------------------------- optional


async def test_an_avatar_can_be_created_without_describing_the_manner_at_all(client):
    # The whole set is optional. A family that cannot face these questions
    # today, or does not know the answers, still gets their avatar.
    await sign_in(client)
    response = await client.post("/api/avatars", json=AVATAR)
    assert response.status_code == 201
    assert response.json()["manner"] == UNANSWERED


async def test_an_avatar_described_only_by_a_biography_is_still_a_working_avatar(client):
    await sign_in(client)
    body = (await client.post("/api/avatars", json=AVATAR)).json()
    assert body["callable"] is False  # only for want of photographs
    assert "Marguerite Chen" in body["disclosure"]
    assert body["crisis_line"]["number"] == "988"


def test_an_undescribed_person_still_yields_a_coherent_prompt():
    # Coherent meaning: the name, the biography, the guardrail, the reply
    # instruction and the affect tag, with no dangling clause where a manner
    # sentence would have been and no placeholder left unfilled.
    prompt = prompt_for()
    assert "Marguerite Chen" in prompt
    assert "cellist" in prompt
    assert "never claim" in prompt.lower()
    assert "{" not in prompt and "}" not in prompt
    assert "  " not in prompt
    assert not any(block.strip() == "" for block in prompt.split("\n\n"))


def test_each_part_of_the_manner_is_independent_of_the_others():
    # Answering one question must not oblige a family to invent the rest.
    for field, value in MANNER.items():
        prompt = prompt_for(**{field: value})
        assert "Marguerite Chen" in prompt, field
        assert "{" not in prompt, field


# ------------------------------------------------- what reaches the prompt


def test_a_phrase_the_family_gave_reaches_the_prompt():
    prompt = prompt_for(characteristic_phrases=["Well now, let's see about that"])
    assert "Well now, let's see about that" in prompt


def test_phrases_are_offered_as_examples_rather_than_as_a_script():
    # The failure this defends against is a recreation that answers three
    # questions in a row with the same saying, which turns the thing a family
    # recognised into a parody of the person who said it. The framing is the
    # only thing standing between those two outcomes.
    prompt = prompt_for(characteristic_phrases=["Well now, let's see about that"])
    assert "examples of how they talked" in prompt
    assert "not lines to recite" in prompt
    assert "never more than one in a reply" in prompt
    assert "never work through the list" in prompt


def test_a_phrase_is_quoted_so_its_edges_are_unambiguous():
    prompt = prompt_for(characteristic_phrases=["Play it like you mean it"])
    assert '"Play it like you mean it"' in prompt


def test_only_as_many_phrases_as_one_reply_can_carry_reach_the_model():
    # The family's own list is never edited; this caps what one prompt carries,
    # because a small model handed a long prompt starts reporting it.
    avatar = described(characteristic_phrases=[f"saying-{i}" for i in range(12)])
    persona = persona_from_avatar(avatar, ATTESTED)
    assert len(persona.characteristic_phrases) == MAX_PHRASES
    assert len(decode_phrases(avatar.characteristic_phrases)) == 12


def test_a_habit_reaches_the_prompt():
    prompt = prompt_for(mannerisms="Tapped the table twice before answering.")
    assert "Tapped the table twice before answering" in prompt


def test_a_habit_may_never_become_a_stage_direction():
    # A model handed a physical habit writes "*taps the table* well now", and
    # the voice then reads the asterisks out loud to somebody's daughter.
    prompt = prompt_for(mannerisms="Tapped the table twice before answering.")
    assert "Never write out actions" in prompt
    assert "never describe your own movements" in prompt


def test_a_habit_ending_in_a_full_stop_does_not_produce_two():
    prompt = prompt_for(mannerisms="Tapped the table twice before answering.")
    assert ".." not in prompt


def test_the_subjects_they_returned_to_reach_the_prompt():
    prompt = prompt_for(topics_loved="The orchestra and the ferry to Victoria")
    assert "The orchestra and the ferry to Victoria" in prompt


def test_a_subject_to_avoid_is_a_trailing_instruction_and_not_background():
    # Suppression is only obeyed near the end of a prompt, which is the same
    # reason the boundaries live there. Up beside the biography it reads as
    # something about the person rather than something to do.
    prompt = prompt_for(
        topics_loved="The orchestra", topics_to_avoid="The last year in hospital"
    )
    assert prompt.index("The orchestra") < prompt.index("The last year in hospital")
    assert "Do not raise these subjects" in prompt


def test_who_is_calling_reaches_the_prompt_before_anything_is_said():
    # Register is chosen on the first token, so a model that learns who it is
    # talking to late has already chosen wrong.
    prompt = prompt_for(caller_relationship="her daughter Ana")
    assert "her daughter Ana" in prompt
    assert prompt.index("her daughter Ana") < prompt.index("Reply in 1 to 3")


def test_each_speech_dial_becomes_an_instruction_about_the_sentence():
    # Rendered as something to do, not as an adjective: "they spoke slowly" is
    # a fact a model agrees with and ignores.
    assert "Keep your sentences short" in prompt_for(speech_pace="slow")
    assert "Do not add humour" in prompt_for(speech_humour="none")
    assert "even when it was unwelcome" in prompt_for(speech_directness="blunt")


def test_being_told_they_never_joked_is_not_the_same_as_not_being_told():
    # The distinction the enum exists for. One suppresses humour, the other
    # says nothing at all about it.
    assert "Do not add humour" in prompt_for(speech_humour="none")
    assert "humour" not in prompt_for().lower()


def test_the_manner_is_described_in_the_language_the_avatar_speaks():
    prompt = prompt_for(
        locale="es",
        country="ES",
        speech_pace="slow",
        characteristic_phrases=["Ya veremos"],
        caller_relationship="su hija Ana",
    )
    assert "Usa frases cortas" in prompt
    assert "no frases para recitar" in prompt
    assert "Con quién hablas: su hija Ana" in prompt


def test_a_dial_nobody_recognises_costs_that_dial_and_nothing_else():
    # An import or an older client sending a word this version does not know
    # must not take the call down with it.
    prompt = prompt_for(speech_pace="galloping", speech_humour="dry")
    assert "Marguerite Chen" in prompt
    assert "dry and understated" in prompt


def test_a_phrase_column_that_will_not_parse_never_stops_a_call():
    # This sits between a database column and somebody waiting to hear their
    # mother. No stored text is worth failing that for; the worst case is one
    # oddly shaped phrase.
    assert decode_phrases("{not json") == ("{not json",)
    assert decode_phrases(None) == ()
    assert decode_phrases("") == ()
    assert decode_phrases('["  ", "kept"]') == ("kept",)


# ------------------------------------------------------ storing and editing


async def test_what_the_family_said_is_stored_and_read_back_unchanged(client):
    await sign_in(client)
    created = (await client.post("/api/avatars", json={**AVATAR, **MANNER})).json()
    fetched = (await client.get(f"/api/avatars/{created['id']}")).json()
    assert fetched["manner"] == MANNER


async def test_a_family_can_describe_the_person_later(client):
    # These are remembered in pieces, weeks apart, and the form has to accept
    # them that way.
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json=AVATAR)).json()["id"]

    updated = await client.patch(
        f"/api/avatars/{avatar_id}", json={**AVATAR, "mannerisms": "Hummed while cooking."}
    )
    assert updated.status_code == 200
    assert updated.json()["manner"]["mannerisms"] == "Hummed while cooking."


async def test_a_client_that_does_not_send_the_manner_cannot_erase_it(client):
    # An older client, or a form that only edits the name, must not wipe out
    # what somebody took the trouble to write down.
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json={**AVATAR, **MANNER})).json()["id"]

    kept = (await client.patch(f"/api/avatars/{avatar_id}", json=AVATAR)).json()
    assert kept["manner"] == MANNER


async def test_a_family_can_take_a_described_habit_back(client):
    # Sent explicitly as null, which is a decision rather than an omission.
    await sign_in(client)
    avatar_id = (await client.post("/api/avatars", json={**AVATAR, **MANNER})).json()["id"]

    body = (
        await client.patch(
            f"/api/avatars/{avatar_id}",
            json={**AVATAR, "mannerisms": None, "speech_humour": None},
        )
    ).json()
    assert body["manner"]["mannerisms"] == ""
    assert body["manner"]["speech_humour"] is None


async def test_a_phrase_that_is_really_a_paragraph_is_refused(client):
    # It would be quoted back verbatim as a line the person used to say.
    await sign_in(client)
    response = await client.post(
        "/api/avatars", json={**AVATAR, "characteristic_phrases": ["and then " * 60]}
    )
    assert response.status_code == 422


async def test_more_phrases_than_one_reply_can_carry_are_refused(client):
    await sign_in(client)
    response = await client.post(
        "/api/avatars",
        json={**AVATAR, "characteristic_phrases": [f"saying {i}" for i in range(MAX_PHRASES + 1)]},
    )
    assert response.status_code == 422


async def test_a_dial_that_is_not_one_of_the_words_offered_is_refused(client):
    await sign_in(client)
    for field, value in (
        ("speech_pace", "galloping"),
        ("speech_humour", "hilarious"),
        ("speech_directness", "evasive"),
    ):
        response = await client.post("/api/avatars", json={**AVATAR, field: value})
        assert response.status_code == 422, field


async def test_a_refused_answer_never_becomes_a_half_described_person(client):
    await sign_in(client)
    await client.post("/api/avatars", json={**AVATAR, "speech_pace": "galloping"})
    assert (await client.get("/api/avatars")).json()["avatars"] == []


# ------------------------------------------------------------- one tenant


async def test_another_tenant_cannot_read_what_a_family_wrote_about_their_dead(client):
    await sign_in(client, "owner@example.com")
    avatar_id = (await client.post("/api/avatars", json={**AVATAR, **MANNER})).json()["id"]

    await sign_in(client, "stranger@example.com")
    response = await client.get(f"/api/avatars/{avatar_id}")
    assert response.status_code == 404
    assert "hospital" not in response.text
    assert (await client.get("/api/avatars")).json()["avatars"] == []


async def test_another_tenant_cannot_rewrite_how_somebody_elses_father_spoke(client):
    await sign_in(client, "owner@example.com")
    avatar_id = (await client.post("/api/avatars", json={**AVATAR, **MANNER})).json()["id"]

    await sign_in(client, "stranger@example.com")
    attempt = await client.patch(
        f"/api/avatars/{avatar_id}",
        json={**AVATAR, "mannerisms": "Shouted at everyone", "speech_humour": "playful"},
    )
    assert attempt.status_code == 404

    await sign_back_in(client, "owner@example.com")
    assert (await client.get(f"/api/avatars/{avatar_id}")).json()["manner"] == MANNER


# ------------------------------------------------------------------ face


def test_a_described_facial_mannerism_is_not_promised_to_the_face():
    # The over-promise this guards against: a family writes "one eyebrow up
    # when he was doubtful", reads nothing to the contrary, and then watches
    # for an eyebrow that cannot happen. Three things are missing, and the
    # third cannot be worked around from here - splat/rig.py averages the two
    # brow channels before driving the expression basis, because the basis has
    # no per-side brow direction, so a one-sided raise is not representable at
    # all. See MANNERISM_MOTION_LIMIT in persona.py for the full account.
    #
    # If a brow gesture is ever added, this test fails, and the copy below has
    # to be revisited on the same day rather than left claiming less than the
    # product now does.
    assert all("brow" not in name for name in GESTURES)
    assert "speech only" in MANNERISM_MOTION_LIMIT

    # And the family is told so, in the words they actually read.
    assert "not how the face moves" in PLACEHOLDERS["en"]["mannerisms"]
    assert "no cómo se mueve la cara" in PLACEHOLDERS["es"]["mannerisms"]
