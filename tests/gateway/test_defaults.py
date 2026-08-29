"""What the product may answer before a family types anything.

A form with fourteen empty boxes gets closed. So the defaults endpoint answers
everything it honestly can - and the whole of this file is about where
"honestly" stops.

A default may shape tone. It may never state a fact about a person. "Warm and
unhurried" is a starting manner and a family overwrites it in the same minute;
a biography, a saying, a habit, the way somebody spoke, are facts about one
specific dead person and can only come from the people who knew them. The
tests below hold that line as a property of the code rather than as a
convention the next person to edit the copy might not know about.
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from avatar.gateway import defaults as defaults_module
from avatar.gateway.app import create_app
from avatar.gateway.defaults import (
    FAMILY_ONLY,
    PREFILLED_KEYS,
    NotADefault,
    defaults_payload,
    resolve,
    starting_values,
)
from avatar.gateway.models import (
    NEUTRAL_BUILD,
    NEUTRAL_HEIGHT_CM,
    Base,
)

ATTESTED = frozenset({"US", "ES"})

# What the create form must end up carrying, and where each part may come from.
REQUIRED_OF_THE_FAMILY = ("display_name", "biography")


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


# ------------------------------------------------------------- the endpoint


async def test_the_form_can_be_opened_before_anybody_has_signed_in(client):
    # It describes the form, not anybody's data, and a person who has not yet
    # made an account still has to be able to see one.
    response = await client.get("/api/avatars/defaults")
    assert response.status_code == 200


async def test_defaults_is_not_mistaken_for_an_avatar_id(client):
    # Routes match in order, and the literal path has to win.
    response = await client.get("/api/avatars/defaults")
    assert response.status_code == 200
    assert "values" in response.json()


async def test_what_comes_back_is_enough_to_open_the_whole_form(client):
    body = (await client.get("/api/avatars/defaults")).json()

    assert body["values"]["locale"]
    assert body["values"]["country"]
    assert body["values"]["voice_description"]
    assert body["values"]["boundaries"]
    assert body["crisis_line"]["number"]
    assert body["body_if_unstated"]["height_cm"] == NEUTRAL_HEIGHT_CM
    assert body["body_if_unstated"]["build"] == NEUTRAL_BUILD.value

    # And a prompt for every box the family has to write in themselves.
    for name in FAMILY_ONLY - {"speech_pace", "speech_humour", "speech_directness"}:
        assert body["placeholders"][name], name


async def test_the_defaults_are_accepted_by_the_form_they_are_for(client):
    # A default the create endpoint would refuse is worse than none: the
    # family meets an error on a field they never touched.
    await client.post(
        "/api/auth/register", json={"email": "a@example.com", "password": "a-long-password"}
    )
    values = (await client.get("/api/avatars/defaults")).json()["values"]

    created = await client.post(
        "/api/avatars",
        json={
            **values,
            "display_name": "Marguerite Chen",
            "biography": "A cellist from Vancouver who taught for thirty years.",
        },
    )
    assert created.status_code == 201


# ------------------------------------------------- nothing invented about a person


def test_no_starting_value_states_anything_about_a_person():
    # The property, not an inspection of the current copy: whatever the values
    # are, none of them may be a field only the family can answer.
    values = starting_values(resolve({}, ATTESTED))
    assert not FAMILY_ONLY & values.keys()
    assert PREFILLED_KEYS >= values.keys()


def test_a_starting_value_that_stated_a_fact_would_fail_loudly(monkeypatch):
    # The guard is checked rather than assumed, because the way this rule gets
    # broken is not somebody editing the constant - it is somebody adding one
    # more helpful line to starting_values. Widening the rule for the length
    # of this test is the only way to see it bite.
    monkeypatch.setattr(
        defaults_module, "FAMILY_ONLY", FAMILY_ONLY | {"voice_description"}
    )
    with pytest.raises(NotADefault, match="must come from the family"):
        starting_values(resolve({}, ATTESTED))


async def test_the_family_is_never_handed_a_life_it_did_not_write(client):
    body = (await client.get("/api/avatars/defaults")).json()
    for name in REQUIRED_OF_THE_FAMILY:
        assert name not in body["values"], name
    for name in FAMILY_ONLY:
        assert name not in body["values"], name
    assert sorted(FAMILY_ONLY) == body["from_the_family_only"]


async def test_the_help_in_an_empty_box_is_a_question_and_not_an_answer(client):
    # A sample answer, however clearly labelled, is still a sentence about a
    # person that the product wrote, and some of it survives into the box.
    body = (await client.get("/api/avatars/defaults")).json()
    assert body["placeholders_are_prompts_not_answers"] is True
    assert "?" in body["placeholders"]["display_name"]
    assert "Where they were from" in body["placeholders"]["biography"]


async def test_a_neutral_body_is_shown_as_what_silence_produces_never_as_an_answer(client):
    # Pre-filling 170cm and "average" would record a height nobody ever
    # stated, and "we were not told" and "they were average" have to stay
    # different things.
    body = (await client.get("/api/avatars/defaults")).json()
    for name in ("height_cm", "build", "shoulders", "posture"):
        assert name not in body["values"], name
    assert "never submitted" in body["body_if_unstated_note"]


async def test_the_guardrail_sentence_is_shown_rather_than_hidden(client):
    # A family that can read it can argue with it, which is the only way it
    # stays true to what they actually want.
    body = (await client.get("/api/avatars/defaults")).json()
    assert "never claim" in body["values"]["boundaries"].lower()


# ------------------------------------------------------ where the form opens


def test_a_request_that_says_nothing_still_opens_somewhere_usable():
    # No language header, no geo header. The country falls back to the first
    # one the operator has attested, and the language then follows from it
    # rather than from a global default - both reported as what they are.
    resolved = resolve({}, ATTESTED)
    assert resolved.locale == "en"
    assert resolved.country in ATTESTED
    assert resolved.country_source == "fallback"
    assert resolved.locale_source == "country"
    assert resolved.crisis_line is not None


def test_the_browsers_language_opens_the_form_in_that_language():
    resolved = resolve({"accept-language": "es-ES,es;q=0.9,en;q=0.5"}, ATTESTED)
    assert resolved.locale == "es"
    assert resolved.locale_source == "accept-language"


def test_a_language_this_product_cannot_speak_is_stepped_over_not_stopped_at():
    # A browser asking for Catalan and then Spanish should get Spanish.
    resolved = resolve({"accept-language": "ca,es;q=0.9,en;q=0.5"}, ATTESTED)
    assert resolved.locale == "es"


def test_the_least_preferred_language_does_not_win():
    resolved = resolve({"accept-language": "en;q=0.2,es;q=0.9"}, ATTESTED)
    assert resolved.locale == "es"


def test_a_malformed_language_header_costs_that_entry_and_nothing_else():
    resolved = resolve({"accept-language": "es;q=banana,en"}, ATTESTED)
    assert resolved.locale in {"en", "es"}
    assert resolved.country in ATTESTED


def test_the_edge_networks_country_is_used_when_it_is_offered():
    resolved = resolve({"cf-ipcountry": "ES"}, ATTESTED)
    assert resolved.country == "ES"
    assert resolved.country_source == "geo-header"
    # And the language follows the country when the browser said nothing.
    assert resolved.locale == "es"
    assert resolved.locale_source == "country"


def test_the_region_in_a_language_tag_is_used_when_there_is_no_geo_header():
    resolved = resolve({"accept-language": "es-ES"}, ATTESTED)
    assert resolved.country == "ES"
    assert resolved.country_source == "accept-language"


def test_a_country_this_product_cannot_serve_is_never_offered():
    # Mexico has a crisis line on file but the operator has not attested it.
    # Pre-filling it hands a family a form that is refused on submit, with an
    # error about crisis lines they can do nothing about.
    resolved = resolve({"cf-ipcountry": "MX", "accept-language": "es-MX"}, ATTESTED)
    assert resolved.country == "ES"
    assert resolved.country_source == "language"


def test_an_unknown_country_code_is_ignored_rather_than_believed():
    # Cloudflare sends XX for unknown and T1 for Tor.
    for value in ("XX", "T1", "", "nonsense"):
        resolved = resolve({"cf-ipcountry": value}, ATTESTED)
        assert resolved.country in ATTESTED, value


def test_a_form_in_one_language_does_not_open_on_another_countrys_crisis_line():
    resolved = resolve({"accept-language": "es"}, ATTESTED)
    assert resolved.country == "ES"
    assert resolved.crisis_line.number == "024"


def test_with_no_country_attested_the_form_still_answers():
    # Nothing can be created yet, which is the correct state for a product
    # that cannot point a distressed person at real help - but the endpoint
    # says so plainly instead of failing.
    payload = defaults_payload({}, frozenset())
    assert payload["country"] is None
    assert payload["crisis_line"] is None
    assert payload["sources"]["country"] == "none-attested"
    assert payload["locale"] == "en"


def test_where_each_answer_came_from_is_reported():
    # "Your browser told us" and "we guessed" are different things to show
    # somebody, and support has no other way to explain a form that opened in
    # the wrong language.
    payload = defaults_payload({"accept-language": "es-ES", "cf-ipcountry": "US"}, ATTESTED)
    assert payload["sources"] == {"locale": "accept-language", "country": "geo-header"}
