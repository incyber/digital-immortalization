"""The character is entirely the customer's, and two parts of it are not."""

from dataclasses import dataclass

import pytest

from avatar.persona import (
    DEFAULT_BOUNDARIES,
    InvalidProfile,
    build_system_prompt,
    persona_from_avatar,
)
from avatar.safety.crisis_lines import UnsupportedCountry
from avatar.vision.state import SceneState

ATTESTED = frozenset({"US", "ES"})


@dataclass
class FakeAvatar:
    id: str = "av-1"
    display_name: str = "Marguerite Chen"
    locale: str = "en"
    country: str = "US"
    biography: str = "A cellist from Vancouver who taught for thirty years."
    voice_description: str = "Dry, unhurried, fond of understatement."
    boundaries: str = ""


def test_a_persona_is_built_from_the_customers_own_avatar():
    persona = persona_from_avatar(FakeAvatar(), ATTESTED)
    assert persona.display_name == "Marguerite Chen"
    assert "cellist" in persona.biography


def test_no_character_ships_with_the_application():
    # There is no default subject and no profile file. An avatar exists only
    # because a customer described one.
    from pathlib import Path

    assert not Path("src/avatar/profiles").exists()


def test_an_avatar_without_a_name_is_refused():
    with pytest.raises(InvalidProfile, match="needs a name"):
        persona_from_avatar(FakeAvatar(display_name="  "), ATTESTED)


def test_an_avatar_without_a_description_is_refused():
    # An empty description does not make a lesser recreation, it makes an
    # invented person.
    with pytest.raises(InvalidProfile, match="description of who the person was"):
        persona_from_avatar(FakeAvatar(biography=""), ATTESTED)


def test_an_unattested_country_prevents_the_avatar_existing():
    with pytest.raises(UnsupportedCountry):
        persona_from_avatar(FakeAvatar(country="MX"), ATTESTED)


def test_boundaries_default_rather_than_being_empty():
    persona = persona_from_avatar(FakeAvatar(boundaries=""), ATTESTED)
    assert persona.boundaries == DEFAULT_BOUNDARIES["en"]
    assert "never claim" in persona.boundaries.lower()


def test_the_customers_own_boundaries_are_used_when_given():
    persona = persona_from_avatar(FakeAvatar(boundaries="Never discusses money."), ATTESTED)
    assert persona.boundaries == "Never discusses money."


def test_the_disclosure_is_generated_not_stored():
    # A customer must not be able to soften or delete it by editing a field.
    persona = persona_from_avatar(FakeAvatar(), ATTESTED)
    assert "Marguerite Chen" in persona.disclosure
    assert "synthetic recreation" in persona.disclosure


def test_the_disclosure_follows_the_locale():
    persona = persona_from_avatar(FakeAvatar(locale="es", country="ES"), ATTESTED)
    assert "recreación sintética" in persona.disclosure


def test_the_crisis_line_comes_from_the_registry_not_the_customer():
    persona = persona_from_avatar(FakeAvatar(country="US"), ATTESTED)
    assert persona.crisis_line.number == "988"
    assert not hasattr(FakeAvatar(), "crisis_line_number")


def test_the_prompt_names_the_customers_person():
    prompt = build_system_prompt(persona_from_avatar(FakeAvatar(), ATTESTED))
    assert "Marguerite Chen" in prompt
    assert "cellist" in prompt


def test_the_prompt_has_no_unfilled_placeholders():
    prompt = build_system_prompt(persona_from_avatar(FakeAvatar(), ATTESTED))
    assert "{" not in prompt and "}" not in prompt


def test_boundaries_come_after_history():
    persona = persona_from_avatar(FakeAvatar(boundaries="Never discusses money."), ATTESTED)
    recent = [{"role": "user", "content": "hello"}]
    prompt = build_system_prompt(persona, SceneState(), recent, now=0.0)
    assert prompt.index("HISTORICAL_CONTEXT") < prompt.index("Never discusses money.")


def test_history_is_capped():
    persona = persona_from_avatar(FakeAvatar(), ATTESTED)
    recent = [{"role": "user", "content": f"turn-{i}"} for i in range(40)]
    prompt = build_system_prompt(persona, SceneState(), recent, now=0.0)
    assert "turn-0:" not in prompt
    assert "turn-39" in prompt


def test_an_empty_scene_adds_no_observation():
    persona = persona_from_avatar(FakeAvatar(), ATTESTED)
    prompt = build_system_prompt(persona, SceneState(), now=0.0)
    assert "camera" not in prompt.lower()


def test_a_populated_scene_adds_exactly_one_observation():
    persona = persona_from_avatar(FakeAvatar(), ATTESTED)
    scene = SceneState()
    scene.update("a woman in a grey coat", now=0.0)
    prompt = build_system_prompt(persona, scene, now=1.0)
    assert prompt.lower().count("through the camera") == 1


def test_the_persona_dict_carries_what_the_guardrail_needs():
    data = persona_from_avatar(FakeAvatar(), ATTESTED).as_dict()
    assert data["crisis_line_number"] == "988"
    assert data["crisis_line_name"]
    assert data["locale"] == "en"
