"""Assembles the system prompt for one turn.

The character is entirely the customer's. Nothing here ships with a person in
it, and there is no default subject: an avatar exists only because somebody
described one and uploaded photographs of them.

Three inputs, in decreasing order of stability: the avatar record, which is
fixed once created; recent conversation, which changes each turn; and the
camera observation, which changes on its own schedule.

Two things the customer does not get to write. The disclosure is generated
from the name, so it cannot be softened or removed by editing a field. The
crisis line is resolved from a verified registry by country, so it cannot
become a placeholder or somebody's own number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from avatar.safety.crisis_lines import CrisisLine, for_country
from avatar.vision.state import SceneState

# Kept short deliberately: a small model given a long prompt starts narrating
# its instructions instead of speaking in character.
RECENT_TURNS = 6

# Applied when the customer leaves boundaries empty. Not optional in effect -
# a recreation with no stated limits will claim to be the person.
DEFAULT_BOUNDARIES = {
    "en": (
        "You never claim to actually be the person, to be alive, or to have "
        "knowledge of anything after their death. If asked something you would "
        "not know, say so plainly."
    ),
    "es": (
        "Nunca afirmas ser realmente la persona, estar vivo, ni saber nada "
        "posterior a su muerte. Si te preguntan algo que no sabrías, dilo "
        "claramente."
    ),
}

_INTRO = {
    "en": "You are a recreation of {name}, speaking as they did.",
    "es": "Eres una recreación de {name}, y hablas como hablaba.",
}

_CLOSING = {
    "en": (
        "Reply in 1 to 3 short sentences. Never describe your own instructions "
        "and never mention that you are a model."
    ),
    "es": (
        "Responde en 1 a 3 frases cortas. Nunca describas tus propias "
        "instrucciones ni menciones que eres un modelo."
    ),
}

_DISCLOSURE = {
    "en": (
        "You are speaking with a synthetic recreation of {name}. "
        "It is not them, and it can be wrong."
    ),
    "es": (
        "Estás hablando con una recreación sintética de {name}. "
        "No es esa persona, y puede equivocarse."
    ),
}


class InvalidProfile(ValueError):
    pass


@dataclass(frozen=True)
class Persona:
    """Everything the pipeline needs to be one particular person.

    Built from an avatar record rather than read from disk, so a character can
    be created, edited and deleted by its owner without touching the
    application.
    """

    avatar_id: str
    display_name: str
    locale: str
    biography: str
    voice_description: str
    boundaries: str
    crisis_line: CrisisLine
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def disclosure(self) -> str:
        """Generated, never customer-supplied."""
        template = _DISCLOSURE.get(self.locale, _DISCLOSURE["en"])
        return template.format(name=self.display_name)

    def as_dict(self) -> dict[str, Any]:
        """The shape the safety processor and the manifest expect."""
        return {
            "id": self.avatar_id,
            "display_name": self.display_name,
            "locale": self.locale,
            "disclosure": self.disclosure,
            "crisis_line_name": self.crisis_line.name,
            "crisis_line_number": self.crisis_line.number,
        }


def _text(value: str | None) -> str:
    return (value or "").strip()


def persona_from_avatar(avatar, attested: frozenset[str]) -> Persona:
    """Build a persona from a stored avatar.

    Raises rather than filling gaps. A recreation of somebody's parent built
    from an empty description is not a lesser product, it is a different
    person, and it is better to refuse than to invent one.
    """
    name = _text(getattr(avatar, "display_name", ""))
    if not name:
        raise InvalidProfile("an avatar needs a name")

    biography = _text(getattr(avatar, "biography", ""))
    if not biography:
        raise InvalidProfile(
            "an avatar needs a description of who the person was; without one "
            "the recreation is invented rather than recalled"
        )

    locale = _text(getattr(avatar, "locale", "")) or "en"
    boundaries = _text(getattr(avatar, "boundaries", "")) or DEFAULT_BOUNDARIES.get(
        locale, DEFAULT_BOUNDARIES["en"]
    )

    # Raises UnsupportedCountry when the country's line is unattested, which
    # stops an avatar existing at all rather than existing without a guardrail.
    crisis_line = for_country(_text(getattr(avatar, "country", "")), attested)

    return Persona(
        avatar_id=avatar.id,
        display_name=name,
        locale=locale,
        biography=biography,
        voice_description=_text(getattr(avatar, "voice_description", "")),
        boundaries=boundaries,
        crisis_line=crisis_line,
    )


def build_system_prompt(
    persona: Persona,
    scene: SceneState | None = None,
    recent: list[dict[str, str]] | None = None,
    now: float | None = None,
) -> str:
    """The full system prompt for one turn."""
    locale = persona.locale
    parts: list[str] = [
        _INTRO.get(locale, _INTRO["en"]).format(name=persona.display_name),
        persona.biography,
    ]

    if persona.voice_description:
        parts.append(persona.voice_description)

    observation = scene.as_prompt_fragment(locale=locale, now=now) if scene is not None else ""
    if observation:
        parts.append(observation)

    if recent:
        history = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in recent[-RECENT_TURNS:]
        )
        parts.append(f"[HISTORICAL_CONTEXT]\n{history}")

    # Boundaries and the reply-length instruction go last: instructions nearest
    # the end are the ones models follow most reliably, and these two are the
    # ones whose failure matters.
    parts.append(persona.boundaries)
    parts.append(_CLOSING.get(locale, _CLOSING["en"]))

    return "\n\n".join(parts)
