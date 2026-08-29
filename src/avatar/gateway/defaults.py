"""What the product may fill in before a family types anything.

A bereaved person opening a form with fourteen empty boxes closes it. So the
product answers everything it honestly can, and the form becomes a
conversation that has already started rather than an intake sheet.

The whole of this module hangs on one distinction, and it is enforced here in
code rather than left to whoever writes the form copy next year:

    A default may shape tone. It may never state a fact about a person.

"Warm, unhurried" is a starting tone - it describes the manner of the
recreation before anyone has said anything about the person, and a family will
overwrite it in the same minute. A biography, a phrase he used, a habit she
had, the way he talked: those are facts about one specific dead person, and
the only source for them is the family. Pre-filling one of those is the
product quietly inventing somebody's father, and it would never be visible as
an invention afterwards, because it would arrive looking exactly like an
answer.

FAMILY_ONLY below is that rule as a set. PREFILLED_KEYS is what may be handed
back as a starting value. They are checked against each other at import, and
the values are checked again every time they are built, so an edit that
pre-fills a biography fails in the test run rather than on a family's screen.

Free-text fields still get help: a placeholder. Placeholders here are
questions, never sample answers - "where they were from, what they did with
their days" rather than a made-up life. A question cannot be mistaken for
something the product knew, which is the point.

The neutral body is returned but deliberately not pre-filled. See
NEUTRAL_BODY_IS_NOT_AN_ANSWER.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from avatar.gateway.models import (
    NEUTRAL_BUILD,
    NEUTRAL_HEIGHT_CM,
    NEUTRAL_POSTURE,
    NEUTRAL_SHOULDERS,
)
from avatar.persona import DEFAULT_BOUNDARIES
from avatar.safety.crisis_lines import BY_COUNTRY, CrisisLine, selectable
from avatar.services.voices import BY_LOCALE, DEFAULT_LOCALE

# Fields that may only ever come from the family. Every one of them is a claim
# about one particular person; none of them has a defensible default.
FAMILY_ONLY = frozenset(
    {
        "display_name",
        "biography",
        "characteristic_phrases",
        "mannerisms",
        "topics_loved",
        "topics_to_avoid",
        "caller_relationship",
        "speech_pace",
        "speech_humour",
        "speech_directness",
    }
)

# Fields the product may hand back already answered.
#
# The three speech dials are in FAMILY_ONLY and not here, which looks
# inconsistent beside voice_description and is not. A prose tone is read as
# the product's suggestion and rewritten wholesale; a dial set to "measured"
# is stored as a discrete claim, renders into the prompt as a flat statement
# about how the person spoke, and survives untouched because nothing about it
# looks unanswered.
PREFILLED_KEYS = frozenset({"locale", "country", "voice_description", "boundaries"})

# Why the neutral body is shown but not pre-filled.
#
# models.py keeps "we were not told" and "they were average" as different
# states on purpose, and pre-filling 170cm/average into the form would collapse
# them the first time a family pressed submit: the record would then say a
# height nobody had ever stated. So the neutral body is returned under its own
# key, as what the build will use if nobody says otherwise, and the fields stay
# empty.
NEUTRAL_BODY_IS_NOT_AN_ANSWER = (
    "What the build uses when nothing is stated. Shown so a family knows what "
    "silence produces; never submitted as though they had answered."
)

# A starting tone, and the one place in this file that puts words in the
# recreation's mouth. Phrased as a manner rather than as a description of a
# voice, so that left untouched it says how the recreation should carry itself
# and claims nothing about how anybody actually sounded.
STARTING_VOICE = {
    "en": "Warm and unhurried. Never rushed by a question.",
    "es": "Cálido y sin prisa. Ninguna pregunta le mete prisa.",
}

# Placeholders are questions. A sample answer, however clearly labelled, is
# still a sentence about a person that the product wrote, and some of it
# survives into the box.
PLACEHOLDERS: dict[str, dict[str, str]] = {
    "en": {
        "display_name": "What did the people in this family call them?",
        "biography": (
            "Where they were from, what they did with their days, who was "
            "around them, and the one thing anyone who knew them would say."
        ),
        "voice_description": (
            "How did they sound - quick or slow, loud or quiet, and what did "
            "their voice do when they were pleased?"
        ),
        "characteristic_phrases": (
            "Something they said so often the family can still hear it. A few "
            "words is enough; put each saying on its own."
        ),
        "mannerisms": (
            "Habits anyone who knew them would recognise - how they started a "
            "sentence, what they always did before answering. This shapes how "
            "they speak, not how the face moves."
        ),
        "topics_loved": "What did they always end up talking about?",
        "topics_to_avoid": (
            "Anything this family would rather not hear raised. Left empty, "
            "nothing is off limits."
        ),
        "caller_relationship": (
            "Who will be sitting in front of this - their daughter, their "
            "brother, a grandchild?"
        ),
        "boundaries": (
            "Anything it should refuse to do or claim, beyond what is already "
            "in the box."
        ),
    },
    "es": {
        "display_name": "¿Cómo le llamaban en la familia?",
        "biography": (
            "De dónde era, a qué dedicaba los días, quién estaba a su "
            "alrededor y lo que diría cualquiera que le conociera."
        ),
        "voice_description": (
            "¿Cómo sonaba: deprisa o despacio, alto o bajo, y qué hacía su voz "
            "cuando estaba contento?"
        ),
        "characteristic_phrases": (
            "Algo que decía tantas veces que la familia aún lo oye. Con unas "
            "pocas palabras basta; escribe cada dicho por separado."
        ),
        "mannerisms": (
            "Costumbres que cualquiera reconocería: cómo empezaba una frase, "
            "qué hacía siempre antes de responder. Esto marca cómo habla, no "
            "cómo se mueve la cara."
        ),
        "topics_loved": "¿De qué acababa hablando siempre?",
        "topics_to_avoid": (
            "Lo que esta familia preferiría no oír. Si lo dejas vacío, no hay "
            "ningún tema vetado."
        ),
        "caller_relationship": (
            "¿Quién se va a sentar delante: su hija, su hermano, un nieto?"
        ),
        "boundaries": (
            "Cualquier cosa que deba negarse a hacer o afirmar, además de lo "
            "que ya está en el recuadro."
        ),
    },
}

# Headers a CDN or edge network puts the caller's country in. First one that
# carries a country this product can actually serve wins. Nothing here is
# trusted for anything but a form default: a spoofed header pre-fills a
# dropdown, and the crisis line is still resolved from the stored country by
# the registry, never from this.
GEO_HEADERS = (
    "cf-ipcountry",
    "x-vercel-ip-country",
    "x-appengine-country",
    "x-country-code",
    "x-geo-country",
)


class NotADefault(RuntimeError):
    """Raised when a starting value would state a fact about a person.

    Loud on purpose, and raised at the moment the values are built rather than
    only at import, so it cannot be reached by any route. A failed defaults
    request is a form with empty boxes; the alternative is a family being shown
    an invented life and never being told it was invented.
    """


_leaked = FAMILY_ONLY & PREFILLED_KEYS
if _leaked:
    raise NotADefault(
        f"{sorted(_leaked)} may only come from the family, and cannot be pre-filled"
    )


@dataclass(frozen=True)
class Resolved:
    """Where the language and country came from, as well as what they are.

    The source is reported rather than kept private because "we guessed" and
    "your browser told us" are different things to show somebody, and because
    support otherwise has no way to explain a form that opened in the wrong
    language.
    """

    locale: str
    country: str | None
    locale_source: str
    country_source: str
    crisis_line: CrisisLine | None = None
    attested: frozenset[str] = field(default_factory=frozenset)


def _accept_language_tags(header: str | None) -> list[str]:
    """The tags of an Accept-Language header, best first.

    Ordered by q-value, and a tag with no q is worth 1.0 as the specification
    says. Anything unparseable is dropped rather than defaulting to zero: a
    malformed header should cost that one entry, not the whole preference.
    """
    entries: list[tuple[float, int, str]] = []
    for position, part in enumerate((header or "").split(",")):
        tag, _, params = part.strip().partition(";")
        tag = tag.strip()
        if not tag or tag == "*":
            continue
        quality = 1.0
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        # Position breaks ties, so equally weighted tags keep the order the
        # browser sent them in.
        entries.append((-quality, position, tag))

    return [tag for _, _, tag in sorted(entries)]


def _locale_from_tags(tags: list[str]) -> str | None:
    """The first language we have a voice for.

    Walks past languages this system cannot speak rather than stopping at the
    first one. A browser asking for Catalan then Spanish should get Spanish,
    not English.
    """
    for tag in tags:
        subtag = tag.split("-")[0].lower()
        if subtag in BY_LOCALE:
            return subtag
    return None


def _region_from_tags(tags: list[str]) -> str | None:
    """The region of the first tag that carries one - "es-MX" gives MX."""
    for tag in tags:
        parts = tag.replace("_", "-").split("-")
        if len(parts) > 1 and len(parts[1]) == 2 and parts[1].isalpha():
            return parts[1].upper()
    return None


def _country_from_headers(headers: Mapping[str, str]) -> str | None:
    for name in GEO_HEADERS:
        value = (headers.get(name) or "").strip().upper()
        # Cloudflare sends XX for unknown and T1 for Tor; both fail this
        # lookup on their own, along with anything else made up.
        if value in BY_COUNTRY:
            return value
    return None


def resolve(headers: Mapping[str, str], attested: frozenset[str]) -> Resolved:
    """Work out which language and country to open the form in.

    Never returns a country the operator has not attested a crisis line for.
    Pre-filling one would hand a family a form that is refused on submit, with
    an error about crisis lines they have no way to act on - which is a worse
    experience than the dropdown simply opening somewhere else.
    """
    tags = _accept_language_tags(headers.get("accept-language"))

    locale = _locale_from_tags(tags)
    locale_source = "accept-language" if locale else ""

    serviceable = [line.country for line in selectable(attested)]

    country: str | None = None
    country_source = "none-attested"

    geo = _country_from_headers(headers)
    region = _region_from_tags(tags)

    if geo and geo in serviceable:
        country, country_source = geo, "geo-header"
    elif region and region in serviceable:
        country, country_source = region, "accept-language"
    elif serviceable:
        # Nothing in the request points anywhere we can serve. Prefer a country
        # that at least speaks the language the browser asked for, so a Spanish
        # form does not open on a United States crisis line.
        matching = [c for c in serviceable if BY_COUNTRY[c].locale == locale]
        if matching:
            country, country_source = matching[0], "language"
        else:
            country, country_source = serviceable[0], "fallback"

    if not locale:
        # The country's own language beats the global default: a form opening
        # in Spain should open in Spanish even from a browser that said
        # nothing at all.
        if country is not None:
            locale, locale_source = BY_COUNTRY[country].locale, "country"
        else:
            locale, locale_source = DEFAULT_LOCALE, "fallback"

    return Resolved(
        locale=locale,
        country=country,
        locale_source=locale_source,
        country_source=country_source,
        crisis_line=BY_COUNTRY[country] if country is not None else None,
        attested=attested,
    )


def starting_values(resolved: Resolved) -> dict[str, Any]:
    """The fields the form may open already answered.

    Checked against FAMILY_ONLY on the way out. The check is here, in the
    function that produces the values, rather than only over the constant
    above, because the way this rule gets broken is not somebody editing
    PREFILLED_KEYS - it is somebody adding one more helpful line to this
    function.
    """
    locale = resolved.locale
    values: dict[str, Any] = {
        "locale": locale,
        "country": resolved.country,
        "voice_description": STARTING_VOICE.get(locale, STARTING_VOICE["en"]),
        # The guardrail sentence, shown rather than hidden. A family that can
        # read it can argue with it, which is the only way it stays true to
        # what they want.
        "boundaries": DEFAULT_BOUNDARIES.get(locale, DEFAULT_BOUNDARIES["en"]),
    }

    invented = FAMILY_ONLY & values.keys()
    if invented:
        raise NotADefault(
            f"{sorted(invented)} states something about a specific person and "
            "must come from the family, not from a default"
        )
    return values


def placeholders_for(locale: str) -> dict[str, str]:
    """Prompts for the free-text boxes, in the form's language."""
    return dict(PLACEHOLDERS.get(locale, PLACEHOLDERS["en"]))


def neutral_body() -> dict[str, Any]:
    """What the build falls back on, quoted from models.py rather than repeated."""
    return {
        "height_cm": NEUTRAL_HEIGHT_CM,
        "build": NEUTRAL_BUILD.value,
        "shoulders": NEUTRAL_SHOULDERS.value,
        "posture": NEUTRAL_POSTURE.value,
    }


def defaults_payload(headers: Mapping[str, str], attested: frozenset[str]) -> dict[str, Any]:
    """Everything the create form can be opened with.

    Three separate sections, and they are separate because they carry
    different weight: `values` is answered, `placeholders` is asked, and
    `from_the_family_only` is the list of things this product will not answer
    on a family's behalf. A client that flattened them into one blob would be
    reintroducing exactly the confusion this module exists to prevent.
    """
    resolved = resolve(headers, attested)
    crisis = resolved.crisis_line

    return {
        "locale": resolved.locale,
        "country": resolved.country,
        "values": starting_values(resolved),
        "placeholders": placeholders_for(resolved.locale),
        "placeholders_are_prompts_not_answers": True,
        "from_the_family_only": sorted(FAMILY_ONLY),
        "body_if_unstated": neutral_body(),
        "body_if_unstated_note": NEUTRAL_BODY_IS_NOT_AN_ANSWER,
        "sources": {
            "locale": resolved.locale_source,
            "country": resolved.country_source,
        },
        # A verified fact rather than a default, and shown so a family can see
        # which number this recreation will give somebody in trouble.
        "crisis_line": (
            {"name": crisis.name, "number": crisis.number} if crisis is not None else None
        ),
    }
