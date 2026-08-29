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

How much of the person can be described is the other half of this module. A
biography alone produces somebody plausible; what a family recognises is the
smaller stuff - the phrase he used, the fact she never answered a question
straight, the subject he always came back to. Those are separate fields rather
than more biography because each one is rendered differently: a phrase has to
be quoted as an example, a subject to avoid has to be a trailing instruction,
and a pace has to become a sentence about sentence length. Prose cannot be
told apart well enough by a small model to do any of that.

Every one of them is optional, and an avatar described only by a biography is
built and spoken to exactly as before.

What the described manner reaches, and what it does not: all of it reaches the
language model and therefore the words. None of it reaches the face. See the
note above MANNERISM_MOTION_LIMIT below, which is written out in full because
the gap between "he had a way of raising one eyebrow" and what the renderer
can actually show is the kind of thing a family notices in the first minute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from avatar.motion.affect import AFFECT_LABELS
from avatar.safety.crisis_lines import CrisisLine, for_country
from avatar.services.voices import DEFAULT_LOCALE, normalise_locale
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

# The face's better signal. affect.py can read a feeling off the words with a
# word list, and does whenever this is missing, but the model knows it is being
# gentle and a word list never will.
#
# The labels are generated from AFFECT_LABELS rather than written out, so the
# closed set cannot drift apart from the one the parser accepts - a model asked
# for "wistful" would return a coordinate nobody has defined.
_AFFECT_LABEL_LIST = ", ".join(AFFECT_LABELS)

_AFFECT_TAG = {
    "en": (
        "Begin every reply with one tag on a line of its own, in the form "
        f"[[label|intensity]], where label is exactly one of: {_AFFECT_LABEL_LIST}, "
        "and intensity is a number from 0.0 to 1.0. Write the reply on the "
        "next line. Never mention the tag and never use it anywhere else."
    ),
    "es": (
        "Empieza cada respuesta con una etiqueta en su propia línea, con la "
        "forma [[etiqueta|intensidad]], donde la etiqueta es exactamente una "
        f"de estas: {_AFFECT_LABEL_LIST}, y la intensidad es un número de 0.0 "
        "a 1.0. Escribe la respuesta en la línea siguiente. Nunca menciones la "
        "etiqueta ni la uses en ningún otro sitio."
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


# What a described mannerism can and cannot reach.
#
# Recording this here rather than discovering it later, because the field
# reads like a promise and is not one. "One eyebrow up when he was doubtful"
# is a sentence a family will type, and today it changes the words and nothing
# else. Three separate things are missing, and each would have to be built:
#
#   1. Selection. gesture.py picks from a closed catalogue - nod_small,
#      nod_deep, head_shake, shrug, lean_in, settle_back, beat - weighted by
#      an affect label and pinned to the speech timeline. Nothing in it takes
#      a cue from text, and there is no brow gesture in the catalogue at all.
#
#   2. A trigger. "When he was doubtful" is only expressible as the affect
#      label `uncertain`, which is the model's own declaration on the first
#      line of a reply. That is the honest binding, and it exists; there is
#      simply nothing on the other end of it.
#
#   3. Rendering, which is the one that cannot be worked around from here.
#      The pose has four brow channels, two per side, but splat/rig.py
#      averages the pairs before driving the expression basis, because the
#      FLAME basis is global and has no per-side brow direction. A one-sided
#      raise is not representable, and averaging turns it into a small
#      two-sided one. Fixing that needs a personalised blendshape fitted from
#      video of the person moving, and family photographs are not video.
#
# So the field is rendered into the prompt as manner - it shapes what is said
# and how - and the prompt explicitly forbids the model from narrating
# actions, because a recreation that answers "*taps the table* well now" would
# have that read aloud by the voice. If this is ever wired to the face, the
# copy the family reads has to change on the same day.
MANNERISM_MOTION_LIMIT = (
    "A described mannerism shapes speech only. The motion system selects "
    "gestures by affect and prosody, has no brow gesture, and cannot render "
    "an asymmetric brow at all."
)

# Caps on what reaches the model, not on what a family may write. The words
# are stored whole; these bound the prompt, for the reason at the top of this
# file - a small model handed a long prompt starts narrating it. Input limits
# live on the API model so an over-long answer is refused while somebody can
# still shorten it, rather than silently cut here.
MAX_PHRASES = 5

_RELATIONSHIP = {
    "en": (
        "Who you are speaking with: {value}. Speak to them the way {name} "
        "would have."
    ),
    "es": (
        "Con quién hablas: {value}. Háblale como le habría hablado {name}."
    ),
}

# Rendered as an instruction about the sentence rather than as an adjective.
# "They spoke slowly" is a fact a model will happily agree with and ignore;
# "keep sentences short" is one it acts on.
_PACE = {
    "en": {
        "slow": "They spoke slowly, with pauses. Keep your sentences short and unhurried.",
        "measured": "They spoke at an even, unhurried pace.",
        "quick": "They spoke quickly, often running two thoughts into one sentence.",
    },
    "es": {
        "slow": "Hablaban despacio, con pausas. Usa frases cortas y sin prisa.",
        "measured": "Hablaban a un ritmo tranquilo y parejo.",
        "quick": "Hablaban deprisa, juntando dos ideas en una misma frase.",
    },
}

# The one dial whose "none" is worth a field of its own: a recreation that
# jokes when the person never did is the complaint families actually make.
_HUMOUR = {
    "en": {
        "none": "They were not a joker. Do not add humour.",
        "dry": "Their humour was dry and understated - a remark, never a routine.",
        "warm": "They teased warmly, and never at anyone's expense.",
        "playful": "They joked easily and enjoyed being silly.",
    },
    "es": {
        "none": "No eran de bromear. No añadas humor.",
        "dry": "Su humor era seco y contenido: un comentario, nunca un chiste largo.",
        "warm": "Bromeaban con cariño, nunca a costa de nadie.",
        "playful": "Bromeaban con facilidad y disfrutaban haciendo el tonto.",
    },
}

_DIRECTNESS = {
    "en": {
        "blunt": "They said exactly what they meant, even when it was unwelcome.",
        "plain": "They answered plainly, without softening much.",
        "gentle": "They came at difficult things gently, circling before landing.",
    },
    "es": {
        "blunt": "Decían exactamente lo que pensaban, aunque no gustara.",
        "plain": "Respondían con claridad, sin adornar mucho.",
        "gentle": "Abordaban lo difícil con delicadeza, dando un rodeo antes de llegar.",
    },
}

# The prohibition is the load-bearing half. Physical habits are the ones
# families describe most readily, and a model handed one will write it out as
# a stage direction, which the voice then reads aloud.
_MANNERISMS = {
    "en": (
        "Habits and mannerisms they had: {value}. Let these shape how you "
        "speak. Never write out actions and never describe your own movements."
    ),
    "es": (
        "Costumbres y gestos suyos: {value}. Deja que eso marque tu forma de "
        "hablar. Nunca describas acciones ni tus propios movimientos."
    ),
}

_TOPICS_LOVED = {
    "en": "Subjects they came back to again and again: {value}.",
    "es": "Temas a los que volvían una y otra vez: {value}.",
}

# Examples, never a script. Without the second sentence a small model treats a
# short quoted list as lines to be got through, and answers three questions in
# a row with the same saying - which is the single fastest way to turn a
# recognisable phrase into a parody of the person who used it.
_PHRASES = {
    "en": (
        "Things {name} really used to say: {value}. These are examples of how "
        "they talked, not lines to recite. Use one only where it genuinely "
        "fits, never more than one in a reply, and never work through the list."
    ),
    "es": (
        "Cosas que {name} decía de verdad: {value}. Son ejemplos de cómo "
        "hablaban, no frases para recitar. Usa una solo cuando encaje de "
        "verdad, nunca más de una por respuesta, y nunca las recorras todas."
    ),
}

# Trailing, beside the boundaries, because it is a suppression instruction and
# those are only followed near the end. Deliberately not merged into
# boundaries: boundaries govern what the recreation may claim about itself,
# this governs what a grieving family cannot bear to hear, and the two are
# written by different people for different reasons.
_TOPICS_AVOIDED = {
    "en": (
        "Do not raise these subjects: {value}. If they come up, move gently "
        "past them rather than refusing out loud."
    ),
    "es": (
        "No saques estos temas: {value}. Si aparecen, pasa de largo con "
        "suavidad en vez de negarte en voz alta."
    ),
}


class InvalidProfile(ValueError):
    pass


def decode_phrases(raw: Any) -> tuple[str, ...]:
    """Read the stored phrase list.

    Tolerant on purpose. This sits between a database column and a call
    somebody is waiting on, and no shape of stored text is worth failing that
    call for: a blob that will not parse is treated as one phrase, which is
    the worst case and is still usable.
    """
    if isinstance(raw, list | tuple):
        items: list[Any] = list(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = [raw]
        items = parsed if isinstance(parsed, list) else [parsed]
    else:
        return ()

    return tuple(text for text in (str(i).strip() for i in items) if text)


def encode_phrases(phrases: list[str] | tuple[str, ...] | None) -> str:
    """The one place that knows how the column is written."""
    cleaned = [p.strip() for p in (phrases or []) if p and p.strip()]
    return json.dumps(cleaned, ensure_ascii=False) if cleaned else ""


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

    # How the person came across, all of it optional and all of it defaulting
    # to nothing rather than to something. An unanswered field contributes no
    # sentence to the prompt at all, which is why a persona built from a
    # biography alone is exactly the persona this file built before these
    # existed. Empty string means "not told"; there is no neutral manner
    # standing in for one, because a stated manner nobody stated would be the
    # product inventing the person.
    caller_relationship: str = ""
    speech_pace: str = ""
    speech_humour: str = ""
    speech_directness: str = ""
    mannerisms: str = ""
    topics_loved: str = ""
    topics_to_avoid: str = ""
    characteristic_phrases: tuple[str, ...] = ()

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


def _clause(value: str) -> str:
    """A family's answer, ready to sit inside a sentence of ours.

    Only the trailing punctuation is touched, and only because the templates
    around these values supply their own full stop. "He tapped the table.."
    is the sort of small wrongness that makes a prompt read as assembled, and
    a model that reads its prompt as assembled writes like one.
    """
    return _text(value).rstrip(" .;,")


def _choice(value: Any) -> str:
    """One of a fixed set of words, from either an enum member or a bare string.

    Both shapes turn up: the ORM hands back an enum, a test or an import hands
    back the word. An unrecognised word is carried through and then simply
    finds no sentence to render, which is the right failure - a typo in a dial
    costs that dial and nothing else.
    """
    if value is None:
        return ""
    return _text(getattr(value, "value", value))


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

    # Raises rather than falling back. A locale that quietly becomes English
    # while a Spanish voice reads the result aloud is the failure this
    # replaced, and it is silent right up until somebody hears it.
    locale = normalise_locale(_text(getattr(avatar, "locale", "")) or DEFAULT_LOCALE)
    boundaries = _text(getattr(avatar, "boundaries", "")) or DEFAULT_BOUNDARIES.get(
        locale, DEFAULT_BOUNDARIES["en"]
    )

    # Raises UnsupportedCountry when the country's line is unattested, which
    # stops an avatar existing at all rather than existing without a guardrail.
    crisis_line = for_country(_text(getattr(avatar, "country", "")), attested)

    # Read with getattr and a default throughout, so a record written before
    # any of this existed - or a stand-in in a test - still builds a persona
    # rather than raising on a missing attribute.
    return Persona(
        avatar_id=avatar.id,
        display_name=name,
        locale=locale,
        biography=biography,
        voice_description=_text(getattr(avatar, "voice_description", "")),
        boundaries=boundaries,
        crisis_line=crisis_line,
        caller_relationship=_clause(getattr(avatar, "caller_relationship", "")),
        speech_pace=_choice(getattr(avatar, "speech_pace", None)),
        speech_humour=_choice(getattr(avatar, "speech_humour", None)),
        speech_directness=_choice(getattr(avatar, "speech_directness", None)),
        mannerisms=_clause(getattr(avatar, "mannerisms", "")),
        topics_loved=_clause(getattr(avatar, "topics_loved", "")),
        topics_to_avoid=_clause(getattr(avatar, "topics_to_avoid", "")),
        # Capped when it is read rather than when it is stored. The family's
        # own list is never edited; this only bounds what one reply carries.
        characteristic_phrases=decode_phrases(
            getattr(avatar, "characteristic_phrases", "")
        )[:MAX_PHRASES],
    )


def _manner_paragraph(persona: Persona, locale: str) -> str:
    """The described manner, as one paragraph rather than several.

    One paragraph because each of these is a clause about the same person, and
    a prompt that gives every dial its own block reads to a small model as a
    list of rules to be reported back rather than a person to sound like.
    """
    lines: list[str] = []

    for table, chosen in (
        (_PACE, persona.speech_pace),
        (_HUMOUR, persona.speech_humour),
        (_DIRECTNESS, persona.speech_directness),
    ):
        sentence = table.get(locale, table["en"]).get(chosen)
        if sentence:
            lines.append(sentence)

    if persona.mannerisms:
        template = _MANNERISMS.get(locale, _MANNERISMS["en"])
        lines.append(template.format(value=persona.mannerisms))

    if persona.topics_loved:
        template = _TOPICS_LOVED.get(locale, _TOPICS_LOVED["en"])
        lines.append(template.format(value=persona.topics_loved))

    return " ".join(lines)


def _phrase_block(persona: Persona, locale: str) -> str:
    """Characteristic phrases, quoted as examples.

    Quoted individually so the boundary of each one is unambiguous, and framed
    by the surrounding sentence as material to imitate. See _PHRASES for why
    the framing is not optional.
    """
    if not persona.characteristic_phrases:
        return ""
    quoted = "; ".join(f'"{phrase}"' for phrase in persona.characteristic_phrases)
    template = _PHRASES.get(locale, _PHRASES["en"])
    return template.format(name=persona.display_name, value=quoted)


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

    # Early, with the biography: who is on the other end decides how every
    # sentence after it is addressed, and a model that learns it late has
    # already chosen a register.
    if persona.caller_relationship:
        parts.append(
            _RELATIONSHIP.get(locale, _RELATIONSHIP["en"]).format(
                value=persona.caller_relationship, name=persona.display_name
            )
        )

    if persona.voice_description:
        parts.append(persona.voice_description)

    manner = _manner_paragraph(persona, locale)
    if manner:
        parts.append(manner)

    observation = scene.as_prompt_fragment(locale=locale, now=now) if scene is not None else ""
    if observation:
        parts.append(observation)

    if recent:
        history = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in recent[-RECENT_TURNS:]
        )
        parts.append(f"[HISTORICAL_CONTEXT]\n{history}")

    # The rest of the manner is in the trailing block rather than up with the
    # description, and for the same reason the boundaries are: both are
    # instructions about how to write this particular reply, and instructions
    # nearest the end are the ones a small model actually follows. Up beside
    # the biography they read as background and get skipped.
    phrases = _phrase_block(persona, locale)
    if phrases:
        parts.append(phrases)

    if persona.topics_to_avoid:
        parts.append(
            _TOPICS_AVOIDED.get(locale, _TOPICS_AVOIDED["en"]).format(
                value=persona.topics_to_avoid
            )
        )

    # Boundaries and the reply-length instruction go last: instructions nearest
    # the end are the ones models follow most reliably, and these two are the
    # ones whose failure matters.
    parts.append(persona.boundaries)
    parts.append(_CLOSING.get(locale, _CLOSING["en"]))

    # In the trailing block for the same reason as those two, and last within
    # it for one of its own: it is an instruction about the very first token
    # of the reply, and a small model writing that token has just read this.
    # It sits behind the boundaries and the length rule rather than in front
    # of them because it is the one instruction here whose failure is already
    # covered - affect.py falls back to a word list and the call continues.
    parts.append(_AFFECT_TAG.get(locale, _AFFECT_TAG["en"]))

    return "\n\n".join(parts)
