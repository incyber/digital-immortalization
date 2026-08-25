"""Which voice speaks which language.

The bug this exists to prevent: an avatar whose language was recorded as the
free-text word "SPANISH" fell through to English prompts, while a globally
configured Spanish voice model read those English words aloud. The result was
not English and not Spanish - it was Spanish phonemes applied to English
spelling, and it was unintelligible.

Two rules follow from that, and both are enforced here rather than left to
configuration:

  A language is a code from this table or it is not a language. Free text
  means somebody eventually types "SPANISH", or "castellano", or "es-ES ",
  and the mismatch is silent.

  The voice follows the avatar, never a global setting. One deployment serves
  avatars in several languages at once, so a single configured voice is wrong
  for all but one of them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    """A language, and the voice that speaks it."""

    locale: str          # BCP-47 language subtag, lowercase
    name: str            # what to show a customer
    piper_voice: str     # rhasspy/piper-voices identifier
    whisper_language: str  # what to tell speech recognition


VOICES: tuple[Voice, ...] = (
    Voice("en", "English", "en_US-hfc_female-medium", "en"),
    Voice("es", "Español", "es_ES-davefx-medium", "es"),
    Voice("pt", "Português", "pt_BR-faber-medium", "pt"),
    Voice("fr", "Français", "fr_FR-siwis-medium", "fr"),
    Voice("de", "Deutsch", "de_DE-thorsten-medium", "de"),
    Voice("it", "Italiano", "it_IT-riccardo-x_low", "it"),
)

BY_LOCALE = {voice.locale: voice for voice in VOICES}
DEFAULT_LOCALE = "en"


class UnsupportedLocale(ValueError):
    """Raised for a language this system has no voice for.

    Fatal at avatar creation rather than at the first call: an avatar nobody
    can understand is worse than one that could not be created.
    """


def normalise_locale(value: str) -> str:
    """Reduce a language to its subtag, or raise.

    Accepts "es", "es-ES", "es_ES" and surrounding whitespace, because those
    are all things a form or an import will produce. Does not accept "Spanish",
    because guessing at language names is how a mismatch gets in quietly.
    """
    cleaned = (value or "").strip().lower().replace("_", "-")
    subtag = cleaned.split("-")[0]

    if subtag in BY_LOCALE:
        return subtag

    raise UnsupportedLocale(
        f"{value!r} is not a supported language. Use one of: "
        + ", ".join(f"{v.locale} ({v.name})" for v in VOICES)
    )


def voice_for(locale: str) -> Voice:
    return BY_LOCALE[normalise_locale(locale)]


def supported() -> list[Voice]:
    return list(VOICES)
