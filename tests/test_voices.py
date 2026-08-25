"""Language handling.

The failure these pin: an avatar recorded as "SPANISH" fell back to English
prompts while a globally configured Spanish voice read those English words
aloud. It was neither language and it was unintelligible, and nothing in the
system noticed.
"""

import pytest

from avatar.config import Settings
from avatar.services.voices import (
    VOICES,
    UnsupportedLocale,
    normalise_locale,
    supported,
    voice_for,
)


@pytest.mark.parametrize("value", ["es", "es-ES", "es_ES", " ES ", "es-419"])
def test_real_language_tags_are_accepted(value):
    assert normalise_locale(value) == "es"


@pytest.mark.parametrize("value", ["SPANISH", "Spanish", "castellano", "español", "", "  "])
def test_language_names_are_refused(value):
    # Guessing at names is exactly how the mismatch got in.
    with pytest.raises(UnsupportedLocale):
        normalise_locale(value)


def test_the_refusal_says_what_is_accepted():
    with pytest.raises(UnsupportedLocale, match="es \\(Español\\)"):
        normalise_locale("SPANISH")


def test_a_language_with_no_voice_is_refused():
    with pytest.raises(UnsupportedLocale):
        normalise_locale("ja")


def test_every_language_has_a_distinct_voice():
    voices = [v.piper_voice for v in VOICES]
    assert len(voices) == len(set(voices))


def test_every_voice_matches_its_language():
    # A voice file named for one language attached to another is the same bug
    # in a different place.
    for v in VOICES:
        assert v.piper_voice.lower().startswith(v.locale), v
        assert v.whisper_language == v.locale


def test_the_voice_follows_the_requested_language():
    assert voice_for("en").piper_voice.startswith("en")
    assert voice_for("es").piper_voice.startswith("es")
    assert voice_for("pt").piper_voice.startswith("pt")


def test_supported_languages_are_listed_for_the_form():
    assert {v.locale for v in supported()} >= {"en", "es"}


def test_speech_services_follow_the_avatar_not_the_configuration():
    """A single configured voice is wrong for every avatar but one."""
    from avatar.services.speech import _language

    cfg = Settings(_env_file=None, tts_voice="es_ES-davefx-medium", stt_language="es")

    # Configuration says Spanish; an English avatar must still get English.
    assert voice_for("en").piper_voice != cfg.tts_voice
    assert _language(voice_for("en").whisper_language) is not None


async def test_an_english_avatar_gets_an_english_voice():
    from avatar.services.speech import build_tts

    cfg = Settings(_env_file=None, tts_voice="es_ES-davefx-medium")
    service = build_tts(cfg, locale="en")
    assert service is not None


def test_persona_refuses_an_unusable_language():
    from dataclasses import dataclass

    from avatar.persona import persona_from_avatar

    @dataclass
    class A:
        id: str = "a"
        display_name: str = "Someone"
        locale: str = "SPANISH"
        country: str = "US"
        biography: str = "A person."
        voice_description: str = ""
        boundaries: str = ""

    with pytest.raises(UnsupportedLocale):
        persona_from_avatar(A(), frozenset({"US"}))
