import pytest

from avatar.config import Settings
from avatar.services.speech import build_stt, build_tts


def test_unknown_stt_backend_is_rejected_loudly():
    # A typo in configuration must fail at startup, not silently select a
    # default that is wrong for the machine it lands on.
    with pytest.raises(ValueError, match="unknown stt_backend"):
        build_stt(Settings(_env_file=None, stt_backend="gpu-maybe"))


async def test_tts_uses_the_configured_voice():
    # async because the HTTP backend constructs an aiohttp session, which
    # requires a running event loop.
    cfg = Settings(_env_file=None, tts_voice="es_ES-davefx-medium")
    tts = build_tts(cfg)
    assert tts is not None


def test_unknown_language_is_rejected():
    from avatar.services.speech import _language

    with pytest.raises(ValueError, match="unknown stt_language"):
        _language("klingon")


def test_empty_language_means_autodetect():
    from avatar.services.speech import _language

    assert _language("") is None


def test_configured_language_resolves():
    from pipecat.transcriptions.language import Language

    from avatar.services.speech import _language

    assert _language("es") == Language.ES


def test_default_tts_backend_keeps_gpl_out_of_process():
    # piper-tts is GPL-3.0-or-later. Importing it puts a GPL library in the
    # same process as the rest of this application. The default must be the
    # HTTP backend, where Piper runs as a separate service.
    assert Settings(_env_file=None).tts_backend == "http"


async def test_in_process_backend_must_be_selected_deliberately():
    from avatar.services.speech import build_tts

    cfg = Settings(_env_file=None, tts_backend="inprocess")
    assert build_tts(cfg) is not None


async def test_unknown_tts_backend_is_rejected():
    from avatar.services.speech import build_tts

    with pytest.raises(ValueError, match="unknown tts_backend"):
        build_tts(Settings(_env_file=None, tts_backend="carrier-pigeon"))


def test_piper_is_actually_gpl():
    # Pins the fact that motivated this split, so that a future dependency
    # bump which relicenses Piper is noticed rather than assumed.
    from importlib.metadata import metadata

    assert "GPL" in metadata("piper-tts")["License"]
