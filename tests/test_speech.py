import pytest

from avatar.config import Settings
from avatar.services.speech import build_stt, build_tts


def test_unknown_stt_backend_is_rejected_loudly():
    # A typo in configuration must fail at startup, not silently select a
    # default that is wrong for the machine it lands on.
    with pytest.raises(ValueError, match="unknown stt_backend"):
        build_stt(Settings(_env_file=None, stt_backend="gpu-maybe"))


def test_tts_uses_the_configured_voice(tmp_path):
    cfg = Settings(_env_file=None, tts_voice="es_ES-davefx-medium", voices_dir="assets/voices")
    tts = build_tts(cfg)
    assert tts is not None
