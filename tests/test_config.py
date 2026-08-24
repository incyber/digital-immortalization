"""Settings is the only place the process reads its environment.

Every other module takes a Settings instance, which is what makes the
local/cloud split in the design a configuration change rather than a code
change.
"""
from avatar.config import Settings


def test_defaults_are_local_first():
    s = Settings(_env_file=None)
    assert s.renderer_backend == "viseme"
    assert s.stt_backend == "mlx"
    assert s.vision_interval_s == 4.0
    assert s.llm_base_url.endswith("/v1")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RENDERER_BACKEND", "musetalk")
    assert Settings(_env_file=None).renderer_backend == "musetalk"
