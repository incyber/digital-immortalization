"""Assembly-level checks. The pipeline's behaviour is covered by the processor
tests; these pin the wiring decisions that are easy to get silently wrong."""
import pytest

from avatar.config import Settings
from avatar.persona import load_profile
from avatar.realtime.agent import build_renderer


def test_viseme_backend_is_the_local_default():
    r = build_renderer(Settings(_env_file=None))
    assert r.size == (512, 512)
    assert r.fps == 25


def test_musetalk_backend_fails_loudly_rather_than_silently_downgrading():
    # Selecting the GPU backend on a machine without one must be an error. A
    # paying customer silently served the stand-in avatar is worse than a
    # refused start.
    cfg = Settings(_env_file=None, renderer_backend="musetalk")
    with pytest.raises(NotImplementedError, match="sub-project 2"):
        build_renderer(cfg)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown renderer_backend"):
        build_renderer(Settings(_env_file=None, renderer_backend="wishful"))


def test_missing_assets_fall_back_to_the_stand_in(tmp_path):
    r = build_renderer(Settings(_env_file=None), assets_path=str(tmp_path / "nope"))
    assert r is not None


def test_pipeline_places_the_guardrail_before_the_model():
    # Read the source order rather than instantiating the whole graph: the
    # ordering is the invariant, and it must not drift during refactoring.
    from pathlib import Path

    src = Path("src/avatar/realtime/agent.py").read_text()
    assert src.index("CrisisProcessor(profile)") < src.index("aggregators.user()")
    assert src.index("aggregators.user()") < src.index("llm,")


def test_pipeline_places_the_renderer_after_tts():
    from pathlib import Path

    src = Path("src/avatar/realtime/agent.py").read_text()
    assert src.index("            tts,") < src.index("RendererProcessor(stage")
    assert src.index("RendererProcessor(stage") < src.index("transport.output()")


def test_shipped_profile_is_usable_by_the_agent():
    profile = load_profile("src/avatar/profiles/colon.json")
    assert profile["id"] and profile["crisis_line_number"]
