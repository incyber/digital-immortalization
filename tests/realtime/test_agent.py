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


def test_worker_is_awaited_before_the_runner_starts():
    # add_workers is a coroutine. Calling it without awaiting registers nothing
    # and the agent joins a room then does nothing, which is silent.
    from pathlib import Path

    src = Path("src/avatar/realtime/agent.py").read_text()
    assert "await runner.add_workers(task)" in src


def test_video_publisher_sits_between_renderer_and_transport():
    # Pipecat's LiveKit transport discards video frames (write_video_frame
    # returns False), so the publisher must consume them before they reach it.
    from pathlib import Path

    src = Path("src/avatar/realtime/agent.py").read_text()
    assert src.index("RendererProcessor(stage") < src.index("LiveKitVideoPublisher(")
    assert src.index("LiveKitVideoPublisher(") < src.index("transport.output()")


def test_pipecat_livekit_transport_still_lacks_video_output():
    # If a future Pipecat release implements this, the direct publisher becomes
    # redundant and should be removed. This test is the reminder.
    import inspect

    from pipecat.transports.livekit.transport import LiveKitOutputTransport

    source = inspect.getsource(LiveKitOutputTransport.write_video_frame)
    assert "return False" in source, (
        "Pipecat's LiveKit transport now writes video; drop LiveKitVideoPublisher"
    )


def test_vad_is_attached_to_the_aggregator_not_the_transport():
    # LiveKitParams has no vad_analyzer field and pydantic ignores unknown
    # keys, so passing it there is silently accepted and the agent then hears
    # nothing at all. Assert the field really is absent, so this cannot be
    # "fixed" back to the transport without the test failing.
    from pipecat.transports.livekit.transport import LiveKitParams

    assert "vad_analyzer" not in LiveKitParams.model_fields

    from pathlib import Path

    src = Path("src/avatar/realtime/agent.py").read_text()
    assert "LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())" in src
