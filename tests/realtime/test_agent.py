"""Assembly-level checks. The pipeline's behaviour is covered by the processor
tests; these pin the wiring decisions that are easy to get silently wrong."""
import pytest

from avatar.config import Settings
from avatar.realtime.agent import build_renderer


def test_viseme_backend_is_the_local_default():
    r = build_renderer(Settings(_env_file=None))
    assert r.size == (512, 512)
    assert r.fps == 25


def test_musetalk_backend_fails_loudly_rather_than_silently_downgrading():
    # Selecting the GPU backend with no service to reach must be an error. A
    # paying customer silently served the stand-in avatar is worse than a
    # refused start.
    cfg = Settings(_env_file=None, renderer_backend="musetalk", musetalk_url="")
    with pytest.raises(ValueError, match="no GPU service"):
        build_renderer(cfg)


def test_musetalk_backend_is_built_when_a_service_is_configured():
    cfg = Settings(
        _env_file=None, renderer_backend="musetalk", musetalk_url="http://gpu:7100"
    )
    r = build_renderer(cfg)

    assert r.size == (cfg.video_width, cfg.video_height)
    assert r.fps == 25


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


def test_no_character_is_shipped_with_the_application():
    # The agent loads whoever the customer described; nothing is bundled.
    from pathlib import Path

    src = Path("src/avatar/realtime/agent.py").read_text()
    assert "load_persona" in src
    assert "profiles/" not in src


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


def _a_manifest():
    from avatar.marking.manifest import SessionManifest

    return SessionManifest(
        session_id="call-1",
        avatar_id="colon",
        avatar_display_name="Cristóbal Colón",
        consent_record_id="c-1",
        rights_holder="Public domain",
        started_at="2026-08-24T12:00:00Z",
        watermark_payload="00",
    )


async def test_declare_or_raise_succeeds_once_the_room_becomes_reachable():
    from avatar.realtime.agent import declare_or_raise

    class FakeParticipant:
        async def set_metadata(self, *_):
            pass

        async def set_attributes(self, *_):
            pass

    class FakeRoom:
        local_participant = FakeParticipant()

    attempts_made = {"n": 0}

    def get_room():
        # The room is unreachable for the first couple of tries, which is
        # the ordinary case of the transport still finishing its connect -
        # not itself a reason to give up.
        attempts_made["n"] += 1
        if attempts_made["n"] < 3:
            raise RuntimeError("room not connected yet")
        return FakeRoom()

    await declare_or_raise(get_room, _a_manifest(), attempts=5, delay_seconds=0)
    assert attempts_made["n"] == 3


async def test_declare_or_raise_gives_up_and_raises_once_the_budget_is_spent():
    """The bug this closes: a declaration that never publishes must not let
    the call proceed with only a log line nobody reads to show for it."""
    from avatar.realtime.agent import DeclarationFailed, declare_or_raise

    def get_room():
        raise RuntimeError("never connects")

    with pytest.raises(DeclarationFailed):
        await declare_or_raise(get_room, _a_manifest(), attempts=3, delay_seconds=0)


async def test_end_call_undeclared_publishes_the_reason_before_disconnecting():
    from avatar.marking.declare import ATTR_DECLARATION_FAILED
    from avatar.realtime.agent import end_call_undeclared

    order = []

    class FakeParticipant:
        async def set_attributes(self, attrs):
            order.append(("attributes", attrs))

    class FakeRoom:
        local_participant = FakeParticipant()

        async def disconnect(self):
            order.append(("disconnect", None))

    room = FakeRoom()
    await end_call_undeclared(lambda: room)

    assert order[0] == ("attributes", {ATTR_DECLARATION_FAILED: "true"})
    assert order[1][0] == "disconnect"


async def test_end_call_undeclared_still_disconnects_if_publishing_the_reason_fails():
    from avatar.realtime.agent import end_call_undeclared

    disconnected = {"called": False}

    class FakeParticipant:
        async def set_attributes(self, attrs):
            raise RuntimeError("room already gone")

    class FakeRoom:
        local_participant = FakeParticipant()

        async def disconnect(self):
            disconnected["called"] = True

    await end_call_undeclared(lambda: FakeRoom())
    assert disconnected["called"] is True


async def test_end_call_undeclared_is_a_no_op_when_the_room_was_never_reached():
    from avatar.realtime.agent import end_call_undeclared

    def get_room():
        raise RuntimeError("never connected")

    # Must not raise: there is nothing to notify or disconnect.
    await end_call_undeclared(get_room)


def test_the_pipeline_task_and_the_declaration_are_awaited_concurrently():
    # A declaration only awaited after runner.run() returns would mean
    # declaring once the call is already over - runner.run() is what actually
    # starts media flowing (see WorkerRunner._setup_session).
    from pathlib import Path

    src = Path("src/avatar/realtime/agent.py").read_text()
    assert "run_task = asyncio.create_task(runner.run())" in src
    assert src.index("run_task = asyncio.create_task") < src.index("await declare_or_raise(")


def test_a_failed_declaration_cancels_the_pipeline_and_ends_the_call():
    from pathlib import Path

    src = Path("src/avatar/realtime/agent.py").read_text()
    assert "except DeclarationFailed:" in src
    assert src.index("except DeclarationFailed:") < src.index("run_task.cancel()")
    assert src.index("run_task.cancel()") < src.index("await end_call_undeclared(")
