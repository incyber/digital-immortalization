"""The publisher must tolerate frames arriving before the room is connected,
because the renderer starts publishing idle frames the moment the pipeline
starts and connection completes a beat later."""
import asyncio

import pytest
from pipecat.frames.frames import OutputImageRawFrame
from pipecat.processors.frame_processor import FrameDirection

from avatar.realtime.video_publisher import LiveKitVideoPublisher


async def wire(publisher, collected):
    from pipecat.clocks.system_clock import SystemClock
    from pipecat.processors.frame_processor import FrameProcessorSetup
    from pipecat.utils.asyncio.task_manager import TaskManager

    await publisher.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(loop=asyncio.get_running_loop()),
            pipeline_worker=None,
        )
    )

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        collected.append(frame)

    publisher.push_frame = capture  # type: ignore[method-assign]
    return publisher


def a_frame(size=(4, 4)):
    w, h = size
    return OutputImageRawFrame(image=bytes(w * h * 3), size=size, format="RGB")


@pytest.fixture
def collected():
    return []


async def test_frames_before_connection_are_dropped_quietly(collected):
    def not_connected():
        raise RuntimeError("cannot access local participant before connecting")

    p = await wire(LiveKitVideoPublisher(not_connected, 4, 4, 25), collected)
    await p.process_frame(a_frame(), FrameDirection.DOWNSTREAM)
    assert collected == [], "video frames are consumed, never forwarded"


async def test_video_frames_are_not_forwarded_downstream(collected):
    # Pipecat's LiveKit transport discards them, so forwarding is pure waste.
    def not_connected():
        raise RuntimeError("not connected")

    p = await wire(LiveKitVideoPublisher(not_connected, 4, 4, 25), collected)
    await p.process_frame(a_frame(), FrameDirection.DOWNSTREAM)
    assert not any(isinstance(f, OutputImageRawFrame) for f in collected)


async def test_rgb_is_expanded_to_rgba_with_opaque_alpha():
    # LiveKit takes RGBA; the renderer emits RGB24. Getting the interleave
    # wrong produces a colour-shifted or transparent avatar.
    captured = {}

    class FakeSource:
        def capture_frame(self, frame):
            captured["data"] = bytes(frame.data)

    class FakeParticipant:
        async def publish_track(self, track, options):
            return None

    class FakeRoom:
        local_participant = FakeParticipant()

    p = LiveKitVideoPublisher(lambda: FakeRoom(), 2, 1, 25)
    frame = OutputImageRawFrame(image=bytes([10, 20, 30, 40, 50, 60]), size=(2, 1), format="RGB")

    import avatar.realtime.video_publisher as mod

    original = mod.rtc.VideoSource
    mod.rtc.VideoSource = lambda w, h: FakeSource()
    mod.rtc.LocalVideoTrack.create_video_track = staticmethod(lambda name, source: object())
    try:
        await p._capture(frame)
    finally:
        mod.rtc.VideoSource = original

    assert captured["data"] == bytes([10, 20, 30, 255, 40, 50, 60, 255])
