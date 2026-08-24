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


async def test_frames_are_marked_as_synthetic_before_publishing(collected):
    """No frame reaches a viewer unmarked. This is the Article 50 obligation."""
    import numpy as np

    from avatar.marking.watermark import detect

    captured = {}

    class FakeSource:
        def capture_frame(self, frame):
            captured["data"] = bytes(frame.data)

    class FakeParticipant:
        async def publish_track(self, track, options):
            return None

    class FakeRoom:
        local_participant = FakeParticipant()

    payload = b"avtr" + bytes([1, 9, 9, 9])
    p = LiveKitVideoPublisher(lambda: FakeRoom(), 256, 256, 25, watermark_payload=payload)

    rng = np.random.default_rng(3)
    rgb = rng.integers(60, 200, (256, 256, 3), dtype=np.uint8)
    frame = OutputImageRawFrame(image=rgb.tobytes(), size=(256, 256), format="RGB")

    import avatar.realtime.video_publisher as mod

    original = mod.rtc.VideoSource
    mod.rtc.VideoSource = lambda w, h: FakeSource()
    mod.rtc.LocalVideoTrack.create_video_track = staticmethod(lambda name, source: object())
    try:
        await p._capture(frame)
    finally:
        mod.rtc.VideoSource = original

    published = np.frombuffer(captured["data"], dtype=np.uint8).reshape(256, 256, 4)
    assert detect(published[:, :, :3]) == payload


async def test_an_unmarkable_frame_is_dropped_rather_than_published(collected):
    """Publishing an unmarked synthetic frame is the one outcome not allowed."""
    import numpy as np

    captured = {}

    class FakeSource:
        def capture_frame(self, frame):
            captured["called"] = True

    class FakeParticipant:
        async def publish_track(self, track, options):
            return None

    class FakeRoom:
        local_participant = FakeParticipant()

    # 8x8 is below the watermark's minimum block capacity, so embed() raises.
    p = LiveKitVideoPublisher(lambda: FakeRoom(), 8, 8, 25, watermark_payload=b"avtr\x01\x00\x00\x00")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    frame = OutputImageRawFrame(image=rgb.tobytes(), size=(8, 8), format="RGB")

    import avatar.realtime.video_publisher as mod

    original = mod.rtc.VideoSource
    mod.rtc.VideoSource = lambda w, h: FakeSource()
    mod.rtc.LocalVideoTrack.create_video_track = staticmethod(lambda name, source: object())
    try:
        await p._capture(frame)
    finally:
        mod.rtc.VideoSource = original

    assert "called" not in captured, "an unmarkable frame must not be published"
