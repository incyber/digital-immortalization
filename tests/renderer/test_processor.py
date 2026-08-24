"""The processor's job is to never lose audio and to stop video instantly on
barge-in. Both are tested against a fake stage so the assertions are about
wiring rather than about pixels."""
import asyncio

import pytest
import pytest_asyncio
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    OutputImageRawFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from avatar.renderer.base import VideoFrame
from avatar.renderer.processor import RendererProcessor


class FakeStage:
    """Counts what it was asked to do."""

    fps = 25
    size = (8, 8)

    def __init__(self):
        self.prepared = 0
        self.cancels = 0
        self._blank = VideoFrame(data=bytes(8 * 8 * 3), width=8, height=8)

    async def prepare(self, avatar_id):
        self.prepared += 1

    async def cancel(self):
        self.cancels += 1

    async def render(self, audio):
        for _ in range(3):
            yield self._blank

    async def idle(self):
        while True:
            yield self._blank
            await asyncio.sleep(0.01)


@pytest.fixture
def collected():
    return []


async def wire(processor, collected):
    """Give a processor the task manager it would get from a real pipeline.

    RendererProcessor publishes idle frames from a managed background task, so
    it cannot be exercised as a bare object; without this the first idle start
    raises "TaskManager is not initialized".
    """
    from pipecat.clocks.system_clock import SystemClock
    from pipecat.processors.frame_processor import FrameProcessorSetup
    from pipecat.utils.asyncio.task_manager import TaskManager

    manager = TaskManager(loop=asyncio.get_running_loop())
    await processor.setup(
        FrameProcessorSetup(
            clock=SystemClock(), task_manager=manager, pipeline_worker=None
        )
    )

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        collected.append(frame)

    processor.push_frame = capture  # type: ignore[method-assign]
    return processor


@pytest_asyncio.fixture
async def processor(collected):
    return await wire(RendererProcessor(FakeStage()), collected)


async def test_audio_is_passed_through_before_any_video(processor, collected):
    audio = TTSAudioRawFrame(audio=b"\x00\x00" * 100, sample_rate=16000, num_channels=1)
    await processor.process_frame(audio, FrameDirection.DOWNSTREAM)
    assert isinstance(collected[0], TTSAudioRawFrame), "audio must never wait on rendering"


async def test_audio_produces_video(processor, collected):
    audio = TTSAudioRawFrame(audio=b"\x00\x00" * 100, sample_rate=16000, num_channels=1)
    await processor.process_frame(audio, FrameDirection.DOWNSTREAM)
    assert sum(isinstance(f, OutputImageRawFrame) for f in collected) == 3


async def test_interruption_cancels_the_stage(processor, collected):
    await processor.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    assert processor._stage.cancels == 1
    await processor._stop_idle()


async def test_renderer_failure_does_not_swallow_audio(collected):
    class BrokenStage(FakeStage):
        async def render(self, audio):
            raise RuntimeError("gpu fell over")
            yield  # pragma: no cover

    p = await wire(RendererProcessor(BrokenStage()), collected)
    await p.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 100, sample_rate=16000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await p._stop_idle()
    assert any(isinstance(f, TTSAudioRawFrame) for f in collected), (
        "a broken renderer must degrade the video, not break the call"
    )


async def test_bot_stopped_speaking_resumes_idle(processor, collected):
    await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.03)
    await processor._stop_idle()
    assert any(isinstance(f, OutputImageRawFrame) for f in collected), (
        "a video track that stops publishing reads as a dropped connection"
    )
