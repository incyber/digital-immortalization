"""Bridges a RendererStage into the Pipecat graph.

Audio is passed through untouched and mirrored into the renderer, so the video
track is a consequence of the audio rather than a parallel source that could
drift from it. That is also why a renderer failure degrades to idle frames
rather than silence: the design's ordering is that audio continuity outranks
video fidelity.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    OutputImageRawFrame,
    StartFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from avatar.renderer.base import AudioChunk, RendererStage


class RendererProcessor(FrameProcessor):
    """Turns speech audio into a published video track."""

    def __init__(self, stage: RendererStage, avatar_id: str = "default"):
        super().__init__()
        self._stage = stage
        self._avatar_id = avatar_id
        self._idle_task: asyncio.Task | None = None
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._stage.prepare(self._avatar_id)
            await self.push_frame(frame, direction)
            self._start_idle()
            return

        if isinstance(frame, TTSAudioRawFrame):
            # Audio first, so it is never delayed by rendering, and so a
            # renderer exception cannot swallow the reply.
            await self.push_frame(frame, direction)
            await self._stop_idle()
            self._speaking = True
            await self._render(frame)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._speaking = False
            await self.push_frame(frame, direction)
            self._start_idle()
            return

        if isinstance(frame, InterruptionFrame):
            # Barge-in. The queued frames must stop arriving faster than the
            # person notices they talked over something.
            await self._stage.cancel()
            self._speaking = False
            await self.push_frame(frame, direction)
            self._start_idle()
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop_idle()
            await self._stage.cancel()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _render(self, frame: TTSAudioRawFrame) -> None:
        chunk = AudioChunk(pcm=frame.audio, sample_rate=frame.sample_rate)
        try:
            async for video in self._stage.render(chunk):
                await self.push_frame(
                    OutputImageRawFrame(
                        image=video.data, size=(video.width, video.height), format="RGB"
                    ),
                    FrameDirection.DOWNSTREAM,
                )
        except Exception as exc:  # noqa: BLE001
            # A call without lip-sync is degraded; a call without audio is
            # broken. The audio for this chunk has already been pushed.
            logger.error(f"renderer failed, falling back to idle frames: {exc}")
            self._start_idle()

    def _start_idle(self) -> None:
        # Uses the processor's own task manager rather than asyncio directly:
        # a raw task is not tracked by the pipeline, so it is not cancelled on
        # shutdown and can be collected before it ever runs.
        if self._idle_task is not None and not self._idle_task.done():
            return
        self._idle_task = self.create_task(self._publish_idle())

    async def _stop_idle(self) -> None:
        if self._idle_task is not None:
            await self.cancel_task(self._idle_task)
            self._idle_task = None

    async def _publish_idle(self) -> None:
        """Publish idle frames until speech resumes.

        A video track that stops publishing during a silence reads as a frozen
        or dropped connection, which is why this runs continuously rather than
        only between turns.
        """
        try:
            async for video in self._stage.idle():
                if self._speaking:
                    return
                await self.push_frame(
                    OutputImageRawFrame(
                        image=video.data, size=(video.width, video.height), format="RGB"
                    ),
                    FrameDirection.DOWNSTREAM,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"idle publishing stopped: {exc}")
