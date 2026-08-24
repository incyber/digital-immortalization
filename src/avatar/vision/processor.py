"""Camera frames in, one sentence of scene description out.

Runs beside the conversation rather than inside it. The description is written
to SceneState; nothing in the turn path ever awaits this work.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, InputImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from avatar.config import Settings
from avatar.vision.describe import describe_frame, encode_frame
from avatar.vision.sampler import MotionGate
from avatar.vision.state import SceneState


class VisionSampler(FrameProcessor):
    """Selects frames worth describing and describes them in the background."""

    def __init__(
        self, scene: SceneState, cfg: Settings, on_observation=None, locale: str = "en"
    ):
        super().__init__()
        self._scene = scene
        self._locale = locale
        self._cfg = cfg
        self._gate = MotionGate(cfg.vision_interval_s, cfg.vision_motion_threshold)
        self._on_observation = on_observation
        self._inflight: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputImageRawFrame):
            try:
                self._maybe_describe(frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"vision sampling skipped this frame: {exc}")

        await self.push_frame(frame, direction)

    def _maybe_describe(self, frame: InputImageRawFrame) -> None:
        # One description at a time. Without this, a slow vision model would
        # queue calls faster than they complete and the interval ceiling would
        # stop bounding anything.
        if self._inflight is not None and not self._inflight.done():
            return

        width, height = frame.size
        rgb = np.frombuffer(frame.image, dtype=np.uint8).reshape(height, width, 3)
        if not self._gate.should_send(rgb, now=time.monotonic()):
            return

        jpeg = encode_frame(rgb)
        self._inflight = asyncio.create_task(self._describe(jpeg))

    async def _describe(self, jpeg: bytes) -> None:
        text = await describe_frame(
            jpeg,
            self._cfg.vlm_model,
            self._cfg.vlm_base_url,
            self._cfg.vision_timeout_s,
            locale=self._locale,
        )
        if not text:
            return  # failure leaves the previous observation standing
        self._scene.update(text)
        logger.info(f"scene: {self._scene.description}")
        if self._on_observation is not None:
            await self._on_observation(text)
