"""CPU renderer.

Composites a mouth plate onto an idle loop, choosing the plate from the
short-term loudness of the speech audio. Envelope only - no phoneme alignment,
because the TTS stage emits no timings and adding a forced aligner would put a
model dependency into the one component defined to have none.

The result is visibly synthetic. That is the intended trade: it runs anywhere,
needs no GPU and no weights, and lets the call loop, turn timing and barge-in
be developed and tested at full speed without the production renderer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from avatar.renderer.base import AudioChunk, VideoFrame
from avatar.renderer.plates import PLATE_COUNT, AvatarAssets

# Loudness at or above this fraction of the running ceiling selects the widest
# plate. Below the floor, the mouth is closed.
_SILENCE_FLOOR = 0.06
# The ceiling adapts to the speech actually arriving, so a quiet voice still
# reaches wide plates. It decays so one loud burst does not flatten the rest.
_CEILING_DECAY = 0.995
_CEILING_MIN = 200.0


class VisemeRenderer:
    """A RendererStage backed by numpy compositing."""

    def __init__(self, assets: AvatarAssets | Path | str):
        if isinstance(assets, (str, Path)):
            assets = AvatarAssets.load(Path(assets))
        self._assets = assets
        self._cursor = 0  # position in the idle loop, preserved across calls
        self._cancel = asyncio.Event()
        self._ceiling = _CEILING_MIN
        self._last_indices: list[int] = []

    @property
    def fps(self) -> int:
        return self._assets.fps

    @property
    def size(self) -> tuple[int, int]:
        return self._assets.size

    async def prepare(self, avatar_id: str) -> None:
        """Nothing to load: assets arrived through the constructor. Present so
        this class satisfies the same protocol as the GPU backend, which does
        have real work to do here."""
        self._cancel.clear()

    async def cancel(self) -> None:
        """Signal in-flight generators to stop.

        Returns effectively instantly. The generators check this event once per
        frame, and a frame is a numpy slice assignment, so the worst case is
        one frame of compositing - far inside the 100 ms the contract allows.
        """
        self._cancel.set()

    def _compose(self, plate_index: int) -> VideoFrame:
        """One frame: the next idle frame with a mouth plate pasted in."""
        frames = self._assets.idle_frames
        base = frames[self._cursor % len(frames)]
        self._cursor += 1

        x, y, w, h = self._assets.mouth_box
        out = base.copy()
        out[y : y + h, x : x + w] = self._assets.plates[plate_index]
        return VideoFrame(data=out.tobytes(), width=out.shape[1], height=out.shape[0])

    def _plate_indices(self, audio: AudioChunk) -> list[int]:
        """Quantise the audio into one plate index per output frame.

        Windows are exactly 1/fps long, so frame count tracks audio duration
        and lip motion stays aligned with what is being heard.
        """
        samples = np.frombuffer(audio.pcm, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return []

        per_frame = max(1, audio.sample_rate // self.fps)
        n_frames = max(1, samples.size // per_frame)
        usable = samples[: n_frames * per_frame].reshape(n_frames, per_frame)
        rms = np.sqrt((usable**2).mean(axis=1))

        indices: list[int] = []
        for value in rms:
            self._ceiling = max(value, self._ceiling * _CEILING_DECAY, _CEILING_MIN)
            level = value / self._ceiling
            if level < _SILENCE_FLOOR:
                indices.append(0)
            else:
                scaled = (level - _SILENCE_FLOOR) / (1.0 - _SILENCE_FLOOR)
                indices.append(int(np.clip(round(scaled * (PLATE_COUNT - 1)), 0, PLATE_COUNT - 1)))
        return indices

    async def render(self, audio: AudioChunk) -> AsyncIterator[VideoFrame]:
        """Frames lip-synced to one chunk of speech."""
        self._cancel.clear()
        indices = self._plate_indices(audio)
        self._last_indices = indices
        for index in indices:
            if self._cancel.is_set():
                return
            yield self._compose(index)
            # Hand control back so a cancel arriving mid-chunk is seen on the
            # next frame rather than after the whole chunk has been produced.
            await asyncio.sleep(0)

    async def idle(self) -> AsyncIterator[VideoFrame]:
        """Endless closed-mouth frames, paced to the declared frame rate."""
        self._cancel.clear()
        period = 1.0 / self.fps
        while not self._cancel.is_set():
            yield self._compose(0)
            await asyncio.sleep(period)

    def last_plate_indices(self, frames: list[VideoFrame] | None = None) -> list[int]:
        """Plate indices chosen for the most recent render call.

        Exposed for tests: asserting on chosen indices is precise, where
        asserting on pixels would be brittle.
        """
        return list(self._last_indices)
