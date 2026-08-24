"""The renderer boundary.

This is the only component in the system that needs a GPU, and the only one
carrying unresolved licence exposure from upstream model weights. Both facts
are contained by keeping every implementation behind one small protocol:

    VisemeRenderer    CPU, no weights, runs anywhere. Development and tests.
    MuseTalkRenderer  CUDA, real lip-sync. Production. Sub-project 2.

Both satisfy the same contract tests. The call loop, turn timing and barge-in
behave identically under either, so only pixels differ between a laptop and a
GPU server.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class VideoFrame:
    """One rendered frame, raw RGB24, row-major, no padding.

    len(data) is always width * height * 3. Raw rather than encoded because the
    transport encodes it again on the way out; encoding here would be wasted
    work on the latency path.
    """

    data: bytes
    width: int
    height: int


@dataclass(frozen=True)
class AudioChunk:
    """Signed 16-bit little-endian mono PCM, as produced by the TTS stage."""

    pcm: bytes
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


@runtime_checkable
class RendererStage(Protocol):
    """Turns speech audio into frames of a talking likeness."""

    async def prepare(self, avatar_id: str) -> None:
        """Load weights and per-avatar state. Called once per session, before
        any render call. Expensive work belongs here, not on the turn path."""

    def idle(self) -> AsyncIterator[VideoFrame]:
        """Endless frames for when the likeness is not speaking.

        Endless is deliberate: a call has long silences, and a video track that
        stops publishing reads as a frozen or dropped connection.
        """

    def render(self, audio: AudioChunk) -> AsyncIterator[VideoFrame]:
        """Frames lip-synced to one chunk of speech.

        Yields incrementally. Waiting for the whole chunk would add its full
        duration to time-to-first-frame.
        """

    async def cancel(self) -> None:
        """Abandon in-flight rendering. Must return within 100 ms.

        This bound is what makes barge-in feel instant: when a person talks
        over the likeness, the frames already queued have to stop arriving
        faster than the person notices. Enforced by the contract tests.
        """

    @property
    def fps(self) -> int: ...

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of every frame this stage emits."""
