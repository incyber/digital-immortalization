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
    # Where this chunk begins on the session's speech timeline - seconds of
    # synthesised audio committed so far, not wall-clock time.
    #
    # The distinction decides whether motion can be timed at all. The model
    # produces a sentence before the voice synthesises it, and the voice
    # synthesises it before it is rendered, so a nod that must peak on a
    # stressed syllable can be started before that syllable is heard. Timed on
    # a wall clock, every nod arrives after the beat it was meant to land on.
    #
    # Defaulted so every existing caller and test stays valid.
    t0: float = 0.0

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate


@runtime_checkable
class MotionSource(Protocol):
    """A continuous stream of animation parameters on the speech timeline.

    Separate from the renderer because motion is not pixels. What the head and
    face are doing can be computed, asserted on and reviewed with no GPU and no
    renderer at all, which is what keeps the expensive half of this system
    testable on a laptop.
    """

    def pose_at(self, t: float) -> object:
        """The pose at time t. Pure: no state change, no allocation, no waiting.

        Called once per frame on the render path, so it must never block. If a
        plan has not been refined yet the caller gets the coarse one rather
        than waiting for a better answer that would arrive late.
        """

    def interrupt(self, t: float) -> None:
        """Abandon planned motion from t onward. Must be effectively immediate.

        Called before the renderer's own cancel() during barge-in, so the face
        is already correct by the time frames resume.
        """

    @property
    def timeline(self) -> float:
        """Seconds of speech committed so far."""


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

    async def attach_motion(self, source: MotionSource | None) -> None:
        """Bind a motion source, or clear it.

        Optional by design. A renderer that only animates a mouth ignores it
        entirely; one that drives a whole face and body reads a pose per frame.
        Adding this as a method rather than a parameter on render() is what
        keeps every existing backend substitutable and the contract suite
        unchanged.
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
