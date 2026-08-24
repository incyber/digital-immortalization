"""Measures end of user speech to first avatar audio, over several turns.

Turn one is reported separately and never asserted on. An earlier version of
this test measured exactly one turn and read 2026 ms as the cost of speech
recognition; it was almost entirely model loading. Measured directly:
mlx-whisper takes 868 ms on its first transcription and 72 ms on every one
after, and the language model behaves the same way. Warm-up now runs before the
pipeline starts, but the cold/warm split is kept here because collapsing it is
what made the original number wrong.

Asserts against the median of the warm turns, and only when LATENCY_ASSERT=1 -
the budget in the design is a cloud-hardware target.
"""

import asyncio
import os
import time

import numpy as np
import pytest
from livekit import rtc

from avatar.config import Settings
from avatar.gateway.sessions import mint_token
from tests.e2e.test_call import _open_session

BUDGET_S = 1.5
SPEECH_THRESHOLD = 500
TURNS = 4

UTTERANCES = [
    "Buenos días.",
    "¿Cómo estás hoy?",
    "Háblame del mar.",
    "¿Qué recuerdas?",
]

pytestmark = pytest.mark.skipif(
    os.environ.get("E2E") != "1", reason="set E2E=1 and start infra to run"
)


@pytest.fixture
def cfg():
    return Settings(_env_file=None)


async def test_time_to_first_reply(cfg):
    from piper import PiperVoice

    session = await _open_session()

    voice = PiperVoice.load(f"{cfg.voices_dir}/{cfg.tts_voice}.onnx")
    sample_rate = voice.config.sample_rate
    utterances = [
        b"".join(c.audio_int16_bytes for c in voice.synthesize(text))
        for text in UTTERANCES[:TURNS]
    ]

    room = rtc.Room()
    drains: set[asyncio.Task] = set()
    heard = asyncio.Event()
    reply_at: list[float] = []
    # Updated on every non-silent frame from the avatar, armed or not. Used to
    # tell whether it has finished the previous reply before the next
    # utterance begins.
    last_voice_at = [0.0]

    @room.on("track_subscribed")
    def _on_track(track, publication, participant):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        async def drain_audio():
            async for event in rtc.AudioStream(track):
                samples = np.frombuffer(event.frame.data, dtype=np.int16)
                if not samples.size or int(np.abs(samples).max()) <= SPEECH_THRESHOLD:
                    continue
                now = time.perf_counter()
                last_voice_at[0] = now
                # armed[0] gates recording, so the tail of the previous reply
                # cannot be timed against the next utterance. Without it the
                # measurement goes negative, because the avatar is still
                # speaking when the next turn starts.
                if armed[0] and not heard.is_set():
                    reply_at.append(now)
                    heard.set()

        task = asyncio.create_task(drain_audio())
        drains.add(task)
        task.add_done_callback(drains.discard)

    armed = [False]

    token = mint_token(cfg, session["room"], identity="timer", name="Timer")
    await room.connect(cfg.livekit_url, token)

    source = rtc.AudioSource(sample_rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("speech", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    samples_per_chunk = sample_rate // 100
    bytes_per_chunk = samples_per_chunk * 2

    async def speak(pcm: bytes) -> float:
        for offset in range(0, len(pcm), bytes_per_chunk):
            block = pcm[offset : offset + bytes_per_chunk]
            if len(block) < bytes_per_chunk:
                block += bytes(bytes_per_chunk - len(block))
            await source.capture_frame(rtc.AudioFrame(block, sample_rate, 1, samples_per_chunk))
            await asyncio.sleep(0.01)
        # The clock starts when the person stops talking, not when the silence
        # is recognised: endpoint detection is inside the budget, not before it.
        return time.perf_counter()

    async def hold_silence(chunks: int) -> None:
        for _ in range(chunks):
            await source.capture_frame(
                rtc.AudioFrame(bytes(bytes_per_chunk), sample_rate, 1, samples_per_chunk)
            )
            await asyncio.sleep(0.01)

    async def wait_until_quiet(quiet_s: float = 1.0, timeout_s: float = 30.0) -> None:
        """Block until the avatar has produced no speech for quiet_s."""
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if time.perf_counter() - last_voice_at[0] >= quiet_s:
                return
            await hold_silence(10)

    measured: list[float] = []
    try:
        await asyncio.sleep(3)

        for pcm in utterances:
            # The avatar must be silent before the turn starts, or its previous
            # reply is what gets measured.
            await wait_until_quiet()
            armed[0] = False
            heard.clear()
            before = len(reply_at)
            spoke_at = await speak(pcm)
            # Armed only once the user has stopped talking, which is also when
            # the clock starts.
            armed[0] = True
            await hold_silence(150)

            try:
                await asyncio.wait_for(heard.wait(), timeout=60)
            except TimeoutError:
                continue

            if len(reply_at) > before:
                elapsed = reply_at[-1] - spoke_at
                assert elapsed > 0, "measured a reply that began before the utterance ended"
                measured.append(elapsed)

            armed[0] = False
    finally:
        await room.disconnect()

    assert measured, "the avatar never replied, so there is nothing to measure"

    print(f"\n  turn 1 (cold):  {measured[0] * 1000:7.0f} ms")
    for i, value in enumerate(measured[1:], start=2):
        print(f"  turn {i} (warm):  {value * 1000:7.0f} ms")

    warm = sorted(measured[1:])
    if warm:
        median = warm[len(warm) // 2]
        print(f"  warm median:    {median * 1000:7.0f} ms")
    print(f"  design budget:  {BUDGET_S * 1000:7.0f} ms  (cloud hardware)")

    if os.environ.get("LATENCY_ASSERT") == "1":
        assert warm, "no warm turns were measured"
        median = warm[len(warm) // 2]
        assert median <= BUDGET_S, f"warm median {median:.2f}s exceeds the {BUDGET_S}s budget"
