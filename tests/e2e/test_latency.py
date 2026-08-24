"""Measures end of user speech to first avatar audio.

Always reports. Asserts only when LATENCY_ASSERT=1, because the budget in the
design is a cloud-hardware target and this machine runs a quantised Whisper, a
3B model and a CPU renderer. The number is still the one that matters: if it
drifts on the same hardware, something regressed.
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
    pcm = b"".join(c.audio_int16_bytes for c in voice.synthesize("Buenos días."))
    sample_rate = voice.config.sample_rate

    room = rtc.Room()
    first_audio: list[float] = []
    first_video: list[float] = []
    drains: set[asyncio.Task] = set()
    heard = asyncio.Event()

    @room.on("track_subscribed")
    def _on_track(track, publication, participant):
        async def drain_audio():
            async for event in rtc.AudioStream(track):
                samples = np.frombuffer(event.frame.data, dtype=np.int16)
                if samples.size and int(np.abs(samples).max()) > SPEECH_THRESHOLD:
                    first_audio.append(time.perf_counter())
                    heard.set()
                    return

        async def drain_video():
            async for _ in rtc.VideoStream(track):
                first_video.append(time.perf_counter())
                return

        runner = drain_audio if track.kind == rtc.TrackKind.KIND_AUDIO else drain_video
        task = asyncio.create_task(runner())
        drains.add(task)
        task.add_done_callback(drains.discard)

    token = mint_token(cfg, session["room"], identity="timer", name="Timer")
    await room.connect(cfg.livekit_url, token)

    source = rtc.AudioSource(sample_rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("speech", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    try:
        await asyncio.sleep(4)  # let the agent settle and load its models

        samples_per_chunk = sample_rate // 100
        bytes_per_chunk = samples_per_chunk * 2
        for offset in range(0, len(pcm), bytes_per_chunk):
            block = pcm[offset : offset + bytes_per_chunk]
            if len(block) < bytes_per_chunk:
                block += bytes(bytes_per_chunk - len(block))
            await source.capture_frame(rtc.AudioFrame(block, sample_rate, 1, samples_per_chunk))
            await asyncio.sleep(0.01)

        # The clock starts when the person stops talking, not when the silence
        # is recognised: endpoint detection is inside the budget, not before it.
        spoke_at = time.perf_counter()

        for _ in range(150):
            await source.capture_frame(
                rtc.AudioFrame(bytes(bytes_per_chunk), sample_rate, 1, samples_per_chunk)
            )
            await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(heard.wait(), timeout=60)
        except TimeoutError:
            pass
    finally:
        await room.disconnect()

    assert first_audio, "the avatar never replied, so there is nothing to measure"
    elapsed = first_audio[0] - spoke_at

    print(f"\n  end of speech -> first avatar audio: {elapsed * 1000:.0f} ms")
    print(f"  design budget (cloud hardware):      {BUDGET_S * 1000:.0f} ms")

    if os.environ.get("LATENCY_ASSERT") == "1":
        assert elapsed <= BUDGET_S, f"{elapsed:.2f}s exceeds the {BUDGET_S}s budget"
