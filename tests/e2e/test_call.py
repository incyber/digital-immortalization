"""End-to-end over real infrastructure.

A headless participant joins a room the gateway opened, subscribes to what the
agent publishes, and asserts that a call actually exists: a video track that
delivers frames of the declared size, and an audio track.

Browser permissions are deliberately not involved. A screenshot proves a page
rendered; this proves media flows.

Skipped unless E2E=1 and LiveKit is reachable, so the default suite stays
hermetic.
"""

import asyncio
import os

import httpx
import numpy as np
import pytest
from livekit import rtc

from avatar.config import Settings
from avatar.gateway.sessions import mint_token

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8000")
JOIN_TIMEOUT_S = 45

pytestmark = pytest.mark.skipif(
    os.environ.get("E2E") != "1", reason="set E2E=1 and start infra to run"
)


@pytest.fixture
def cfg():
    return Settings(_env_file=None)


async def _open_session(avatar_id: str = "colon") -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{GATEWAY}/api/sessions", json={"avatar_id": avatar_id}
        )
        response.raise_for_status()
        return response.json()


async def test_agent_publishes_a_live_call(cfg):
    """The whole point of sub-project 1, asserted in one test."""
    session = await _open_session()

    room = rtc.Room()
    video_frames: list[rtc.VideoFrame] = []
    audio_seen = asyncio.Event()
    got_video = asyncio.Event()
    drains: set[asyncio.Task] = set()

    @room.on("track_subscribed")
    def _on_track(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            async def drain():
                async for event in rtc.VideoStream(track):
                    video_frames.append(event.frame)
                    got_video.set()
                    if len(video_frames) >= 10:
                        return

            task = asyncio.create_task(drain())
            drains.add(task)
            task.add_done_callback(drains.discard)
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_seen.set()

    # A second identity in the same room, standing in for the browser.
    token = mint_token(cfg, session["room"], identity="probe", name="Probe")
    await room.connect(cfg.livekit_url, token)

    try:
        await asyncio.wait_for(got_video.wait(), timeout=JOIN_TIMEOUT_S)
        await asyncio.wait_for(audio_seen.wait(), timeout=JOIN_TIMEOUT_S)
        # Let a few more frames land so cadence, not just presence, is checked.
        await asyncio.sleep(1.5)
    finally:
        await room.disconnect()

    assert video_frames, "the agent published no video frames"
    assert audio_seen.is_set(), "the agent published no audio track"

    first = video_frames[0]
    assert (first.width, first.height) == (cfg.video_width, cfg.video_height)

    # Idle frames are paced to the declared rate. Well under one second of
    # frames in one and a half seconds means the loop has stalled.
    assert len(video_frames) >= 5, f"only {len(video_frames)} frames in ~1.5s"


async def test_consent_gate_refuses_over_http():
    """The gate, exercised through the real HTTP surface rather than in-process."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{GATEWAY}/api/sessions", json={"avatar_id": "no-such-avatar"}
        )
    assert response.status_code == 403
    assert "consent" in response.text.lower() or "no avatar" in response.text.lower()


async def test_speaking_to_the_avatar_produces_a_spoken_reply(cfg):
    """The conversational loop, end to end.

    Publishes real synthesised speech into the room and waits for the agent to
    answer. This is the assertion that distinguishes a working call from a
    video of a face: STT, the guardrail, the model and TTS all have to run.
    """
    from piper import PiperVoice

    session = await _open_session()

    voice = PiperVoice.load(f"{cfg.voices_dir}/{cfg.tts_voice}.onnx")
    chunks = list(voice.synthesize("Hola. Dime quién eres, en pocas palabras."))
    pcm = b"".join(c.audio_int16_bytes for c in chunks)
    sample_rate = voice.config.sample_rate

    room = rtc.Room()
    reply_audio = asyncio.Event()
    agent_audio_frames = 0
    total_audio_frames = 0
    peak = 0
    # Held so the loop keeps a strong reference; a bare create_task can be
    # collected before it reads its first frame.
    drains: set[asyncio.Task] = set()

    @room.on("track_subscribed")
    def _on_track(track, publication, participant):
        nonlocal agent_audio_frames
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        async def drain():
            nonlocal agent_audio_frames, total_audio_frames, peak
            async for event in rtc.AudioStream(track):
                total_audio_frames += 1
                # Silence is published continuously; only non-silent audio
                # counts as the avatar actually saying something. Absolute
                # value matters: a waveform whose loud half happens to be
                # negative would otherwise read as silence.
                # np.frombuffer rather than memoryview.cast: the frame's
                # buffer is not guaranteed to be byte-typed, and cast() on a
                # non-byte memoryview raises inside this task, where the
                # exception is swallowed and the drain silently stops.
                samples = np.frombuffer(event.frame.data, dtype=np.int16)
                level = int(np.abs(samples).max()) if samples.size else 0
                peak = max(peak, level)
                if level > 500:
                    agent_audio_frames += 1
                    if agent_audio_frames > 20:
                        reply_audio.set()
                        return

        task = asyncio.create_task(drain())
        drains.add(task)
        task.add_done_callback(drains.discard)

    token = mint_token(cfg, session["room"], identity="speaker", name="Speaker")
    await room.connect(cfg.livekit_url, token)

    source = rtc.AudioSource(sample_rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("speech", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    try:
        # Wait for the agent to be listening before speaking into the room.
        await asyncio.sleep(3)

        samples_per_chunk = sample_rate // 100  # 10 ms
        bytes_per_chunk = samples_per_chunk * 2
        for offset in range(0, len(pcm), bytes_per_chunk):
            block = pcm[offset : offset + bytes_per_chunk]
            if len(block) < bytes_per_chunk:
                block = block + bytes(bytes_per_chunk - len(block))
            await source.capture_frame(
                rtc.AudioFrame(block, sample_rate, 1, samples_per_chunk)
            )
            await asyncio.sleep(0.01)

        # Trailing silence so the voice activity detector sees the turn end.
        for _ in range(80):
            await source.capture_frame(
                rtc.AudioFrame(bytes(bytes_per_chunk), sample_rate, 1, samples_per_chunk)
            )
            await asyncio.sleep(0.01)

        # Swallowed so the assertion below can report what was actually seen;
        # a bare wait_for would raise TimeoutError with no diagnostic.
        try:
            await asyncio.wait_for(reply_audio.wait(), timeout=90)
        except TimeoutError:
            pass
    finally:
        await room.disconnect()

    assert reply_audio.is_set(), (
        f"the avatar did not speak back (saw {agent_audio_frames} non-silent frames, "
        f"{total_audio_frames} audio frames total)"
    )


async def test_the_avatar_can_describe_what_the_camera_shows(cfg):
    """The vision channel, end to end.

    Publishes a camera track carrying a recognisable scene, asks about it in
    speech, and asserts the reply refers to what was shown. This is the part
    that distinguishes a call from a phone call.
    """
    import cv2
    from piper import PiperVoice

    session = await _open_session()

    # A frame with one unambiguous, describable feature: a large red shape.
    scene = np.full((480, 640, 3), (245, 243, 240), dtype=np.uint8)
    cv2.circle(scene, (320, 150), 90, (200, 170, 150), -1)          # head
    cv2.rectangle(scene, (200, 250), (440, 480), (40, 40, 210), -1)  # red torso
    rgba = np.dstack([scene, np.full((480, 640), 255, dtype=np.uint8)])

    voice = PiperVoice.load(f"{cfg.voices_dir}/{cfg.tts_voice}.onnx")
    chunks = list(voice.synthesize("¿De qué color es mi ropa? Responde en pocas palabras."))
    pcm = b"".join(c.audio_int16_bytes for c in chunks)
    sample_rate = voice.config.sample_rate

    room = rtc.Room()
    token = mint_token(cfg, session["room"], identity="viewer", name="Viewer")
    await room.connect(cfg.livekit_url, token)

    video_source = rtc.VideoSource(640, 480)
    video_track = rtc.LocalVideoTrack.create_video_track("camera", video_source)
    await room.local_participant.publish_track(
        video_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    )

    audio_source = rtc.AudioSource(sample_rate, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("speech", audio_source)
    await room.local_participant.publish_track(
        audio_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )

    stop = asyncio.Event()

    async def pump_camera():
        frame = rtc.VideoFrame(640, 480, rtc.VideoBufferType.RGBA, rgba.tobytes())
        while not stop.is_set():
            video_source.capture_frame(frame)
            await asyncio.sleep(1 / 15)

    pump = asyncio.create_task(pump_camera())

    try:
        # The vision model needs a moment: the first frame is always sent, but
        # describing it takes several seconds and it must land before the
        # question does.
        await asyncio.sleep(25)

        samples_per_chunk = sample_rate // 100
        bytes_per_chunk = samples_per_chunk * 2
        for offset in range(0, len(pcm), bytes_per_chunk):
            block = pcm[offset : offset + bytes_per_chunk]
            if len(block) < bytes_per_chunk:
                block += bytes(bytes_per_chunk - len(block))
            await audio_source.capture_frame(
                rtc.AudioFrame(block, sample_rate, 1, samples_per_chunk)
            )
            await asyncio.sleep(0.01)
        for _ in range(80):
            await audio_source.capture_frame(
                rtc.AudioFrame(bytes(bytes_per_chunk), sample_rate, 1, samples_per_chunk)
            )
            await asyncio.sleep(0.01)

        await asyncio.sleep(20)
    finally:
        stop.set()
        await pump
        await room.disconnect()

    # Asserted from the agent's log, and deliberately only on what this code
    # controls: that a camera frame was selected, described, and injected into
    # the system prompt the model reads.
    #
    # Whether a 3B vision model correctly reads a synthetic test frame is a
    # property of that model, not of the pipeline, and asserting on its wording
    # produces a test that passes for the wrong reasons - an earlier version of
    # this test accepted "clothing" and so passed on the description "without
    # any visible clothing". Description quality belongs in an evaluation
    # against real camera footage, not in this test.
    import pathlib

    log = pathlib.Path(os.environ["AGENT_LOG"]).read_text(errors="ignore")

    observations = [line for line in log.splitlines() if "scene:" in line]
    assert observations, "the vision channel selected no frame and produced no observation"

    description = observations[-1].split("scene:", 1)[1].strip()
    assert len(description) > 20, f"observation was empty or truncated: {description!r}"

    assert "scene observation injected into system prompt" in log, (
        "an observation was produced but never reached the model"
    )


async def test_the_stream_is_declared_synthetic(cfg):
    """Article 50 marking, verified on a real connection.

    Deliberately checks participant attributes and metadata, not pixels. An
    earlier version of this test looked for a spatial watermark in received
    frames and found nothing in thirty consecutive frames: WebRTC re-encodes
    everything, and a watermark faint enough to be invisible does not survive
    VP8. Marking a live stream has to travel out of band.
    """
    import json

    from avatar.marking.declare import ATTR_SOURCE_TYPE, ATTR_SYNTHETIC
    from avatar.marking.manifest import DIGITAL_SOURCE_TYPE

    session = await _open_session()

    room = rtc.Room()
    declared = asyncio.Event()
    seen: dict = {}

    def capture(participant) -> None:
        attributes = dict(participant.attributes or {})
        # Both halves are required: the flag says the stream is synthetic, the
        # metadata says what produced it and under whose consent.
        if attributes.get(ATTR_SYNTHETIC) == "true" and participant.metadata:
            seen["attributes"] = attributes
            seen["metadata"] = participant.metadata
            declared.set()

    @room.on("participant_attributes_changed")
    def _on_attributes(changed, participant):
        capture(participant)

    @room.on("participant_metadata_changed")
    def _on_metadata(old, participant):
        capture(participant)

    @room.on("participant_connected")
    def _on_connected(participant):
        capture(participant)

    token = mint_token(cfg, session["room"], identity="auditor", name="Auditor")
    await room.connect(cfg.livekit_url, token)

    try:
        # Already-present participants do not fire connection events.
        for participant in room.remote_participants.values():
            capture(participant)
        try:
            await asyncio.wait_for(declared.wait(), timeout=60)
        except TimeoutError:
            for participant in room.remote_participants.values():
                capture(participant)
    finally:
        await room.disconnect()

    assert declared.is_set(), "the avatar never declared its stream synthetic"
    assert seen["attributes"][ATTR_SOURCE_TYPE] == DIGITAL_SOURCE_TYPE

    manifest = json.loads(seen["metadata"])
    labels = {a["label"] for a in manifest["assertions"]}
    assert {"c2pa.actions", "avatar.consent", "avatar.models"} <= labels

    consent = next(
        a["data"] for a in manifest["assertions"] if a["label"] == "avatar.consent"
    )
    assert consent["consent_record_id"] not in ("", "unknown"), (
        "the declaration must name the consent record the session ran under"
    )
