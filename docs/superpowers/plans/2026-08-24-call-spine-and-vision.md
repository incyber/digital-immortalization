# Call Spine and Vision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser call where a person speaks to a rendered likeness, is interrupted-capable, and is seen through their camera — running end to end on a MacBook with no GPU.

**Architecture:** LiveKit carries WebRTC in both directions. One Pipecat pipeline per session joins the room as a participant: VAD → STT → crisis check → context assembly → LLM → sentence split → TTS → renderer → published audio and video tracks. Camera frames arrive on a sampled side-channel that never blocks a turn. The renderer sits behind a four-method interface with a CPU implementation now and a GPU implementation later.

**Tech Stack:** Python 3.12, Pipecat 1.7.0, LiveKit (server as container, `livekit-api` for tokens), mlx-whisper STT, Ollama for LLM and VLM, Piper TTS, FastAPI + SQLAlchemy + SQLite/Postgres, Next.js 15 + `@livekit/components-react`, numpy + OpenCV + ffmpeg for the renderer.

**Spec:** `docs/superpowers/specs/2026-08-24-live-avatar-call-design.md`

## Global Constraints

- Python `>=3.11,<3.13`. Pipecat 1.7.0 requires `>=3.11`; several ML wheels lag on 3.13.
- No system-level installs. `uv` and any server binaries live in `.tools/`, which is gitignored. Containers for LiveKit, Postgres, Redis.
- `RendererStage.cancel()` must return within 100 ms. Asserted in contract tests, not aspirational.
- Latency budget: end of user speech to first published video frame ≤ 1500 ms on cloud hardware. Local is allowed to exceed it; the test is skipped unless `LATENCY_ASSERT=1`.
- Vision: at most one VLM call per 4 s per session. Never on the turn path.
- No session token issued unless `consent_records.status == 'verified'`. Enforced in one place.
- Sub-project 1 ships `PiperTTSService`, not Chatterbox. Voice cloning is sub-project 3 in the spec's build order; both satisfy Pipecat's `TTSService` contract, so the swap is contained.
- Local STT is `mlx-whisper` (Metal). Linux/cloud is `faster-whisper`. Selected by `STT_BACKEND`.

### Verified API signatures

These were introspected from the installed packages on 2026-08-24. Do not deviate.

```python
LiveKitTransport(url: str, token: str, room_name: str, params: LiveKitParams | None = None,
                 input_name: str | None = None, output_name: str | None = None)

# LiveKitParams fields used here (pydantic model):
#   audio_in_enabled, audio_out_enabled, audio_out_sample_rate,
#   video_in_enabled, video_out_enabled, video_out_is_live,
#   video_out_width, video_out_height, video_out_framerate, video_out_color_format

TTSService.run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]
STTService.run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]
FrameProcessor.process_frame(self, frame: Frame, direction: FrameDirection)
FrameProcessor.push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM)

OutputImageRawFrame(image: bytes, size: tuple[int, int], format: str | None)
InputImageRawFrame(image: bytes, size: tuple[int, int], format: str | None)
TTSAudioRawFrame(audio: bytes, sample_rate: int, num_channels: int, context_id: str | None = None)

LLMContext(messages: list[...] | None = None, tools=NOT_GIVEN, tool_choice=NOT_GIVEN)
LLMContext.add_message(...), .get_messages(), .set_messages(...)
```

Import paths that moved in 1.x and are easy to get wrong:

```python
from pipecat.processors.aggregators.llm_context import LLMContext   # NOT openai_llm_context
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
```

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/avatar/config.py` | One `Settings` object, env-driven. Nothing else reads `os.environ`. |
| `src/avatar/safety/crisis.py` | Keyword guardrail. Pure, synchronous, no I/O. |
| `src/avatar/safety/keywords.py` | Per-locale term lists. Data, not logic. |
| `src/avatar/renderer/base.py` | `RendererStage` protocol, `VideoFrame`, `AudioChunk`. |
| `src/avatar/renderer/viseme.py` | CPU renderer. Envelope-driven mouth compositing. |
| `src/avatar/renderer/plates.py` | Builds viseme plates and idle loop from a source clip. |
| `src/avatar/renderer/processor.py` | Pipecat processor wrapping any `RendererStage`. |
| `src/avatar/services/tts_piper.py` | `PiperTTSService(TTSService)`. |
| `src/avatar/services/stt.py` | Backend selection for STT. |
| `src/avatar/services/llm.py` | Ollama via OpenAI-compatible base URL. |
| `src/avatar/vision/sampler.py` | Motion-gated frame selection from the camera track. |
| `src/avatar/vision/describe.py` | VLM call. Async, bounded, failure-tolerant. |
| `src/avatar/vision/state.py` | `SceneState` — the only thing the pipeline reads. |
| `src/avatar/persona.py` | System prompt assembly from profile + scene + memory. |
| `src/avatar/gateway/app.py` | FastAPI app factory. |
| `src/avatar/gateway/models.py` | SQLAlchemy tables. |
| `src/avatar/gateway/consent.py` | The gate. One function, one rule. |
| `src/avatar/gateway/sessions.py` | LiveKit token issuance. |
| `src/avatar/realtime/agent.py` | Pipeline assembly and room join. |
| `apps/web/` | Next.js client. |
| `infra/docker-compose.yml` | LiveKit, Postgres, Redis. |
| `Makefile` | `make dev`, `make test`, `make infra-up`. |

---

## Task 1: Configuration and toolchain

**Files:**
- Create: `src/avatar/config.py`, `Makefile`, `infra/docker-compose.yml`, `.env.example`
- Modify: `.gitignore`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` (pydantic-settings `BaseSettings`) with fields `livekit_url`, `livekit_api_key`, `livekit_api_secret`, `database_url`, `redis_url`, `llm_base_url`, `llm_model`, `vlm_model`, `stt_backend`, `renderer_backend`, `vision_interval_s`, `vision_motion_threshold`. Accessor `get_settings()` returns a cached singleton.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from avatar.config import Settings

def test_defaults_are_local_first():
    s = Settings(_env_file=None)
    assert s.renderer_backend == "viseme"
    assert s.stt_backend == "mlx"
    assert s.vision_interval_s == 4.0
    assert s.llm_base_url.endswith("/v1")

def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RENDERER_BACKEND", "musetalk")
    assert Settings(_env_file=None).renderer_backend == "musetalk"
```

- [ ] **Step 2: Run it, confirm ImportError**

`.tools/uv run pytest tests/test_config.py -v` → FAIL, no module `avatar.config`.

- [ ] **Step 3: Implement `Settings`**

Use `pydantic_settings.BaseSettings` with `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`. Defaults: `llm_base_url="http://localhost:11434/v1"`, `llm_model="llama3.2:3b"`, `vlm_model="qwen2.5vl:3b"`, `database_url="sqlite+aiosqlite:///./avatar.db"`, `livekit_url="ws://localhost:7880"`.

- [ ] **Step 4: Tests pass**

- [ ] **Step 5: Write `infra/docker-compose.yml`**

LiveKit `livekit/livekit-server:latest` with `--dev` (binds 7880, generates the `devkey`/`secret` pair), Postgres 16, Redis 7. `platform: linux/arm64` is unnecessary — all three publish multi-arch images.

- [ ] **Step 6: Write `Makefile`** with `infra-up`, `infra-down`, `dev`, `test`, `agent`, `web`.

- [ ] **Step 7: Commit**

---

## Task 2: Crisis guardrail

The one component carried forward from the old prototypes, because the mechanism was right: a deterministic check that runs before the model, so breaking character is not a decision the model gets to make.

**Files:**
- Create: `src/avatar/safety/keywords.py`, `src/avatar/safety/crisis.py`
- Test: `tests/test_crisis.py`

**Interfaces:**
- Produces: `check(text: str, locale: str = "en") -> CrisisMatch | None`, where `CrisisMatch` is a frozen dataclass with `term: str`, `locale: str`. And `safety_reply(locale: str, line_name: str, line_number: str) -> str`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_crisis.py
from avatar.safety.crisis import check

def test_matches_direct_statement_en():
    assert check("i want to kill myself").term == "kill myself"

def test_matches_spanish():
    assert check("quiero matarme", locale="es").term == "matarme"

def test_word_boundaries_not_substrings():
    # "asesinato" contains no standalone crisis term; must not match on fragments
    assert check("hablamos de un asesinato en la novela", locale="es") is None

def test_ordinary_speech_does_not_match():
    assert check("i could kill for a coffee") is None

def test_case_and_accent_insensitive():
    assert check("QUIERO MATARME", locale="es") is not None
    assert check("quiero suicidarme", locale="es") is not None
```

- [ ] **Step 2: Run, confirm failure**

- [ ] **Step 3: Implement**

Normalise with `unicodedata.normalize("NFKD", ...)` and strip combining marks, lowercase, then match each term with `re.search(rf"\b{re.escape(term)}\b", ...)`. Word boundaries are what makes `test_word_boundaries_not_substrings` pass; substring matching would fail it.

Note the deliberate false-negative in `test_ordinary_speech_does_not_match`: the phrase "kill for a coffee" contains "kill" but the term list holds "kill myself", not "kill". The spec is explicit that this is a mechanism demonstration, not a classifier.

- [ ] **Step 4: Tests pass. Step 5: Commit.**

---

## Task 3: Renderer interface and contract tests

**Files:**
- Create: `src/avatar/renderer/base.py`
- Test: `tests/renderer/test_contract.py`, `tests/renderer/conftest.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class VideoFrame:
    data: bytes            # raw RGB24
    width: int
    height: int

@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes             # signed 16-bit LE mono
    sample_rate: int

class RendererStage(Protocol):
    async def prepare(self, avatar_id: str) -> None: ...
    async def idle(self) -> AsyncIterator[VideoFrame]: ...
    async def render(self, audio: AudioChunk) -> AsyncIterator[VideoFrame]: ...
    async def cancel(self) -> None: ...
    @property
    def fps(self) -> int: ...
    @property
    def size(self) -> tuple[int, int]: ...
```

- [ ] **Step 1: Write the contract test suite, parameterised over implementations**

```python
# tests/renderer/conftest.py
import pytest
from avatar.renderer.viseme import VisemeRenderer

def _impls():
    yield pytest.param(lambda assets: VisemeRenderer(assets), id="viseme")
    # MuseTalkRenderer joins this list in sub-project 2 and must pass unchanged.

@pytest.fixture(params=list(_impls()))
def renderer_factory(request):
    return request.param
```

```python
# tests/renderer/test_contract.py
import time, pytest
from avatar.renderer.base import AudioChunk

async def test_render_emits_frames_at_declared_size(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets); await r.prepare("test")
    frames = [f async for f in r.render(AudioChunk(pcm=b"\x00\x00" * 16000, sample_rate=16000))]
    assert frames, "one second of audio must produce frames"
    assert all((f.width, f.height) == r.size for f in frames)
    assert all(len(f.data) == f.width * f.height * 3 for f in frames)

async def test_frame_count_tracks_audio_duration(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets); await r.prepare("test")
    one_sec = AudioChunk(pcm=b"\x00\x00" * 16000, sample_rate=16000)
    n = len([f async for f in r.render(one_sec)])
    assert abs(n - r.fps) <= 1

async def test_idle_is_endless(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets); await r.prepare("test")
    it = r.idle()
    seen = [await anext(it) for _ in range(50)]
    assert len(seen) == 50

async def test_cancel_returns_within_100ms(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets); await r.prepare("test")
    agen = r.render(AudioChunk(pcm=b"\x00\x00" * 160000, sample_rate=16000))
    await anext(agen)
    t0 = time.perf_counter()
    await r.cancel()
    assert (time.perf_counter() - t0) < 0.100

async def test_render_after_cancel_still_works(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets); await r.prepare("test")
    await r.cancel()
    frames = [f async for f in r.render(AudioChunk(pcm=b"\x00\x00" * 16000, sample_rate=16000))]
    assert frames
```

- [ ] **Step 2: Run, confirm collection error (no implementation yet)**
- [ ] **Step 3: Write `base.py` only — protocol and dataclasses. Step 4: Commit.**

---

## Task 4: Avatar asset builder

Produces what both renderers consume. Without this the viseme renderer has nothing to composite.

**Files:**
- Create: `src/avatar/renderer/plates.py`
- Test: `tests/renderer/test_plates.py`

**Interfaces:**
- Produces: `build_assets(source_video: Path, out_dir: Path, size=(512,512), fps=25) -> AvatarAssets`, and `AvatarAssets` with `idle_frames: list[np.ndarray]`, `mouth_box: tuple[int,int,int,int]`, `plates: list[np.ndarray]` (6 entries, closed → wide open), `fps: int`, `size: tuple[int,int]`. `AvatarAssets.load(dir)` / `.save(dir)` round-trip through `.npz` plus a JSON sidecar.
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write failing tests** asserting `len(plates) == 6`, `mouth_box` inside frame bounds, `save`/`load` round-trip equality, and that a clip with no detectable face raises `NoFaceDetected`.
- [ ] **Step 2: Run, confirm failure**
- [ ] **Step 3: Implement.** ffmpeg decodes to raw frames at the target fps and size. OpenCV's bundled `haarcascade_frontalface_default.xml` locates the face — chosen over a learned detector precisely because it ships inside the pinned `opencv-python-headless` wheel and adds no weight file with its own licence. The mouth box is the lower third of the face box, widened 15%. Plates are cut from the frames whose mouth-region vertical gradient energy spans the observed range, ranked and sampled at six points.
- [ ] **Step 4: Tests pass. Step 5: Commit.**

---

## Task 5: Viseme renderer

**Files:**
- Create: `src/avatar/renderer/viseme.py`
- Test: covered by Task 3's contract suite plus `tests/renderer/test_viseme.py`

**Interfaces:**
- Consumes: `RendererStage`, `VideoFrame`, `AudioChunk` (Task 3); `AvatarAssets` (Task 4).
- Produces: `VisemeRenderer(assets: AvatarAssets | Path)`.

- [ ] **Step 1: Write behaviour tests beyond the contract**

```python
# tests/renderer/test_viseme.py
import numpy as np
from avatar.renderer.base import AudioChunk

def _tone(sr=16000, secs=1.0, amp=20000):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    return (np.sin(2 * np.pi * 220 * t) * amp).astype(np.int16).tobytes()

async def test_loud_audio_opens_mouth_wider_than_silence(tmp_assets):
    r = VisemeRenderer(tmp_assets); await r.prepare("t")
    loud = [f async for f in r.render(AudioChunk(_tone(), 16000))]
    quiet = [f async for f in r.render(AudioChunk(b"\x00\x00" * 16000, 16000))]
    assert r.last_plate_indices(loud) != [0] * len(loud)
    assert r.last_plate_indices(quiet) == [0] * len(quiet)
```

- [ ] **Step 2: Run, confirm failure**
- [ ] **Step 3: Implement.** Split the PCM into `1/fps`-second windows, take RMS per window, normalise against a rolling ceiling, quantise to a plate index 0–5, alpha-composite that plate into `mouth_box` over the idle frame for that position in the loop. `cancel()` sets an `asyncio.Event` the generator checks each frame — that is what keeps it inside 100 ms, since no single frame takes anywhere near that long.
- [ ] **Step 4: Whole contract suite passes. Step 5: Commit.**

---

## Task 6: Renderer processor

Bridges any `RendererStage` into the Pipecat graph.

**Files:**
- Create: `src/avatar/renderer/processor.py`
- Test: `tests/renderer/test_processor.py`

**Interfaces:**
- Consumes: `RendererStage`.
- Produces: `RendererProcessor(stage: RendererStage)`, a `FrameProcessor`.

Behaviour: on `TTSAudioRawFrame`, pass the audio through unchanged and additionally push `OutputImageRawFrame(image=f.data, size=(f.width,f.height), format="RGB")` for each rendered frame. On `BotStoppedSpeakingFrame`, resume publishing `idle()`. On `StartInterruptionFrame`, call `stage.cancel()` and resume idle.

- [ ] **Step 1: Write failing test** driving a fake stage, asserting that an interruption frame triggers `cancel()` and that audio frames are never swallowed.
- [ ] **Step 2: Run, confirm failure. Step 3: Implement. Step 4: Pass. Step 5: Commit.**

---

## Task 7: Piper TTS service

**Files:**
- Create: `src/avatar/services/tts_piper.py`
- Test: `tests/test_tts_piper.py`

**Interfaces:**
- Produces: `PiperTTSService(voice_path: Path, sample_rate: int = 22050)` subclassing `pipecat.services.tts_service.TTSService`, implementing `run_tts(self, text, context_id)` as an async generator yielding `TTSAudioRawFrame`.

- [ ] **Step 1: Write failing test** asserting `run_tts("hello", "ctx")` yields at least one `TTSAudioRawFrame` whose `sample_rate` matches and whose `audio` length is a multiple of 2 (16-bit mono).
- [ ] **Step 2: Run, confirm failure. Step 3: Implement** by running synthesis in a thread via `asyncio.to_thread` so the event loop is never blocked. **Step 4: Pass. Step 5: Commit.**

---

## Task 8: Vision channel

**Files:**
- Create: `src/avatar/vision/state.py`, `src/avatar/vision/sampler.py`, `src/avatar/vision/describe.py`
- Test: `tests/vision/test_sampler.py`, `tests/vision/test_state.py`

**Interfaces:**
- Produces:
  - `SceneState` — `.description: str`, `.updated_at: float`, `.update(text)`, `.as_prompt_fragment() -> str` returning `""` when empty and otherwise a single observation line capped at 200 characters.
  - `MotionGate(interval_s: float, threshold: float)` — `.should_send(frame_rgb: np.ndarray, now: float) -> bool`.
  - `describe_frame(jpeg: bytes, model: str, base_url: str) -> str`.
  - `VisionSampler(FrameProcessor)` wiring the three together off `InputImageRawFrame`.

- [ ] **Step 1: Write failing tests**

```python
# tests/vision/test_sampler.py
def test_first_frame_always_sends(gate, frame): assert gate.should_send(frame, now=0.0)

def test_identical_frame_within_interval_is_dropped(gate, frame):
    gate.should_send(frame, 0.0)
    assert not gate.should_send(frame, 1.0)

def test_identical_frame_after_interval_is_still_dropped(gate, frame):
    gate.should_send(frame, 0.0)
    assert not gate.should_send(frame, 10.0)   # interval permits, motion gate refuses

def test_changed_frame_after_interval_sends(gate, frame, other_frame):
    gate.should_send(frame, 0.0)
    assert gate.should_send(other_frame, 10.0)

def test_changed_frame_within_interval_is_dropped(gate, frame, other_frame):
    gate.should_send(frame, 0.0)
    assert not gate.should_send(other_frame, 1.0)   # rate limit outranks motion
```

That last pair encodes the rule precisely: **both** conditions must hold. The interval is a ceiling on cost; the motion threshold suppresses redundant spend below it.

- [ ] **Step 2: Run, confirm failure. Step 3: Implement.** Downscale to 64×64 greyscale for the diff. `describe_frame` posts to Ollama's `/api/chat` with the image base64-encoded, wrapped in `asyncio.wait_for(..., timeout=8)`; on any exception it returns `""` and the caller leaves `SceneState` untouched — the spec requires vision failure never to block a turn. **Step 4: Pass. Step 5: Commit.**

---

## Task 9: Persona assembly

**Files:**
- Create: `src/avatar/persona.py`, `src/avatar/profiles/colon.json`
- Test: `tests/test_persona.py`

**Interfaces:**
- Consumes: `SceneState` (Task 8).
- Produces: `build_system_prompt(profile: dict, scene: SceneState, recent: list[dict]) -> str`.

- [ ] **Step 1: Write failing tests** asserting the rendered prompt contains no unfilled `{{placeholder}}`, that an empty `SceneState` contributes no observation line, and that a populated one contributes exactly one.
- [ ] **Step 2: Run, confirm failure. Step 3: Implement**, carrying forward the persona structure from `prompt-v2.txt`. **Step 4: Pass. Step 5: Commit.**

---

## Task 10: Gateway, data model, and the consent gate

**Files:**
- Create: `src/avatar/gateway/models.py`, `consent.py`, `sessions.py`, `app.py`
- Test: `tests/gateway/test_consent.py`, `tests/gateway/test_sessions.py`

**Interfaces:**
- Produces: tables `users`, `avatars`, `consent_records`, `sessions`, `scene_observations`, `safety_events` per spec section 7. `assert_consented(db, avatar_id) -> ConsentRecord` raising `ConsentError` with a reason. `POST /api/sessions {avatar_id}` returning `{url, token, room}`.

- [ ] **Step 1: Write the failing gate tests — every status value**

```python
# tests/gateway/test_consent.py
import pytest
from avatar.gateway.consent import assert_consented, ConsentError

@pytest.mark.parametrize("status", ["pending", "rejected", "revoked"])
async def test_non_verified_status_is_refused(db, avatar, status):
    await set_status(db, avatar, status)
    with pytest.raises(ConsentError) as e:
        await assert_consented(db, avatar.id)
    assert status in str(e.value)

async def test_verified_status_passes(db, avatar):
    await set_status(db, avatar, "verified")
    assert (await assert_consented(db, avatar.id)).status == "verified"

async def test_missing_record_is_refused(db, avatar_without_consent):
    with pytest.raises(ConsentError):
        await assert_consented(db, avatar_without_consent.id)
```

```python
# tests/gateway/test_sessions.py
async def test_session_endpoint_403s_without_consent(client, avatar):
    assert (await client.post("/api/sessions", json={"avatar_id": str(avatar.id)})).status_code == 403

async def test_session_endpoint_returns_joinable_token(client, verified_avatar):
    r = await client.post("/api/sessions", json={"avatar_id": str(verified_avatar.id)})
    assert r.status_code == 200
    body = r.json()
    assert body["url"].startswith("ws")
    assert len(body["token"].split(".")) == 3     # a real JWT, not a stub
```

- [ ] **Step 2: Run, confirm failure. Step 3: Implement.** Token minting uses `livekit.api.AccessToken` with a grant scoped to the single room. **Step 4: Pass. Step 5: Commit.**

---

## Task 11: Realtime agent

**Files:**
- Create: `src/avatar/realtime/agent.py`, `src/avatar/services/stt.py`, `src/avatar/services/llm.py`
- Test: `tests/realtime/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `build_pipeline(cfg, assets, profile) -> tuple[Pipeline, SceneState]` and `run_agent(room, token) -> None`.

Pipeline order — the crisis processor sits before the context aggregator, which is what makes the guardrail structural rather than advisory:

```python
Pipeline([
    transport.input(),
    VisionSampler(scene, cfg),
    stt,
    CrisisProcessor(profile),
    context_aggregator.user(),
    llm,
    tts,
    RendererProcessor(stage),
    transport.output(),
    context_aggregator.assistant(),
])
```

- [ ] **Step 1: Write failing tests** with scripted STT and LLM stubs: a normal utterance reaches the LLM; a crisis utterance does not and emits the fixed reply plus a `safety_events` row; an interruption mid-reply cancels the renderer.
- [ ] **Step 2: Run, confirm failure. Step 3: Implement. Step 4: Pass. Step 5: Commit.**

---

## Task 12: Web client

**Files:**
- Create: `apps/web/` — `app/page.tsx`, `app/call/[avatarId]/page.tsx`, `components/CallStage.tsx`, `components/Disclosure.tsx`, `lib/frames.ts`

**Interfaces:**
- Consumes: `POST /api/sessions`.

- [ ] **Step 1: Scaffold Next.js 15 with TypeScript and Tailwind**
- [ ] **Step 2: Build the call surface** — `LiveKitRoom` with the agent's video track full-bleed, self-view picture-in-picture, mute and end controls.
- [ ] **Step 3: Build the disclosure banner** — persistent, non-dismissible, rendered before connection.
- [ ] **Step 4: Publish the camera track.** Frame gating lives server-side in `VisionSampler`; the client simply publishes at low resolution (320×240, 15 fps) since nothing else consumes it.
- [ ] **Step 5: Manual verification** — load the page, confirm both tracks, confirm the banner is present before connect. **Step 6: Commit.**

---

## Task 13: End-to-end and latency tests

**Files:**
- Create: `tests/e2e/test_call.py`, `tests/e2e/conftest.py`

- [ ] **Step 1: Write the e2e test** — a headless LiveKit participant joins, publishes fixture speech and a fixture camera frame, and asserts that audio and video tracks are received and that the transcript references the fixture image.
- [ ] **Step 2: Write the latency test** — measure end-of-speech to first video frame, report always, assert only when `LATENCY_ASSERT=1`.
- [ ] **Step 3: Run against live infra. Step 4: Commit.**

---

## Self-Review

**Spec coverage:** §5.1 → Task 12. §5.2 → Task 10. §5.3 → Tasks 2, 11. §5.4 → Task 8. §5.5 → Tasks 3, 5, 6. §5.6 → Task 4. §6 → Task 1. §7 → Task 10. §8 → Tasks 10, 12. §9 → Tasks 6, 8, 11. §10 → Tasks 3, 11, 13. §11 → Task 13.

Two spec items deliberately deferred with justification recorded in Global Constraints: Chatterbox TTS (spec §6 lists it in the end-state matrix; spec §12 assigns voice cloning to sub-project 3) and `MuseTalkRenderer` (spec §12, sub-project 2). The contract suite in Task 3 is written so both arrive without changes to it.

**Placeholder scan:** none.

**Type consistency:** `AudioChunk(pcm, sample_rate)`, `VideoFrame(data, width, height)`, and `AvatarAssets` field names are used identically in Tasks 3, 4, 5, and 6. `SceneState.as_prompt_fragment()` is named the same in Tasks 8, 9, and 11. `assert_consented` is named the same in Task 10's two test files.
