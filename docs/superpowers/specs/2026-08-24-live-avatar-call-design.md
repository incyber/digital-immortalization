# Live Avatar Call — Design

Date: 2026-08-24
Status: Approved for planning
Supersedes: the `mvp/`, `web/`, and `vercel-app/` prototypes in this repository

## 1. Purpose

Replace the current text-and-voice prototypes with a live video call. A user
joins a room in the browser and sees and hears a rendered likeness of a
specific person, speaks to it naturally with interruption, and is seen by it —
the camera feed is part of the conversation, not decoration.

The product is a SaaS. Customers register, submit source material and consent
documentation for one person, and receive a callable likeness.

### Success criteria

A person opens a URL on a laptop or phone, grants microphone and camera, and
holds a two-minute spoken conversation where:

- Replies begin within 1.5 seconds of them finishing a sentence.
- Speaking over a reply stops it within 300 ms.
- The likeness's mouth tracks its own speech.
- At least once, unprompted, the likeness refers to something only visible
  through the camera.
- No session started without a stored, verified consent record.

## 2. Non-goals

Explicitly out of scope for this design:

- Photorealistic full-body avatars, gaze tracking, or emotional face transfer.
- Group calls. One human, one likeness, one room.
- Training or fine-tuning any model. Every model is used zero-shot.
- Mobile native apps. The browser is the client on every device.
- Real-time translation.

## 3. Verified findings

Checked directly against source, not secondary claims.

### Licences

| Component | Licence | Verified |
|---|---|---|
| Pipecat | BSD-2-Clause | GitHub API |
| LiveKit | Apache-2.0 | GitHub API |
| faster-whisper | MIT | GitHub API |
| Chatterbox | MIT | GitHub API |
| MuseTalk — code | MIT (Tencent Music) | LICENSE file read directly |
| MuseTalk — distributed weights | **CreativeML OpenRAIL-M** | HuggingFace API, `cardData.license` |
| ditto-talkinghead | Apache-2.0 | GitHub API |
| `PunithVT/ai-avatar-system` | MIT | GitHub API |

MuseTalk's README section 539 states its code is MIT with "no limitation for
both academic and commercial usage" and that its trained model is "available
for any purpose, even commercially."

**That reading was wrong, and this document asserted it.** The README describes
the code. The weights actually distributed on HuggingFace under
`TMElyralab/MuseTalk` carry `creativeml-openrail-m`, confirmed against the
HuggingFace API. OpenRAIL-M is usable commercially — its Attachment A contains
no anti-impersonation clause, and its restrictions are satisfiable by a
consented memorial product — but Section II obliges the licensee to carry the
use restrictions forward as an enforceable provision in any agreement governing
distribution. In practice: Attachment A must appear in the customer terms of
use with flow-down. That is lawyer time, not an engineering blocker, but
nothing in this design previously said so.

Two further corrections to the same table, both verified directly:

- `stabilityai/sd-vae-ft-mse`, loaded at inference, is MIT. Clean.
- `openai/whisper-tiny`, loaded at inference as the audio encoder, is
  Apache-2.0. Clean.
- The face-parsing weights used in preprocessing are trained on CelebAMask-HQ,
  whose terms restrict use to non-commercial research. They are cached per
  avatar and never touched at runtime, but they must still be replaced. The
  replacement is the shared face-geometry module in the next plan.

### Licence risk that remains open

MuseTalk's README section 542 explicitly defers the licences of its
dependencies: `whisper`, `ft-mse-vae`, `dwpose`, `S3FD`, `face-parsing`,
`LatentSync`. Each carries its own terms, and section 543 restricts its bundled
test data to non-commercial research.

This is a real, unresolved commercial risk and is tracked as task LIC-1 in the
implementation plan: audit every weight file the renderer loads at runtime and
record its licence. The design below isolates all such weights behind a single
component so that replacing any one of them is a contained change.

### A licence problem in what already shipped

`piper-tts`, which sub-project 1 ships and imports in-process, is
**GPL-3.0-or-later** — confirmed from the installed package metadata, which
also ships `COPYING` and a compiled `espeakbridge.so`. `src/avatar/services/
speech.py` described it as MIT. That comment was wrong.

Loading a GPL-3.0 library into the same process as proprietary application code
is the case the licence is designed to reach. Running it as a separate service
is aggregation rather than linking, which is the fix taken in the correctness
plan. The Piper *voice* files are licensed separately from the engine, and
`es_ES-davefx-medium.onnx.json` carries no licence field at all.

### The candidate baseline repository

`PunithVT/ai-avatar-system` — MIT, active, 417 stars, ~12,800 lines of Python
and TypeScript. Audited on 2026-08-24 at depth 1.

Worth taking:

- `backend/app/models.py` — User, Avatar, Session, Message, Conversation.
- `backend/app/middleware/` — rate limiting and security headers.
- `backend/app/services/storage.py`, `cache.py` — S3 and Redis wrappers.
- `backend/alembic/` — migration scaffolding.
- `infrastructure/*.tf`, `scripts/deploy-aws.sh` — GPU instance provisioning.
- `backend/app/services/animator.py` — the MuseTalk subprocess integration,
  adapted rather than copied.

Not taking:

- `backend/app/websocket.py` (936 lines) — a request/response WebSocket
  transport that returns rendered video in chunks. This is the architectural
  difference between this project and that one: a call is not a sequence of
  request/response turns. LiveKit and Pipecat replace it entirely.

A static scan for `shell=True`, `eval`, `exec`, `pickle.loads`, disabled TLS
verification, wildcard CORS, and hardcoded secrets returned nothing. `md5`
appears twice, both times as a cache key, not as a security primitive.

## 4. Architecture

```
Browser ──WebRTC──> LiveKit ──WebRTC──> Realtime agent
  mic + camera         SFU              (Pipecat pipeline)
  <──video + audio──               │
                                   ├── STT (faster-whisper)
                                   ├── Vision sampler ──> VLM
                                   ├── Persona + memory ──> LLM
                                   ├── TTS (Chatterbox)
                                   └── Renderer ──> video track

Gateway (FastAPI) ── auth, consent gate, session issuance, billing
Postgres ── users, avatars, consent records, sessions, transcripts
Redis ── session state, rate limits
S3 ── source material, voice references, rendered assets
```

Four deployable units: `web`, `gateway`, `realtime`, `renderer`. The first
three are CPU-bound; only `renderer` requires a GPU, which is what makes the
cost model tractable.

## 5. Components

### 5.1 `apps/web` — Next.js client

Owns the call UI: a full-bleed video surface for the likeness, a
picture-in-picture self-view, a mute control, and an end-call control. Uses the
LiveKit JS client to publish microphone and camera and subscribe to the agent's
tracks.

Two responsibilities beyond rendering:

- **Consent disclosure.** A persistent, non-dismissible banner naming the
  likeness as a synthetic recreation. Present before the call connects.
- **Camera frame gating.** Described in section 5.4.

### 5.2 `services/gateway` — FastAPI

Registration, login, avatar management, billing, and session issuance. Holds no
real-time state.

The one rule it enforces that nothing else can: `POST /sessions` returns a
LiveKit token only when the requested avatar has a consent record with
`status = 'verified'`. Any other state is a 403 with the reason. This is a hard
gate, not a warning.

### 5.3 `services/realtime` — Pipecat pipeline

Joins the room as a participant. One process per active session. The pipeline:

```
audio in ──> VAD ──> STT ──> crisis check ──> context assembly ──> LLM
                                    │                                │
                              (matched: fixed safety reply)          │
                                                                     v
                                                   sentence split ──> TTS ──> renderer ──> tracks
```

Two design points:

- **The crisis check runs before the LLM, not inside it.** A keyword match
  against a per-locale list short-circuits the pipeline, emits a fixed safety
  message naming a real crisis line, and writes to an audit log. The model is
  never given the chance to decide whether to break character. This mechanism
  is carried forward from the existing `mvp/` prototype, where it was the one
  part worth keeping.
- **Sentence-level streaming.** The LLM streams tokens; the pipeline splits on
  sentence boundaries and hands each completed sentence to TTS immediately, so
  audio and video begin before the model has finished its reply.

### 5.4 Vision channel

The requirement is that the likeness can observe clothing, gestures, and
setting. The constraint is that sending every frame to a vision model is
ruinous in both cost and latency.

Design — sample, don't stream:

1. The client captures at 15 fps for display but only considers a frame for
   upload every 4 seconds.
2. Before upload it computes mean absolute difference against the last uploaded
   frame, downscaled to 64×64 greyscale. Below a threshold, the frame is
   dropped — a still user costs nothing.
3. A candidate frame is downscaled to 512 px on its long edge, JPEG quality 70,
   and sent over the LiveKit data channel.
4. `services/realtime` calls the VLM asynchronously. The call never blocks the
   conversation pipeline.
5. The result is a description of at most 200 characters written to a
   `SceneState` object held in Redis for the session.
6. On every LLM turn, the current `SceneState` is injected into the system
   prompt as an observation.

Three frames are always sent regardless of the motion gate: at call start, at
the first user utterance, and on explicit user request ("can you see this?").

Ceiling: one VLM call per 4 seconds per session. A ten-minute call costs at
most 150 vision calls, and typically far fewer.

### 5.5 `services/renderer` — the swappable stage

This is the component the whole design bends around, because it is the only one
that needs a GPU and the only one with unresolved licence exposure.

```python
class RendererStage(Protocol):
    async def prepare(self, avatar_id: str) -> None:
        """Load weights and precompute per-avatar state. Called once per session."""

    async def idle(self) -> AsyncIterator[VideoFrame]:
        """Frames to publish when the likeness is not speaking."""

    async def render(self, audio: AudioChunk) -> AsyncIterator[VideoFrame]:
        """Frames lip-synced to one chunk of speech audio."""

    async def cancel(self) -> None:
        """Abandon in-flight rendering. Must return within 100 ms."""
```

Two implementations satisfying identical contract tests:

**`VisemeRenderer`** — the local development backend. The idle loop with a mouth
region composited per frame, selected from a small set of viseme shapes driven
by the short-term RMS envelope of the TTS audio. Envelope only — no phoneme
alignment, because Chatterbox does not emit timings and adding a forced aligner
would put a model dependency into the one component defined to have none. Pure
ffmpeg and numpy. No weights, no CUDA, no licence exposure. Runs anywhere.
Visibly synthetic, and deliberately so.

**`MuseTalkRenderer`** — the production backend. MuseTalk v1.5 on an NVIDIA GPU,
adapted from `animator.py` in the baseline repository. Selected by
`RENDERER_BACKEND=musetalk`.

The interface is what makes the local/cloud split honest rather than a
compromise: the call loop, timing behaviour, barge-in, and vision channel are
identical under both backends. Only pixels differ.

`cancel()` returning within 100 ms is a contract requirement, not a target. It
is what makes barge-in feel instant, and it is tested.

### 5.6 Source material

Both backends need per-avatar assets, produced once at avatar creation and
stored in S3. This is the customer's onboarding burden and is stated here
because both renderers depend on it.

| Asset | Requirement | Used by |
|---|---|---|
| Idle clip | 10–30 s, 25 fps, face visible throughout, neutral expression, mouth mostly closed | both |
| Voice reference | 10–30 s clean speech, single speaker, no music | Chatterbox (sub-project 3) |
| Viseme plate set | Derived, not uploaded: 6 mouth crops cut from the idle clip | `VisemeRenderer` |

The idle clip is the hard requirement. A single photograph is not sufficient:
MuseTalk composites a mouth region onto real frames and needs natural head
motion to look like anything but a mask, and the `VisemeRenderer` idle loop has
nothing to loop without it. Customers holding only photographs cannot be served
by this design, and the onboarding flow must say so before payment rather than
after upload.

Both are validated at upload — face detected in every sampled frame, single
speaker, minimum duration — and rejected with a specific reason if not. A
rejection after payment is a refund; a rejection before it is a filter.

## 6. Execution matrix

| Stage | Local (MacBook) | Cloud |
|---|---|---|
| LiveKit | container, native arm64 | same, clustered |
| Pipecat | full speed | same |
| STT | faster-whisper `base`, int8, CPU | `large-v3`, GPU |
| LLM | hosted API | same |
| VLM | hosted API | same |
| TTS | Chatterbox, MPS or CPU | Chatterbox, GPU |
| Renderer | `VisemeRenderer` | `MuseTalkRenderer` |

MuseTalk was excluded locally on the strength of its README's "five minutes per
eight-second clip." **That figure is misleading and this document should not
have relied on it.** It was measured on an RTX 3050 Ti Laptop GPU with **4 GB**
of VRAM, running an 850M-parameter UNet alongside a VAE and Whisper. It
measures memory starvation, not compute.

Benchmarked directly on this project's target Apple silicon (M3 Max, 36 GB
unified, MPS, fp16), the same networks sustain:

| Configuration | Sustained |
|---|---|
| 256 px, batch 8, faithful SD VAE | 10.8 fps |
| 256 px, batch 8, TAESD decoder | 30.4 fps |
| 192 px, batch 8, TAESD decoder | 51.2 fps |

The decoder, not the UNet, is the bottleneck on MPS; TAESD is MIT and decodes
the same latent space. The genuine Apple-silicon blocker is `mmpose`/DWPose in
*preprocessing*, which is a one-time offline step per avatar, not a runtime
cost.

So the renderer interface in section 5.5 is still the right design — it is what
lets the local and cloud backends differ at all — but its justification is now
that local runs a cheaper decoder at a lower resolution, not that the model
cannot run here.

No component requires installing a desktop application. Python dependencies
resolve through `uv`, a single static binary; JavaScript through `npm`. Local
orchestration is `docker compose`; a `make dev` target runs the same services
as plain processes for anyone who does not want a container runtime.

## 7. Data model

New tables beyond those harvested from the baseline repository:

```
consent_records
  id, avatar_id, rights_holder_name, relationship, jurisdiction,
  evidence_s3_key, signed_at, verified_at, verified_by, status, notes
  status ∈ {pending, verified, rejected, revoked}

scene_observations
  id, session_id, observed_at, description, model, tokens_used

safety_events
  id, session_id, matched_term, locale, transcript_excerpt, occurred_at
```

`consent_records.status` is the gate in section 5.2. `revoked` immediately
invalidates future sessions; a rights-holder withdrawing consent is a supported
operation, not an edge case.

Transcripts are retained per the customer's configured retention window, default
30 days, and are deletable on request. Voice references and source video are
retained for the life of the avatar and destroyed with it.

## 8. Legal position

California AB 1836 extends post-mortem personality rights to cover digital
replicas. Tennessee's ELVIS Act covers voice and likeness. There is still no US
federal statute; state exposure varies.

**This section was written as though US state law were the whole picture. It is
not, and one obligation is already live.**

Verified independently:

- **EU AI Act Article 50 has applied since 2 August 2026.** Providers of AI
  systems generating synthetic audio, image or video must mark outputs in a
  **machine-readable** form so they are detectable as AI-generated. The
  Commission adopted implementing guidelines on 20 July 2026. Penalties reach
  €15M or 3% of worldwide annual turnover. This product generates synthetic
  video of a real person and is squarely in scope. The persistent banner in
  section 5.1 satisfies the human-facing half of Article 50 only; machine
  -readable marking is a separate obligation and is addressed in section 8.1.

Reported but **not independently verified** — each needs counsel before it is
relied on:

- New York post-mortem digital-replica right, reported signed 11 December 2025.
- China's deep synthesis provisions: consent plus a visible label plus an
  invisible watermark.
- NO FAKES Act, reported out of Senate Judiciary in June 2026, not law. It
  vests post-mortem rights in "executors, heirs, assignees, or devisees" —
  the same channel the consent gate already models.
- A Danish likeness right, reportedly in force around July 2026.

The encouraging reading, if these hold: every regime treats family or estate
authorisation as the intended consent path, so the consent gate is aimed
correctly. The gap is output marking, not permission.

### 8.1 Synthetic-content marking

Every published video frame carries a machine-readable mark identifying it as
synthetic, and each session records a manifest naming the models that produced
it. Marking applies to idle frames as well as spoken ones: a frame is synthetic
whether or not the likeness is talking.

This is an engineering answer to a marking obligation, not a compliance
certification. A C2PA-conformant implementation and counsel review are both
still required. The manifest is shaped to carry C2PA assertion fields so that
path stays open.

Concrete consequences for this design, all reflected above:

- No session without a verified consent record naming a rights-holder and their
  relationship to the deceased.
- Evidence documents are stored and auditable, not merely attested by checkbox.
- Consent is revocable, and revocation takes effect immediately.
- Synthetic nature is disclosed persistently in the call UI, not once at signup.

This is a shipping requirement in version one. Retrofitting a consent gate into
a system with live customers is far more expensive than building it now.

## 9. Error handling

| Failure | Behaviour |
|---|---|
| STT returns empty | No turn is taken. The likeness stays idle. |
| LLM times out (>8 s) | Spoken filler, one retry, then a graceful apology. |
| TTS fails | Reply is shown as a caption. The call continues. |
| Renderer fails or stalls | Fall back to `idle()` frames with audio intact. A call without lip-sync is degraded; a call without audio is broken. |
| VLM fails | `SceneState` retains its previous value. Never blocks a turn. |
| GPU pool exhausted | Session issuance returns 503 with a retry estimate. Never silently downgrades a paying customer to `VisemeRenderer`. |
| Consent revoked mid-call | Call ends with an explanation. |

The ordering principle: audio continuity outranks video fidelity, and both
outrank feature completeness.

## 10. Testing

- **Contract tests** — one suite run against both `RendererStage`
  implementations. Frame cadence, output dimensions, and the 100 ms `cancel()`
  bound are asserted identically for each.
- **Pipeline tests** — the Pipecat graph with recorded STT, a scripted LLM, and
  silent TTS. Asserts turn boundaries, that barge-in cancels downstream stages,
  and that the crisis check short-circuits before the LLM is called.
- **Consent gate tests** — every `consent_records.status` value against
  `POST /sessions`. Only `verified` yields a token.
- **Latency budget test** — measures end of user speech to first published
  video frame against the 1.5 s target, and fails CI on regression.
- **End-to-end** — a headless LiveKit client joins a room, publishes fixture
  audio and a fixture camera frame, and asserts that audio and video tracks are
  received and that the reply references the fixture image.

## 11. Latency budget

Target: 1.5 s from end of user speech to first video frame, on cloud hardware.

| Stage | Budget |
|---|---|
| VAD endpoint detection | 200 ms |
| STT final transcript | 150 ms |
| LLM first sentence | 600 ms |
| TTS first audio chunk | 250 ms |
| Renderer first frame | 200 ms |
| Transport | 100 ms |

The vision channel is deliberately absent: it is asynchronous and outside the
turn path by design.

## 12. Build order

This design covers more than one implementation plan. Sub-projects, in
dependency order:

1. **Call spine and vision** — sections 5.1, 5.2, 5.3, 5.4, `VisemeRenderer`,
   the consent gate, and the tests in section 10. Runs entirely on a MacBook.
   This is the sub-project the first implementation plan covers.
2. **Production renderer** — `MuseTalkRenderer`, GPU provisioning, the LIC-1
   licence audit, session scheduling against a GPU pool.
3. **Voice identity** — Chatterbox cloning from a reference clip, per-avatar
   voice storage.
4. **Persona and memory** — carry `prompt-v2.txt` forward, replace flat-file
   recall with retrieval over the transcript corpus.
5. **Platform** — billing, avatar onboarding, consent document review workflow.

Sub-project 1 is deliberately the largest risk: it proves the sub-1.5 s
conversational loop with vision, which is the part that either works or kills
the product. Everything after it is substitution behind an interface already
defined and tested.

## 13. Open risks

- **MuseTalk under concurrent load.** The 30 fps figure is a single stream on a
  V100. Cost per concurrent session is unmeasured and is the variable that
  decides the business model. Measured in sub-project 2, not assumed.
- **Dependency weights (LIC-1).** Section 3 states the exposure. Isolated
  behind `RendererStage` so that a replacement is contained.
- **Chatterbox on MPS.** Unverified. If it does not run, local TTS falls back
  to CPU, which is slower but sufficient for a development loop.
- **Consent verification is manual in version one.** A human reviews submitted
  documents. This does not scale past early customers and is revisited in
  sub-project 5.
