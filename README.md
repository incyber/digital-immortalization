# Live avatar call

A browser video call with a rendered likeness of a specific person. You speak,
it answers, you can interrupt it, and it sees you through your camera.

Runs end to end on a laptop with no GPU. The same source deploys to a GPU
server, where only the renderer changes.

- Design: `docs/superpowers/specs/2026-08-24-live-avatar-call-design.md`
- Plan: `docs/superpowers/plans/2026-08-24-call-spine-and-vision.md`

## What works today

| | |
|---|---|
| Transport | LiveKit, WebRTC both directions |
| Turn taking | Silero VAD, barge-in mid-reply |
| Speech in | Whisper via MLX on Apple silicon, faster-whisper elsewhere |
| Language | Any OpenAI-compatible endpoint; Ollama by default |
| Speech out | Piper, out of process (it is GPL-3.0) |
| Video out | Envelope-driven viseme renderer, CPU only |
| Camera in | Motion-gated sampling into a vision model, off the turn path |
| Safety | Keyword crisis guardrail ahead of the model |
| Consent | Verified record required before any session opens |

Measured on an M3 Max, end of speech to first audio back:

| | |
|---|---|
| First turn of a call | ~800 ms |
| Warm median | **~600 ms** |
| Design budget (cloud hardware) | 1500 ms |

An earlier version of this file reported 2.0 s and blamed speech-to-text. That
was wrong: the latency test measured a single turn on a cold process, so it was
timing model loading. Warm speech-to-text is 72 ms, not 1409 ms. Models are now
warmed while the caller is still on the connecting screen, and the test runs
four turns and asserts against the median of the warm ones.

Not yet built, in order: the MuseTalk GPU renderer, voice cloning, retrieval
memory, billing. See section 12 of the design.

## Requirements

Nothing is installed on your machine. `uv` is a single binary in `.tools/`,
and LiveKit, Postgres and Redis run as containers.

- Docker
- Node 20+
- [Ollama](https://ollama.com) for the language and vision models

## Run it

```bash
# once
make install
ollama pull llama3.2:3b
ollama pull qwen2.5vl:3b

# every time
make infra-up                     # LiveKit, Postgres, Redis
make seed                         # demo avatar with a verified consent record
make gateway                      # http://localhost:8000
make web                          # http://localhost:3100
```

Open <http://localhost:3100> and click **Start call**. Allow the microphone;
the camera is optional, and without it the avatar can hear you but not see you.

The first call is slower than the rest — Whisper and the language model load
on demand.

## Tests

```bash
make test                         # 91 tests, no infrastructure needed
make test-e2e                     # 5 tests against real LiveKit and Ollama
make latency                      # measures and prints the turn latency
```

The end-to-end suite joins a real room as a headless participant, publishes
synthesised speech and a camera frame, and asserts that the avatar answers and
that what the camera showed reached the model. It does not use a browser, so it
proves media flows rather than that a page rendered.

## Layout

```
src/avatar/
  config.py         every environment variable, read in one place
  safety/           crisis guardrail, runs before the model
  renderer/         RendererStage protocol, CPU backend, asset builder
  vision/           frame sampling, description, scene state
  gateway/          consent gate, session issuance, agent dispatch
  realtime/         pipeline assembly, LiveKit video publishing
  persona.py        system prompt assembly
apps/web/           Next.js client
infra/              docker-compose
```

## Licensing

Two things here are easy to get wrong, and this project got both wrong at first:

- **`piper-tts` is GPL-3.0-or-later**, not MIT. It runs in its own container
  (`infra/piper/`) rather than being imported, so this is aggregation rather
  than linking. That container is the only GPL code in the project. The voice
  itself is clean: `rhasspy/piper-voices` is MIT and the davefx dataset is CC0.
- **MuseTalk's code is MIT but its distributed weights are CreativeML
  OpenRAIL-M.** Commercially usable, but Section II requires the use
  restrictions to be carried forward into customer terms as an enforceable
  provision. That has not been done yet and must precede the first paying
  customer.

The full table, with what was verified and how, is in section 3 of the design.

## Two things worth knowing

**The renderer is swappable by design.** `RendererStage` has four methods and
one contract suite that every backend must pass. The CPU backend runs anywhere
and looks synthetic; MuseTalk runs on CUDA and looks real. The call loop,
timing and barge-in are identical under both, which is what lets all of this be
developed without a GPU.

**Pipecat's LiveKit transport does not publish video.** `write_video_frame`
returns `False` unconditionally in 1.7.0 — video input works, output was never
wired up. Frames are published directly through the LiveKit SDK in
`src/avatar/realtime/video_publisher.py`. A test asserts the upstream gap still
exists, so this can be deleted when it closes.

## Consent

No session opens without a `consent_records` row in `verified` state naming a
rights-holder. Absent, pending, rejected and revoked all refuse with a reason,
and revocation takes effect on the next request.

This is not decoration. California's AB 1836 extends post-mortem personality
rights to digital replicas and Tennessee's ELVIS Act covers voice and likeness.
A product that recreates a dead person needs documented permission from whoever
holds those rights.

The seeded demo avatar is Cristóbal Colón, who died in 1506 and whose
personality rights have long expired. Its consent record demonstrates the
mechanism; it is not a template for a real one.
