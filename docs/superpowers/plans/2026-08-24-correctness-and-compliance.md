# Correctness and Compliance — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove a wrong latency narrative, two wrong licence claims, a GPL dependency shipping in-process, and a live regulatory gap — before any further feature work.

**Why first:** Every item below is under two days, and each one corrects something that is currently false in the repository or legally exposed. The two-week items (viseme bank, shared face module, MuseTalk renderer) are worth nothing if the foundation misstates its own licences.

**Spec:** `docs/superpowers/specs/2026-08-24-live-avatar-call-design.md`

## What the architecture review overturned

Four premises in the existing design and README were wrong. Three were verified independently before this plan was written.

| Claim as shipped | Reality | Verified |
|---|---|---|
| "STT is ~70% of turn latency (1409 ms)" | Cold model load. Warm STT is **72 ms** measured over four consecutive transcriptions | Yes — rerun locally |
| `speech.py`: "Piper … MIT" | **GPL-3.0-or-later**, loaded in-process, ships `COPYING` and `espeakbridge.so` | Yes — package METADATA |
| Design §3: MuseTalk weights "available for any purpose, even commercially" | Code is MIT; distributed weights on HuggingFace are **`creativeml-openrail-m`** | Yes — HF API |
| Design §6: MuseTalk locally is "an unusable development loop" | The 5-min/8-s figure is a **4 GB** RTX 3050. Measured on MPS: **10.8 fps** faithful, **30.4 fps** with TAESD | Not re-verified — architect's benchmark |

The last one is not acted on in this plan; it changes the MuseTalk build item, which comes later.

---

## Task 1: Warm the models at session start

The single highest-leverage latency change, and the one that makes the latency test honest.

**Files:**
- Create: `src/avatar/realtime/warmup.py`, `tests/realtime/test_warmup.py`
- Modify: `src/avatar/realtime/agent.py`, `tests/e2e/test_latency.py`

**Interfaces:**
- Produces: `async def warm(cfg, stt, tts, llm) -> WarmupReport`, where `WarmupReport` records per-stage elapsed milliseconds and any failure. Never raises.

- [ ] **Step 1: Write the failing unit test** — fakes for each service, asserting all are exercised, that a failing service is logged rather than propagated, and that the report records the failure.
- [ ] **Step 2: Run it, confirm failure.**
- [ ] **Step 3: Implement.** `asyncio.gather(..., return_exceptions=True)` over four coroutines: push ~1 s of silence through STT, request a one-token completion from the LLM, synthesise one short word through TTS, and construct the turn analyser eagerly. A warm-up failure must never prevent a call from starting — it costs latency, not the session.
- [ ] **Step 4: Call it from `run_agent()`** after `build_pipeline()` and before `runner.run()`.
- [ ] **Step 5: Restructure the latency test.** Replace the single measured turn with five sequential turns. Report turn 1 separately as cold; assert the budget against the **median of turns 2–5**. Without this the warm-up is unfalsifiable.
- [ ] **Step 6: Measure and record the real number. Commit.**

---

## Task 2: Correct the false claims in the repository

Documentation, but not cosmetic: three of these are licence statements that someone would rely on.

**Files:** `docs/superpowers/specs/2026-08-24-live-avatar-call-design.md`, `src/avatar/services/speech.py`, `README.md`

- [ ] **Step 1: Design §3** — split the MuseTalk row into code (MIT) and weights (OpenRAIL-M), and record the Section II flow-down obligation.
- [ ] **Step 2: Design §6** — remove the "unusable development loop" justification and record the measured MPS figures with the 4 GB caveat that produced the wrong number.
- [ ] **Step 3: `speech.py`** — correct the Piper comment from MIT to GPL-3.0-or-later and state the in-process linking consequence.
- [ ] **Step 4: README** — correct the latency section to cold/warm, and the licence line.
- [ ] **Step 5: Design §8** — add the regulatory items, each marked verified or unverified. Only EU AI Act Article 50 has been independently checked.
- [ ] **Step 6: Commit.**

---

## Task 3: Remove the GPL dependency from the process

`piper-tts` is GPL-3.0-or-later and is imported into the same process as proprietary application code. Running it as a separate service is aggregation rather than linking.

**Files:** `src/avatar/services/speech.py`, `infra/docker-compose.yml`, `src/avatar/config.py`, `tests/test_speech.py`

- [ ] **Step 1: Write the failing test** — `build_tts` returns an HTTP-backed service when `tts_backend="http"`, and the in-process path is reachable only when explicitly selected.
- [ ] **Step 2: Run it, confirm failure.**
- [ ] **Step 3: Add a Piper HTTP sidecar** to compose, and a `tts_backend` setting defaulting to `http`.
- [ ] **Step 4: Implement the branch** in `build_tts`.
- [ ] **Step 5: Verify the voice licence separately** — `es_ES-davefx-medium.onnx.json` carries no licence field. Record what is found, including "unknown".
- [ ] **Step 6: Confirm end to end, then commit.**

---

## Task 4: Machine-readable synthetic-content marking

EU AI Act Article 50 has applied since 2 August 2026 and requires providers of synthetic audio/image/video to mark outputs in a machine-readable way. Penalties reach €15M or 3% of worldwide turnover. The existing disclosure banner satisfies the human-facing half only.

**Files:** Create `src/avatar/marking/watermark.py`, `src/avatar/marking/manifest.py`, `tests/marking/`; modify `src/avatar/renderer/processor.py`

**Interfaces:**
- Produces: `embed(frame_rgb, payload_bits) -> np.ndarray`, `detect(frame_rgb) -> bytes | None`, and `build_manifest(avatar_id, session_id, model_ids) -> dict`.

- [ ] **Step 1: Write the failing tests** — a marked frame decodes back to its payload; an unmarked frame returns `None`; marking is imperceptible above a stated PSNR floor; marking survives the RGB→RGBA conversion the publisher performs.
- [ ] **Step 2: Run them, confirm failure.**
- [ ] **Step 3: Implement.** A fixed-pattern spatial watermark carrying a short identifier, applied to every rendered frame; plus a declaration manifest recording which models produced the session.
- [ ] **Step 4: Wire into the renderer** so no published frame is unmarked, including idle frames.
- [ ] **Step 5: Verify end to end** — subscribe as a headless participant, decode the payload from a received frame.
- [ ] **Step 6: Commit.**

**Honest scope:** this is a defensible engineering answer to a marking obligation, not legal compliance certification. A C2PA-conformant implementation and counsel review are both still required; the manifest is structured to carry C2PA assertion fields so that path stays open.

---

## Deferred to the next plan

| Item | Effort | Why not now |
|---|---|---|
| Phoneme-indexed viseme bank from real frames | 1.5–2 wks | Requires design §5.6 to change to a *talking* source clip. Decide that first. |
| Shared licence-clean face module (MediaPipe) | 2 wks | Unblocks both MuseTalk and LivePortrait. The real next milestone. |
| `MuseTalkRenderer` | 1.5 wks | Depends on the face module. Batch size capped at 8 by the cancel contract. |
| OpenRAIL-M Attachment A into customer terms | lawyer | Must precede the first paying customer. |
| LivePortrait photo mode | 1 wk | Depends on the face module. |
| GPU pool and concurrency measurement | 1.5 wks | Cannot measure anything real until MuseTalk runs. |
