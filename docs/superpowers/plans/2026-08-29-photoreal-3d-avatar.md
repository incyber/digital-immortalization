# Photoreal 3D Avatar — Implementation Plan

**Goal:** Build a photoreal 3D likeness of a specific person from photographs
and/or video, as a Gaussian splat, animate it with facial and body motion
driven by conversation, and stream it live into a video call the customer can
talk to.

**Renderer: Gaussian splatting.** A mesh *approximates* a face - a model fitted
until it resembles someone. A splat *reconstructs* what a camera actually saw.
For the only question that decides this product - does the family recognise him
- that difference is the whole game. A splat also renders in the viewer's own
browser, which removes the per-call GPU that was the main threat to the unit
economics.

**Why the renderer is being replaced:** every method built so far animates a
2D image. A 2D image has no body, so it cannot stand, walk, gesture or leave
frame — no amount of work on that path reaches the requirement. A 3D character
has a body by construction, and locomotion becomes ordinary animation.

---

## The requirement, in full

Recorded here because it has been lost twice. Anything not on this list is out
of scope; anything on it that a phase does not deliver must be named as
deferred, not quietly dropped.

**The avatar**

1. Built from photographs and/or video the customer supplies. Full-body source
   material is **not** required and usually will not exist.
2. Body type — height, build, weight distribution — inferred from the images
   and from what the family states, not scanned.
3. Photoreal. The bar is "the family says that is him", judged in motion.
4. Moves like a person: walks, stands, sits, turns, steps out of frame.
5. Gestures and expression driven by **what is being said and how it feels**,
   changing continuously. Never a loop.
6. Speaks in the person's own cloned voice.
7. Lip movement matches the words naturally.

**The call**

8. Live two-way conversation, low latency, interruptible mid-sentence.
9. The avatar sees the customer through their camera, roughly once a second,
   and can respond to what it sees.
10. Runs hosted, on the web. Not on a developer's machine.

**The platform**

11. Multi-tenant: each customer registers and gets isolated storage.
12. Scales to hundreds of concurrent customers.
13. Every dependency usable commercially. No research-only models anywhere in
    the shipped path.
14. Rights and consent recorded before any call. Synthetic-media declaration on
    every session. Crisis lines by country.
15. GPU spend bounded by construction: nothing can be left running.

---

## Architecture

    photographs / video
        -> face reconstruction        3D head geometry + skin texture
        -> body inference             shape parameters from images + stated build
        -> character assembly         head + body + hair + clothing, rigged
        -> animation system           locomotion, idles, gestures, facial rig
        -> real-time renderer         one GPU per active call
        -> WebRTC                     into the existing call page

    conversation (existing, unchanged)
        speech in -> language model -> emotion + text
                                    -> voice clone -> audio
                                    -> motion intent -> animation parameters

**What survives unchanged.** The gateway, tenant isolation, consent gate,
persona system, crisis handling, synthetic-media declaration, photo and video
ingestion, R2 storage, LiveKit transport, the web application, and the
conversation loop itself. This plan replaces the renderer and adds an asset
pipeline. It does not restart the product.

**What is deleted.** The plate renderer, the LivePortrait base-clip path, and
the MuseTalk service — once, and only once, the replacement is proven better.

---

## Global constraints

- Every model, mesh and texture in the shipped path must permit commercial use.
  Verified at the source licence, never from a summary. **FLAME 2023 is
  CC-BY and permits commercial use; FLAME 2017/2019/2020 do not.** SMPL-X,
  BFM and InsightFace are research-only unless separately licensed.
- No GPU may be created without a mechanism that stops it if the orchestrator
  dies. Existing pull-lease design applies unchanged.
- No production deploy without the customer asking for that deploy.
- Latency budget: end of the customer's speech to first frame of reply, under
  1.5s hosted.
- 25fps minimum at 720p for the rendered stream. Below that it is not a call.

---

## Component decisions, verified at source

Four parallel investigations settled the stack. Every licence below was read at
its own source and the operative sentence quoted, because this project has
already been bitten twice by a permissive code licence sitting on top of a
restricted model.

**Disqualified, and why**

| Component | Blocker |
|---|---|
| MICA, DECA, EMOCA, inferno | MPI licence: "any use for commercial ... purposes is prohibited" |
| HRN | Apache-2.0 code, but depends on the Basel Face Model: "strictly prohibited ... for any direct or indirect for-profit purposes" |
| FLAME *texture* model | CC BY-NC-SA. Distinct from FLAME 2023 geometry, which is fine |
| SMIRK | MIT code, but the checkpoint is trained on CelebA, FFHQ, MEAD and LRS3 - all non-commercial. The weights are the product |
| AvatarMe | Commercialised and never released. Nothing to license |
| SMPL-X, STAR, SUPR | "any use for commercial ... purposes is prohibited". Commercial terms via Meshcapade, no published price |
| OpenMVS | AGPL, network copyleft |
| Photogrammetry from a family album | Needs 40-150 images, one session, consistent light, 70-80% overlap. An album is the same person across decades, not a static object |

**Chosen**

| Part | Choice | Licence |
|---|---|---|
| Splat optimiser and renderer | gsplat (Nerfstudio) | Apache-2.0 |
| Browser renderer | Spark | MIT |
| Photos-only route | TRELLIS, or LGM / TripoSR | MIT |
| Rig for expression and head pose | FLAME 2023 | CC-BY |
| Landmarks | MediaPipe, already in this system | Apache-2.0 |
| Skin, where a camera saw it | the customer's own images | ours |
| Body, if locomotion is needed | MakeHuman CC0 assets, our own blending | CC0 assets |

**Not used, and why.** The original Inria implementation forbids commercial use
outright. GaussianAvatars is CC-BY-NC-SA (Toyota Motor Europe): "Any commercial
use ... is strictly prohibited". SplattingAvatar is CC-BY-NC-SA and depends on
SMPL. All three prove the technique and none may ship, so the pipeline is
rebuilt clean-room against gsplat and FLAME 2023.

**One thing genuinely unresolved.** Inria's licence notes registration with the
Agence pour la Protection des Programmes. APP is France's software copyright
deposit registry, not a patent office, so it restricts their code rather than
the technique - and gsplat is an independent implementation. But no patent
search was completed, so this is "probably fine, verify before scaling revenue
on it" rather than cleared. A real search of Espacenet, USPTO and WIPO for
filings citing the 3DGS authors is required before this carries a business.

**Two decisions that came out differently than expected**

*No learned face encoder.* Every commercially clean encoder turned out to have
research-licensed weights. Fitting FLAME 2023 directly by optimisation needs no
learned weights at all, and it is the better use of a family album anyway:
twenty photographs are twenty independent observations of one fixed face,
optimised jointly, where a single-image encoder discards nineteen of them.

*No Pixel Streaming.* LiveKit ingest speaks WHIP, RTMP and SRT; Unreal's
signalling is its own protocol, so there is no cable between them. Instead take
Unreal's rendered frames off the GPU and publish them through the LiveKit
publisher this project already has. One encode instead of two, and an entire
subsystem - signalling, matchmaker, a second WebRTC hop - deleted.

**Body shape cannot be inferred from head-and-shoulders photographs.** No
published method does it; every body regressor needs a visible torso. So the
family adjusts it: height, build, shoulders, with a live preview. That is
honest, it is better than a silent guess, and it puts the judgement with the
people who knew him.

---

## Two intake routes, one artefact

Both end in a Gaussian splat of that person, rigged to FLAME 2023 and driven by
the motion system already built. The customer chooses neither; whichever they
have decides.

**Video, when they have it.** One session of the person talking gives genuine
multi-view coverage over time. Every Gaussian comes from a camera that actually
saw them, which is the best likeness obtainable.

**Photographs, when that is all there is.** The missing views are *generated* -
TRELLIS and LGM both emit 3D Gaussians directly from a single image - anchored
on the photograph that best represents them, with the remaining photographs
correcting skin and detail.

**The limit, stated once so it is not discovered later.** Photographs taken
years apart cannot be *reconstructed* into one splat. It is not a volume
problem: static splatting assumes one geometrically consistent subject, and a
person at forty and at fifty-five has no single 3D truth. Generation solves
this by inventing the unseen angles from a model trained on many faces - so a
photo-only likeness is partly invented where no camera looked. It will look
right. Some of it is not measured. Thirty photographs across three decades will
not beat one thirty-second video.

---

## Phase 0 — Prove it, before building it

**Nothing in Phase 1 starts until all four gates pass.** Every architecture
proposed for this product so far failed on a fact that could have been checked
first. These are the checks.

### Gate 1: Does it look like the person?

The entire product dies here if it fails, and it is the cheapest thing to test.

- [ ] Take one real photograph set of one real person.
- [ ] Produce a 3D head from it — geometry and skin texture.
- [ ] Render it under three lighting conditions, still and in motion.
- [ ] Put it beside the photographs and show it to someone who knows the person
      without telling them which is which.

**Run it twice**, because there are two kinds of customer: once from a video,
once from photographs alone, same person, same blind sheet, the same judge told
nothing about which is which.

**Pass:** they identify the person unprompted.
**Fail:** stop. No later phase compensates for a face that is not them.

The harness for this exists: `python -m avatar.cli.likeness blind`.

### Gate 2: Is the splat pipeline legally clear enough to build a business on?

Not verifiable by automated fetch — their licence pages block it. A human must
read the current MetaHuman and Unreal Engine terms and answer:

- [ ] May MetaHuman assets be used in a commercial product that is not a game?
- [ ] May they be streamed to end users from a server?
- [ ] Is there any restriction on creating a likeness of a real person, or of a
      deceased person?
- [ ] Does the 5% Unreal royalty apply to this product's revenue?
- [ ] May MetaHuman assets be used outside Unreal Engine?

**If any answer blocks the product, take the open route** — FLAME 2023 head,
custom rig, web renderer. Lower realism, no licence dependency on Epic. This is
the reason Gate 2 comes before any Unreal work rather than after it.

### Gate 3: Can it render in the viewer's browser?

Server-side splat rendering is known fast - 300fps at 512px on a desktop GPU is
published. The question that matters commercially is the client:

- [ ] Render a real splat in a browser via Spark, on a mid-range phone.
- [ ] Measure frame rate, and the download size of the splat.
- [ ] Confirm the server can drive expression parameters cheaply while the
      viewer's device does the rendering.

**Pass:** 25fps on a mid-range phone, with a download a customer will tolerate.
Per-call GPU cost then goes to zero.
**Fail:** fall back to rendering on a rented GPU per call, and the unit
economics get the cost back.

### Gate 3b: Can one GPU render a call in real time?

- [ ] Stand up one character in the chosen renderer.
- [ ] Stream it to a browser over WebRTC.
- [ ] Measure sustained frame rate at 720p with animation and facial rig active.
- [ ] Measure GPU cost per hour of call.

**Pass:** 25fps sustained, cost per call hour known and acceptable.
**Fail:** the product is offline video messages, not live calls. Say so before
building the live path.

### Gate 4: Does inferred body shape look right?

- [ ] Estimate body parameters from head-and-shoulders photographs plus a
      stated height and build.
- [ ] Render the full figure.
- [ ] Show it to someone who knew the person.

**Pass:** the build reads as plausibly theirs.
**Fail:** the avatar is framed from the chest up and locomotion is dropped.
That is a smaller product, and knowing it now is worth more than discovering it
in month two.

**Phase 0 output:** a written finding for each gate, with evidence, and a
go/no-go on the Unreal route versus the open route.

---

## Phase 1 — One character, one call, end to end

The narrowest path that proves the whole chain. One hand-built character, no
pipeline, no automation.

- [ ] Build one character manually from one photograph set.
- [ ] Rig it: skeleton, facial blendshapes, viseme set.
- [ ] Drive its mouth from an audio file and confirm the lip movement reads.
- [ ] Render it to a video track and publish into a LiveKit room.
- [ ] Join that room from the existing call page and see it.
- [ ] Wire the existing conversation loop to it: speech in, reply out, mouth
      moving, interruptible.

**Deliverable:** a call with a 3D character that talks. Ugly around the edges,
end to end, real.

---

## Phase 2 — Motion that means something

Design settled. Motion is scheduled on the **speech timeline, never the wall
clock**: the model produces a sentence before the voice synthesises it, and the
voice synthesises it before it is rendered, so a nod that must peak on a
stressed syllable can be started before that syllable is heard. On a wall clock
every nod arrives late.

Five layers summed each frame - persona baseline, affect, prosody, gesture, and
an aperiodic noise layer - then rate-limited per channel. The anti-loop
property lives in the last two: drift is a sum of random walks at several time
constants rather than any oscillator, and gestures are generated from
parameterised families rather than replayed, so two nods in one session are
never the same curve.

The requirement is behaviour driven by content, never a loop.

- [ ] Extend the language model's reply with an emotional intent field.
- [ ] Derive prosody from the generated audio: stress, pauses, pitch contour.
- [ ] Map intent plus prosody onto animation parameters — head pose, gaze,
      brow, blink timing, posture, gesture selection.
- [ ] Blend between animation states so transitions are not visible cuts.
- [ ] Verify the same sentence twice produces different motion.

**Deliverable:** the avatar nods, looks away in thought, leans in, and its face
changes with the meaning of what it says.

---

## Phase 3 — Locomotion and staging

- [ ] Walk, turn, sit, stand, step out of frame and return.
- [ ] Decide when movement is appropriate — the model asks for it, or a
      staging rule triggers it.
- [ ] Camera framing that follows sensibly rather than losing the subject.

**Deliverable:** the avatar behaves like a person in a room, not a bust.

---

## Phase 4 — The asset pipeline

Only after a hand-built character has proven the chain. Automating a bad
result produces bad results faster.

- [ ] Photographs in, 3D head out, no human step.
- [ ] Body parameters inferred from images and stated build.
- [ ] Hair and clothing chosen automatically, with a manual override.
- [ ] Assembly, rigging and packaging automated.
- [ ] Build time and GPU cost per avatar measured and bounded.
- [ ] Wire into the existing avatar build flow so the progress bar covers it.

**Deliverable:** a customer uploads photographs and gets a character, unattended.

---

## Phase 5 — Scale and cost

- [ ] One GPU per active call, allocated on call start.
- [ ] Released on call end, and released anyway if the orchestrator dies.
- [ ] Concurrency limit and per-tenant quota.
- [ ] Cost per call measured and reported.
- [ ] Load test at the concurrency the business needs.

**Deliverable:** hundreds of customers without a GPU left running.

---

## Phase 6 — Vision, and finishing

- [ ] Camera sampled once a second into a hosted vision model.
- [ ] What it sees reaches the conversation.
- [ ] Synthetic-media declaration and watermarking on the new renderer.
- [ ] Consent gate verified against the new path.

---

## Risks, ranked

1. **Likeness is not convincing.** Kills the product. Gate 1, first.
2. **Epic's licence forbids this use.** Forces the open route and a realism
   drop. Gate 2, before any Unreal work.
3. **Per-call GPU cost makes the unit economics impossible.** Gate 3.
4. **Uncanny valley.** Real-time 3D humans are identifiably synthetic in
   motion. This is the field's hardest open problem and will not be solved
   here. The bar is "that is him", not "I could not tell it was rendered".
5. **A research-only dependency reaches production.** Every model licence
   verified at source before it is used, not after.
6. **Scope drift back to 2D.** The 2D path cannot meet the requirement. If a
   phase stalls, the answer is a smaller 3D product, not a return to images.

---

## What is explicitly not in this plan

- Photorealism indistinguishable from recorded video. Not achievable in real
  time by anyone today.
- Reconstructing a body from photographs with the accuracy of a scan. The body
  is inferred and plausible, not measured.
- Hands at film quality. Real-time hand animation is coarse; gestures will read
  from a distance and not survive a close-up.
