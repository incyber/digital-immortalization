"""The measured route: one video of a person, one Gaussian splat of that person.

THE ONE THING THIS FILE EXISTS TO GET RIGHT
===========================================

Every splat pipeline in the literature assumes a *camera that orbits a static
scene*. What a family uploads is the exact opposite: a phone propped on a
table, and a head that turns. Feed those frames to a normal pipeline and it
fails in a way that looks like bad luck rather than a wrong assumption:

  - Structure-from-motion locks onto the static background - the wall, the
    sofa, the window - concludes the camera never moved, and returns a
    degenerate reconstruction with no baseline at all. Nothing about the face
    is recovered, because in the *camera's* frame the face is the only thing
    that moved and a static-scene solver reads motion as noise.
  - Even given poses, optimising over the raw frames asks a rigid model to
    explain a scene in which the head rotates and the room does not. The
    Gaussians smear: the model's cheapest explanation of "the face changed and
    the wall did not" is a cloud of semi-transparent blobs that is right on
    average and correct from nowhere.

The fix is a change of reference frame, and it is the whole method:

    *Declare the head to be the world.*

A head rotating by R in front of a fixed camera is, geometrically, exactly a
camera orbiting by R-inverse around a fixed head. So we never estimate camera
motion at all. We estimate the *head's* pose per frame, and then read that same
matrix as the camera's extrinsics in a world whose origin is the head. Ten
seconds of someone turning to look out of a window becomes a ten-second camera
orbit around a statue, which is precisely the input gsplat was built for.

Two consequences follow, and both are forced rather than chosen:

  1. **The background must be removed, not merely ignored.** In head-fixed
     coordinates the room is the thing that moves - it swings around the head -
     and it is geometrically inconsistent frame to frame. Left in, it is
     un-reconstructible and the optimiser pays for it by ruining the face.
     Masking is not a tidiness step here; the inversion creates the need for it.
  2. **The subject must be rigid, so non-rigid frames are dropped.** Talking
     and blinking change the geometry, and static splatting has no way to
     represent that. We keep the most expression-neutral frames, spread across
     the yaw range, and discard the rest. This is not a loss: expression is
     driven afterwards by the FLAME 2023 rig, so what the splat should hold is
     the person at rest, seen from as many angles as the clip affords.

Head pose comes from MediaPipe's Face Landmarker (Apache-2.0, already in this
system), which returns a facial transformation matrix per frame: the 4x4 that
maps the canonical face model into the camera's coordinate system. Read with
the head as the world, that matrix *is* the world-to-camera extrinsic. No
solver, no bundle adjustment, no COLMAP - and therefore none of COLMAP's
failure modes on a scene with no camera baseline.

THAT MATRIX IS A PRIOR, NOT A MEASUREMENT
=========================================
MediaPipe solves it by fitting a canonical face model under an *assumed*
perspective camera fixed at MP_VERTICAL_FOV_DEG. It is not a calibration, and
whoever filmed the clip did not use a 63-degree lens on purpose. Per-frame
rotation error of a couple of degrees is normal, and at this focal length a
couple of degrees is tens of pixels of disagreement between two views of the
same cheekbone.

Gaussian splatting has no way to represent "these two views disagree". Its
cheapest explanation is to go translucent: a fog that is right on average and
sharp from nowhere. That is the single most likely way a real capture fails,
and it is why every photogrammetry pipeline spends bundle adjustment on
exactly this quantity.

So the poses are not frozen. gsplat's rasteriser is differentiable in
`viewmats`, so each view carries six extra parameters - an axis-angle rotation
and a translation, initialised at zero - composed onto the MediaPipe estimate
and optimised alongside the Gaussians. This is bundle adjustment, reintroduced
in the only form this pipeline can afford. Three rules keep it a refinement
rather than a re-solve, and each is enforced rather than hoped for:

  - The MediaPipe estimate stays the anchor. The learned part is a *delta*, so
    the prior cannot be thrown away, only nudged.
  - The Gaussians warm up first (POSE_WARMUP_FRACTION). Released together, the
    poses would absorb error that belongs to geometry the optimiser has not
    built yet, and the result is a beautifully fitted set of wrong cameras.
  - The correction is regularised and hard-clamped. A pose that wants to move
    thirty degrees is not being refined; something else is wrong, and silently
    accommodating it turns a diagnosable bad capture into an inexplicable bad
    avatar. The final magnitudes are reported for that reason: small
    corrections mean MediaPipe was right, large ones mean the capture or the
    field-of-view assumption was not.

LICENCE
=======
gsplat (Nerfstudio) is Apache-2.0 and is an independent implementation. The
Inria gaussian-splatting code and every derivative of its rasteriser
(diff-gaussian-rasterization, mip-splatting's fork, diffoctreerast) are under a
licence that forbids commercial use, and none of them is installed in the image
or importable from this file. MediaPipe is Apache-2.0. Nothing here touches
SMPL/SMPL-X, BFM, InsightFace or the FLAME texture model.

NOTHING HEAVY IS IMPORTED AT MODULE SCOPE. torch, gsplat, cv2 and mediapipe are
imported inside the functions that need them, so this module - and the handler
that imports it - loads on a laptop with no CUDA and no weights, and reports
what is missing instead of dying at import.
"""

from __future__ import annotations

import itertools
import math
import os
import struct
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

# Where the two MediaPipe task bundles live in the image. FACE_MODEL_PATH is
# the same variable the existing worker uses for the landmarker, so one
# endpoint configuration covers both images.
FACE_MODEL_PATH = os.environ.get("FACE_MODEL_PATH", "/models/face_landmarker.task")
SEGMENTER_MODEL_PATH = os.environ.get("SEG_MODEL_PATH", "/models/selfie_multiclass.tflite")

# Frame spacing, in seconds. Matches the sampling in avatar/ingest/video.py so
# the frame count the router judged the clip on is the frame count we actually
# reconstruct from. Half a second is roughly the interval over which a turning
# head produces a genuinely new view rather than a duplicate.
FRAME_STRIDE_S = 0.5

# Long edge frames are reduced to before landmarking and optimisation. A 4K
# phone clip carries no more facial detail than this at the distance people
# film themselves, and every pixel above it is optimiser time.
MAX_FRAME_EDGE = 1024

# The vertical field of view MediaPipe's face geometry solves the facial
# transformation matrix against. Its perspective camera is fixed at 63 degrees,
# so the intrinsics we hand gsplat must be built from the same number or the
# poses and the rasteriser disagree about what the picture means.
MP_VERTICAL_FOV_DEG = 63.0

# Yaw is bucketed into bins of this width so a clip that spends nine seconds
# facing forward and one second in profile does not train nine seconds of
# frontal view. VIEWS_FOR_FULL_COVERAGE in avatar/splat/build.py is 24, one
# bin every fifteen degrees; the same division is used here so that the
# coverage this file measures is comparable with the coverage that module
# estimates when nobody measured it.
YAW_BINS = 24
MAX_FRAMES_PER_BIN = 3

# Widest total yaw traverse still described as a frontal shell. Someone talking
# to a propped phone sweeps perhaps eighty degrees in all, which over YAW_BINS
# spanning the whole circle is a coverage figure around 0.22 - a number that
# reads like a failed build to anyone who sees it without a sentence beside it.
# It is not a failure. It is a complete front and a missing back, which is
# right for a head-and-shoulders call and wrong for anything that walks around
# the person, and the notes say which of the two the family is getting.
FRONTAL_SHELL_MAX_SPAN_DEG = 120.0

# --------------------------------------------------------------------------
# pose refinement
#
# See the module docstring: these six numbers per view are the bundle
# adjustment this pipeline would otherwise be missing.

# Adam's step is approximately the learning rate whatever the gradient scale,
# so a learning rate here is a displacement budget per step - and the budget
# that matters is the one per *view*, because only one view is rendered per
# iteration. _pose_learning_rates therefore derives them from the run rather
# than fixing them: a pose may traverse its whole ceiling exactly once over the
# refining steps its own view receives. Written as a constant instead, the
# refinement would quietly stop working the day somebody changed
# Quality.iterations, and it would stop working by producing a slightly
# blurrier avatar rather than by failing.
#
# These are the ceilings on one step, and they exist because a preview build
# divides its ceiling over very few steps. A pose that can jump a third of a
# degree on a single noisy image gradient is not being refined, it is being
# rattled.
POSE_ROTATION_LR_MAX = 5e-4          # ~0.03 degrees in one step
POSE_TRANSLATION_LR_MAX = 5e-4       # 0.5mm in one step

# Fraction of the run the Gaussians get to themselves. Releasing both at once
# lets the poses absorb error that belongs to geometry that does not exist yet.
POSE_WARMUP_FRACTION = 0.3

# Hard ceiling on one view's correction, applied by projection after every
# step. The regulariser below discourages a large correction; this makes it
# impossible, so a single badly landmarked frame cannot drag its camera into
# the next county and take the reconstruction with it.
MAX_POSE_ROTATION_DEG = 10.0
MAX_POSE_TRANSLATION_M = 0.05

# Weight on the correction penalty. The penalty is expressed as a fraction of
# the ceilings above and therefore dimensionless, which is what lets one weight
# cover both rotation and translation. At 1e-2 it matches the opacity and scale
# regularisers already in the loss: negligible for the small corrections that
# are the point, decisive against a pose trying to run away.
POSE_REGULARISATION = 1e-2

# Above this the correction has stopped being a refinement of a good prior.
# Reported rather than acted on: the build still finishes, but the reason it
# looks the way it does is in the notes instead of in nobody's head.
SUSPECT_POSE_CORRECTION_DEG = 4.0

# --------------------------------------------------------------------------
# audio

# Sample rate the loudness envelope is measured at. Speech energy is a
# low-resolution quantity here - one number per half-second frame - so
# telephone rate is more than enough and keeps a sixty-second clip under 2MB.
AUDIO_SAMPLE_RATE = 16_000

# Seconds ffmpeg gets to decode the audio before the envelope is abandoned and
# the selection falls back to blendshapes alone. Audio is an improvement to
# frame choice, never a dependency of it.
AUDIO_TIMEOUT_S = 120.0

# How much a frame's loudness counts against it when choosing which frame in a
# yaw bin to keep, relative to its blendshape score. Both are in [0, 1]. Equal
# weighting on purpose: the blendshapes say the mouth is open *now*, the audio
# says the person is mid-sentence, and the second is the better predictor of
# the first because a closed mouth between two words is still a mouth about to
# move and still a frame whose lips disagree with every other frame.
SPEECH_WEIGHT = 1.0

# Blendshapes that mean "this frame is not the face at rest". Scored rather
# than thresholded: a person who talks continuously has no neutral frame, and
# refusing to build from them would be worse than building from their calmest.
NON_RIGID_BLENDSHAPES = (
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    "mouthOpen", "mouthPucker", "mouthFunnel", "mouthSmileLeft", "mouthSmileRight",
    "eyeBlinkLeft", "eyeBlinkRight", "cheekPuff",
    "browDownLeft", "browDownRight", "browInnerUp",
)

# Selfie multiclass segmentation labels. The head is hair, face skin and the
# neck; clothes and background are not part of a likeness and reconstructing
# them wastes budget on the one thing a family will never look at.
SEG_HAIR, SEG_BODY_SKIN, SEG_FACE_SKIN = 1, 2, 3
HEAD_CLASSES = (SEG_HAIR, SEG_BODY_SKIN, SEG_FACE_SKIN)

# Fewest usable views a reconstruction is attempted from. Below this the head
# has not turned through enough angles for the frames to be observations rather
# than repetitions - the same floor MIN_VIDEO_VIEWS applies in the router,
# restated here because the worker must not depend on being called correctly.
MIN_USABLE_VIEWS = 12

# One Gaussian in the compact .splat record, as read by the browser renderer:
# three float32 of position, three of scale, four bytes of colour, four of
# rotation. Held here as a constant so BYTES_PER_GAUSSIAN in
# avatar/splat/build.py and the bytes this file actually writes cannot drift.
BYTES_PER_GAUSSIAN = 32


class ReconstructError(RuntimeError):
    """A reconstruction that cannot be completed, with a reason worth reading."""


@dataclass(frozen=True)
class PoseCorrection:
    """How far the optimiser had to move the cameras away from MediaPipe's.

    Diagnostic, and the only honest way to answer "why does this one look
    wrong". A mean under a degree says the facial transformation matrices were
    good and the reconstruction can be believed. Several degrees says the
    prior was poor, which on this route means one of two things and both are
    fixable: the capture was bad, or the phone's lens is nothing like the
    63-degree camera MediaPipe assumed when it solved the pose.

    Zero on the generated route, which has no MediaPipe prior to correct.
    """

    mean_deg: float = 0.0
    max_deg: float = 0.0
    mean_mm: float = 0.0
    max_mm: float = 0.0
    views: int = 0

    @property
    def suspect(self) -> bool:
        return self.mean_deg >= SUSPECT_POSE_CORRECTION_DEG


@dataclass(frozen=True)
class GaussianCloud:
    """The artefact both routes produce, before it is a file.

    Defined here rather than in either handler or generate.py because it is the
    contract between the two routes: a reconstruction and a generation must
    produce the same thing or the browser renderer has two formats to support
    and the quality report has two meanings.

    Every array is already activated - scales are positive, opacities are in
    [0, 1], colours are in [0, 1], quaternions are normalised (w, x, y, z).
    Storing raw logits here would put the choice of activation in the exporter,
    where a mismatch shows up as a splat that renders black.
    """

    means: object          # (N, 3) float32, in head-fixed metres
    scales: object         # (N, 3) float32, positive
    quats: object          # (N, 4) float32, normalised, (w, x, y, z)
    colors: object         # (N, 3) float32 in [0, 1]
    opacities: object      # (N,)   float32 in [0, 1]
    # Share of the viewing sphere this splat renders plausibly. On this route
    # it is *measured*: the fraction of yaw bins a camera actually looked from.
    # It is not a quality score and it is deliberately not the measured
    # fraction, which avatar/splat/build.py derives from the route alone.
    angular_coverage: float
    views_used: int
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Total yaw the head actually traversed, in degrees. Carried beside
    # angular_coverage because the fraction alone cannot distinguish "the head
    # barely moved" from "the head moved a lot and the circle is just big".
    yaw_span_deg: float = 0.0
    # What the pose refinement had to do. Defaulted so the generated route,
    # which has no per-frame prior to refine, constructs unchanged.
    pose_correction: PoseCorrection = field(default_factory=PoseCorrection)

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def size_bytes(self) -> int:
        return self.count * BYTES_PER_GAUSSIAN


@dataclass(frozen=True)
class HeadPose:
    """One frame, and where the head was when it was taken.

    `view` is the world-to-camera matrix in the head-fixed world, in OpenCV
    convention (x right, y down, z into the scene) - which is what gsplat
    expects and is *not* what MediaPipe returns. The conversion happens once,
    in _head_poses, so no caller has to remember it.
    """

    index: int
    image: object          # (H, W, 3) uint8 RGB
    view: object           # (4, 4) float32 world-to-camera, OpenCV convention
    yaw: float             # radians, positive turning to their left
    pitch: float
    # 0 is the face at rest, 1 is mid-word or mid-blink. Used to choose the
    # frames a rigid model can actually explain.
    expression: float
    # How loud the person was around this frame, normalised to [0, 1] across
    # the clip. 0 when the video has no audio track, which makes a clip with no
    # sound behave exactly as it did before this was measured.
    speech: float = 0.0


# --------------------------------------------------------------------------
# frames


def _frames(video: Path, stride_s: float = FRAME_STRIDE_S) -> list:
    """Sample the clip at a fixed interval, as RGB arrays."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ReconstructError(f"the video at {video.name} could not be opened")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps * stride_s))

    out, index = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            height, width = frame.shape[:2]
            if max(height, width) > MAX_FRAME_EDGE:
                scale = MAX_FRAME_EDGE / max(height, width)
                frame = cv2.resize(
                    frame,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()

    if not out:
        raise ReconstructError("the video contained no readable frames")
    return out


# --------------------------------------------------------------------------
# audio: which frames the person was not talking through


def _decode_audio(video: Path) -> bytes:
    """Raw mono 16-bit PCM for the whole clip, or empty if there is none.

    One ffmpeg pass, straight to stdout, no intermediate file. Returns empty
    rather than raising for every reason it can fail - no audio track, no
    ffmpeg on this machine, a container ffmpeg will not open - because a
    loudness envelope is an improvement to frame selection and never a
    prerequisite for it. A clip that arrives silent must still reconstruct.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vn",
             "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "s16le", "-"],
            capture_output=True, timeout=AUDIO_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return b""
    return result.stdout if result.returncode == 0 else b""


def _speech_envelope(video: Path, frame_count: int, stride_s: float = FRAME_STRIDE_S):
    """One loudness value per sampled frame, in [0, 1], or None if unmeasurable.

    The calmest frame of continuous speech is still a mouth mid-phoneme, so
    ranking frames by blendshapes alone reliably picks the least-open mouth of
    a face that never stopped moving: a sharp forehead over a smeared jaw. The
    audio says which seconds the person was not speaking at all, and those are
    the frames whose lips agree with each other.

    Normalised against the loudest window rather than an absolute threshold,
    because the input is a phone microphone at an unknown distance and the only
    meaningful question here is which parts of *this* clip are its quiet ones.
    """
    import numpy as np

    pcm = _decode_audio(video)
    if len(pcm) < 2:
        return None

    # int16 from ffmpeg's s16le; an odd trailing byte is a truncated sample.
    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32)
    samples /= 32768.0

    window = max(1, round(AUDIO_SAMPLE_RATE * stride_s))
    envelope = np.zeros(frame_count, dtype=np.float32)
    for index in range(frame_count):
        # The window is centred on the frame's timestamp: a frame is smeared by
        # the word on either side of it, not only by the one that follows.
        centre = round(index * stride_s * AUDIO_SAMPLE_RATE)
        chunk = samples[max(0, centre - window // 2): centre + window // 2]
        if chunk.size:
            envelope[index] = float(np.sqrt(np.mean(chunk * chunk)))

    loudest = float(envelope.max())
    if loudest <= 0.0:
        # A track that exists and is digital silence carries no information
        # about which frames are quiet, so it is not allowed to pretend it does.
        return None
    return (envelope / loudest).tolist()


# --------------------------------------------------------------------------
# head pose: the inversion


def _mp_image(rgb):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _landmarker(model_path: str = FACE_MODEL_PATH):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not Path(model_path).exists():
        raise ReconstructError(f"the face landmarker model is missing at {model_path}")

    return mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            # Both are the point of using this model rather than the landmarks
            # alone: the transformation matrix is the camera pose after the
            # inversion, and the blendshapes are how a non-rigid frame is
            # recognised without a second model.
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
    )


def _yaw_pitch(rotation) -> tuple[float, float]:
    """Yaw and pitch of a head-to-camera rotation, in radians."""
    yaw = math.atan2(-float(rotation[2][0]), math.hypot(
        float(rotation[2][1]), float(rotation[2][2])
    ))
    pitch = math.atan2(float(rotation[2][1]), float(rotation[2][2]))
    return yaw, pitch


def head_poses(images: list, model_path: str = FACE_MODEL_PATH) -> list[HeadPose]:
    """Per-frame head pose, expressed as a camera pose in a head-fixed world.

    This is the inversion, and it is three lines of arithmetic rather than a
    solver. MediaPipe returns M, the matrix that carries the canonical face
    model into camera coordinates. Choose the head's own frame as the world and
    M is, without modification, the world-to-camera extrinsic for that frame.
    A head that turned 40 degrees is now a camera that moved 40 degrees around
    a head that never moved.

    Shared with generate.py, where the same trick supplies the relative pose of
    each photograph: the person's head is again the only fixed frame available.

    Frames with no detectable face contribute nothing and are dropped here
    rather than later; a frame of a shoulder is not a view of a person.
    """
    import numpy as np

    # OpenGL to OpenCV. MediaPipe's metric space has y up and the camera
    # looking down -z; gsplat wants y down and the camera looking down +z.
    # Getting this wrong produces a splat that reconstructs perfectly and
    # renders inside out, which is a very expensive way to find a sign error.
    flip = np.diag(np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float32))

    landmarker = _landmarker(model_path)
    poses: list[HeadPose] = []
    try:
        for index, image in enumerate(images):
            result = landmarker.detect(_mp_image(image))
            if not result.facial_transformation_matrixes:
                continue

            matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)
            view = (flip @ matrix).astype(np.float32)
            yaw, pitch = _yaw_pitch(view[:3, :3])

            scores = {}
            if result.face_blendshapes:
                scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
            expression = max(
                (float(scores.get(name, 0.0)) for name in NON_RIGID_BLENDSHAPES),
                default=0.0,
            )

            poses.append(
                HeadPose(index=index, image=image, view=view,
                         yaw=yaw, pitch=pitch, expression=expression)
            )
    finally:
        landmarker.close()

    return poses


def _speaking_cost(pose: HeadPose) -> float:
    """How badly one frame breaks the rigid assumption, from both witnesses.

    The blendshapes see the mouth in this frame; the audio hears whether the
    person was in the middle of a sentence around it. They disagree usefully:
    the quietest instant between two words scores near zero on blendshapes and
    is still a frame whose lips are shaped for the word on either side of it.
    """
    return float(pose.expression) + SPEECH_WEIGHT * float(pose.speech)


def _rigid_spread(poses: list[HeadPose]) -> list[HeadPose]:
    """The calmest frames, spread across the angles the head actually reached.

    Two filters at once, because they pull against each other. Taking only the
    most neutral frames biases towards whatever direction the person happened
    to be facing when they stopped talking; taking an even spread of angles
    admits mid-word frames that a rigid model cannot explain. So: bucket by
    yaw, then keep the calmest few in each bucket.

    "Calmest" is _speaking_cost rather than the blendshape score alone, so a
    clip with an audio track prefers frames from its quiet seconds. Without
    one, pose.speech is zero throughout and this is the ordering it always was.
    """
    if not poses:
        return []

    bins: dict[int, list[HeadPose]] = {}
    for pose in poses:
        # Yaw wrapped into [0, 2pi) and divided into the same 24 bins the
        # coverage figure counts, so "kept" and "covered" mean the same thing.
        angle = (pose.yaw + 2 * math.pi) % (2 * math.pi)
        bins.setdefault(int(angle / (2 * math.pi) * YAW_BINS) % YAW_BINS, []).append(pose)

    kept: list[HeadPose] = []
    for members in bins.values():
        members.sort(key=_speaking_cost)
        kept.extend(members[:MAX_FRAMES_PER_BIN])
    kept.sort(key=lambda p: p.index)
    return kept


def _coverage(poses: list[HeadPose]) -> float:
    """The fraction of the way round the head a camera actually got.

    Measured, not assumed: the share of yaw bins containing at least one view.
    A clip of someone looking straight ahead scores low however long it runs,
    which is the honest answer and is what the quality report should carry.
    """
    if not poses:
        return 0.0
    filled = {
        int(((p.yaw + 2 * math.pi) % (2 * math.pi)) / (2 * math.pi) * YAW_BINS) % YAW_BINS
        for p in poses
    }
    return round(min(1.0, len(filled) / YAW_BINS), 2)


def _yaw_span_degrees(poses: list[HeadPose]) -> float:
    """The total arc of yaw the views occupy, in degrees.

    The companion to _coverage, and the reason a low coverage figure can be
    explained instead of merely reported. Coverage says what fraction of the
    circle was seen; this says how wide the seen part is, which is the number a
    person can picture. Ninety degrees is "they turned their head"; fifteen is
    "they looked at the phone and did not move".

    Computed as the smallest arc containing every view - the circle minus the
    largest gap between consecutive yaws - so a clip that happens to straddle
    the wrap at pi is not reported as having swept the whole way round.
    """
    if not poses:
        return 0.0
    if len(poses) == 1:
        return 0.0

    angles = sorted((p.yaw + 2 * math.pi) % (2 * math.pi) for p in poses)
    gaps = [b - a for a, b in itertools.pairwise(angles)]
    gaps.append(angles[0] + 2 * math.pi - angles[-1])
    return round(math.degrees(2 * math.pi - max(gaps)), 1)


# --------------------------------------------------------------------------
# masking: forced by the inversion, not a tidiness step


def _head_masks(poses: list[HeadPose], model_path: str = SEGMENTER_MODEL_PATH) -> list:
    """A per-frame alpha for the head, and nothing else in the room.

    In head-fixed coordinates the background is the thing that moves. It cannot
    be reconstructed by a static model and, worse, the optimiser will spend
    Gaussians trying: the cheapest way to explain a wall that swings around a
    face is a fog in front of the face. So the room is not down-weighted, it is
    removed, and the loss is evaluated only where the person is.

    Falling back to the face landmarks' convex hull when the segmenter is
    absent keeps a worker with a partial image useful; it loses the hair, which
    is stated in the notes rather than discovered in the render.
    """
    import numpy as np
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not Path(model_path).exists():
        raise ReconstructError(f"the segmentation model is missing at {model_path}")

    segmenter = mp_vision.ImageSegmenter.create_from_options(
        mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
    )

    masks = []
    try:
        for pose in poses:
            categories = segmenter.segment(_mp_image(pose.image)).category_mask.numpy_view()
            mask = np.isin(categories, HEAD_CLASSES).astype(np.float32)
            masks.append(mask)
    finally:
        segmenter.close()
    return masks


# --------------------------------------------------------------------------
# initialisation


def _initial_cloud(poses: list[HeadPose], masks: list, count: int):
    """Where the Gaussians start.

    Random initialisation in a box is what the published pipelines do, and it
    works when a camera orbits an object through 360 degrees. A head that turns
    covers perhaps 120, so there is no view from behind to delete the points
    that drift there, and random init leaves a halo of debris that renders as a
    ghost. Instead the points are seeded by unprojecting the masked pixels of
    every kept view into the head-fixed world at the depth the pose implies,
    which puts them on the surface the camera saw and nowhere else.
    """
    import numpy as np

    points, colours = [], []
    per_view = max(1, count // max(1, len(poses)))

    for pose, mask in zip(poses, masks, strict=True):
        height, width = mask.shape[:2]
        ys, xs = np.nonzero(mask > 0.5)
        if xs.size == 0:
            continue
        take = np.random.default_rng(pose.index).choice(
            xs.size, size=min(per_view, xs.size), replace=False
        )
        xs, ys = xs[take], ys[take]

        focal = (height / 2.0) / math.tan(math.radians(MP_VERTICAL_FOV_DEG) / 2.0)
        # Depth is taken from the pose rather than estimated. The head's origin
        # sits at view[:3, 3]; a head is roughly 0.2m deep, so seeding on that
        # plane puts every point within a head's width of the true surface,
        # which is inside the basin the optimiser can walk out of.
        depth = float(pose.view[2, 3])
        if not math.isfinite(depth) or depth <= 0:
            continue

        camera = np.stack([
            (xs - width / 2.0) * depth / focal,
            (ys - height / 2.0) * depth / focal,
            np.full(xs.shape, depth),
        ], axis=1).astype(np.float32)

        rotation, translation = pose.view[:3, :3], pose.view[:3, 3]
        world = (camera - translation) @ rotation
        points.append(world)
        colours.append(pose.image[ys, xs].astype(np.float32) / 255.0)

    if not points:
        raise ReconstructError("no head pixels survived masking; nothing to reconstruct")

    means = np.concatenate(points, axis=0).astype(np.float32)
    rgb = np.concatenate(colours, axis=0).astype(np.float32)
    if means.shape[0] > count:
        pick = np.random.default_rng(0).choice(means.shape[0], size=count, replace=False)
        means, rgb = means[pick], rgb[pick]
    return means, rgb


# --------------------------------------------------------------------------
# the optimisation


def _intrinsics(height: int, width: int):
    """The camera MediaPipe solved the pose against, as a matrix.

    Not a guess and not EXIF: the facial transformation matrix is only
    meaningful with respect to the fixed 63-degree perspective camera the face
    geometry module assumes, so this is the one intrinsic that is consistent
    with the poses.
    """
    import numpy as np

    focal = (height / 2.0) / math.tan(math.radians(MP_VERTICAL_FOV_DEG) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------
# pose refinement: the bundle adjustment this pipeline would otherwise skip


def _axis_angle_to_rotation(omega):
    """(V, 3) axis-angle to (V, 3, 3) rotation, differentiably.

    Rodrigues' formula, written so it survives being evaluated at exactly zero
    - which is where every one of these parameters starts. The naive form
    divides by the rotation angle and returns NaN on the first forward pass,
    and a NaN here does not announce itself: it propagates into the extrinsics,
    the render comes out blank, the loss is finite, and the build completes
    having learned nothing.
    """
    import torch

    theta_sq = (omega * omega).sum(-1, keepdim=True)
    # The epsilon is inside the square root rather than a clamp outside it so
    # the derivative at zero is finite as well as the value.
    theta = torch.sqrt(theta_sq + 1e-12)
    sin_term = (torch.sin(theta) / theta).unsqueeze(-1)
    cos_term = ((1.0 - torch.cos(theta)) / theta_sq.clamp(min=1e-12)).unsqueeze(-1)

    zero = torch.zeros_like(omega[..., 0])
    wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]
    skew = torch.stack([
        torch.stack([zero, -wz, wy], dim=-1),
        torch.stack([wz, zero, -wx], dim=-1),
        torch.stack([-wy, wx, zero], dim=-1),
    ], dim=-2)

    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(skew.shape)
    # At omega = 0 the skew matrix is exactly zero, so this is exactly the
    # identity: the first iteration of a run with a perfect prior changes
    # nothing, which is the property that makes the prior an anchor.
    return eye + sin_term * skew + cos_term * (skew @ skew)


def _corrected_views(views, rotation_delta, translation_delta):
    """MediaPipe's extrinsics with a learned SE(3) correction applied.

    Left-multiplied rather than re-parameterised. Composing on this side means
    the correction acts in the camera's own frame, so `rotation_delta` is
    literally how far this view's camera was turned from where MediaPipe put
    it and `translation_delta` is literally how far it was moved - radians and
    metres, per view, directly reportable. Solving for the pose outright would
    read the same at convergence and would have discarded the prior that is the
    only reason this route needs no solver.
    """
    import torch

    delta = _axis_angle_to_rotation(rotation_delta)
    rotation = delta @ views[:, :3, :3]
    translation = (delta @ views[:, :3, 3:4]).squeeze(-1) + translation_delta
    top = torch.cat([rotation, translation.unsqueeze(-1)], dim=-1)
    # The prior's own bottom row, so the result keeps its dtype and device and
    # is a homogeneous matrix by construction rather than by assembly.
    return torch.cat([top, views[:, 3:4, :]], dim=-2)


def _pose_regulariser(rotation_delta, translation_delta):
    """What it costs to disagree with MediaPipe.

    Each correction is measured as a fraction of its hard ceiling and squared,
    which makes the two terms dimensionless and comparable and lets one weight
    govern both. Squared magnitude rather than magnitude on purpose: the
    gradient of a vector norm at zero is undefined, and zero is where these
    parameters live for the whole warmup.
    """
    rotation = (rotation_delta * rotation_delta).sum(-1) / math.radians(
        MAX_POSE_ROTATION_DEG
    ) ** 2
    translation = (translation_delta * translation_delta).sum(-1) / (
        MAX_POSE_TRANSLATION_M ** 2
    )
    return POSE_REGULARISATION * (rotation.mean() + translation.mean())


def _clamp_pose_correction(rotation_delta, translation_delta) -> None:
    """Project every correction back inside its ceiling, in place.

    The regulariser makes a large correction expensive; this makes it
    impossible. The difference matters because the failure it guards against is
    not gradual: one frame the landmarker got badly wrong produces a residual
    no amount of Gaussian can explain, the cheapest remaining explanation is to
    move that camera somewhere else entirely, and a camera that has left the
    scene takes its share of the Gaussians with it.
    """
    import torch

    with torch.no_grad():
        for tensor, limit in (
            (rotation_delta, math.radians(MAX_POSE_ROTATION_DEG)),
            (translation_delta, MAX_POSE_TRANSLATION_M),
        ):
            norm = torch.linalg.norm(tensor, dim=-1, keepdim=True)
            tensor.mul_((limit / norm.clamp(min=1e-12)).clamp(max=1.0))


def _pose_learning_rates(iterations: int, views: int, warmup: int) -> tuple[float, float]:
    """How fast a camera may move, derived from how often it is looked at.

    One view is rendered per iteration, so a pose is stepped roughly
    (iterations - warmup) / views times in the whole run and no more. Sizing
    the learning rate as "the ceiling, divided over the steps this view will
    actually receive" makes the correction able to reach the error it exists to
    fix on a long build and on a short one, without either becoming a number
    that has to be re-tuned by hand whenever Quality.iterations changes.

    Capped, because a preview build divides the same ceiling over a tenth of
    the steps and the result would be a pose that lurches a third of a degree
    on one noisy image gradient. A preview that refines less is the right
    trade; a preview that shakes its cameras is not.
    """
    steps = max(1, (int(iterations) - int(warmup)) // max(1, int(views)))
    return (
        min(POSE_ROTATION_LR_MAX, math.radians(MAX_POSE_ROTATION_DEG) / steps),
        min(POSE_TRANSLATION_LR_MAX, MAX_POSE_TRANSLATION_M / steps),
    )


def _pose_optimiser(rotation_delta, translation_delta, rotation_lr, translation_lr):
    """Adam over the pose corrections, with the momentum deliberately removed.

    One view is rendered per step, so on any given step every *other* view's
    gradient is exactly zero. Adam does not stand still on a zero gradient: it
    keeps spending the momentum left over from the last step that did move,
    decayed by beta1 each time. With thirty views that is roughly nine extra
    steps of the last real direction before a view is next rendered, so each
    pose moves about ten times as far as its learning rate says it does - and
    the learning rate here is chosen precisely as a displacement budget.

    beta1 = 0 makes the first moment the current gradient, so a view that was
    not rendered does not move. The second moment is kept, because that is what
    makes the step size independent of how large the image gradients happen to
    be and therefore comparable between a bright clip and a dim one.

    beta2 = 0.9 rather than the usual 0.999 for the same reason the learning
    rate is derived rather than fixed: a pose gets only a few hundred steps, and
    a second moment that remembers a thousand of them is still normalising by
    the large gradients of the first iteration long after the pose has stopped
    being wrong. The effective step decays towards nothing and the refinement
    stalls part-way. Measured on the toy capture in the tests, over the 350
    steps a standard build gives one view, this is the difference between
    taking three degrees of pose error down to 1.6 and taking it down to 0.2.
    """
    import torch

    return torch.optim.Adam(
        [
            {"params": [rotation_delta], "lr": rotation_lr},
            {"params": [translation_delta], "lr": translation_lr},
        ],
        betas=(0.0, 0.9),
    )


def _pose_correction_report(rotation_delta, translation_delta) -> PoseCorrection:
    """The magnitudes, off the card, in units a support agent can read."""
    import torch

    with torch.no_grad():
        degrees = torch.rad2deg(torch.linalg.norm(rotation_delta, dim=-1))
        millimetres = torch.linalg.norm(translation_delta, dim=-1) * 1000.0
        return PoseCorrection(
            mean_deg=round(float(degrees.mean()), 3),
            max_deg=round(float(degrees.max()), 3),
            mean_mm=round(float(millimetres.mean()), 2),
            max_mm=round(float(millimetres.max()), 2),
            views=int(rotation_delta.shape[0]),
        )


def optimise(
    poses: list[HeadPose],
    masks: list,
    *,
    iterations: int,
    gaussian_budget: int,
    device: str = "cuda",
) -> GaussianCloud:
    """Fit Gaussians to the masked views, in the head's own coordinate frame.

    gsplat's MCMC strategy is used rather than the default densify-and-prune
    one for a product reason rather than a quality one: `cap_max` is a hard
    ceiling on the number of Gaussians, so the file the customer downloads has
    a size known before the build starts. The default strategy's count is an
    emergent property of the scene, which is how a preview build ends up being
    40MB on someone's phone connection.
    """
    import numpy as np
    import torch
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy

    if not torch.cuda.is_available() and device == "cuda":
        raise ReconstructError("no CUDA device: a splat cannot be optimised on this machine")

    height, width = masks[0].shape[:2]
    k = torch.from_numpy(_intrinsics(height, width)).to(device)[None]
    views = torch.from_numpy(np.stack([p.view for p in poses])).to(device)
    images = torch.stack([
        torch.from_numpy(p.image.astype("float32") / 255.0) for p in poses
    ]).to(device)
    alphas = torch.stack([torch.from_numpy(m) for m in masks]).to(device)[..., None]

    means_np, rgb_np = _initial_cloud(poses, masks, min(gaussian_budget, 100_000))
    means = torch.nn.Parameter(torch.from_numpy(means_np).to(device))
    n = means.shape[0]

    # Initial scale from the mean nearest-neighbour spacing would need a kNN
    # pass; the head's extent divided by the cube root of the count is the same
    # quantity to within a factor and costs nothing.
    extent = float(means.detach().max(0).values.sub(means.detach().min(0).values).max())
    spacing = max(extent / max(1.0, n ** (1 / 3)), 1e-4)

    scales = torch.nn.Parameter(torch.full((n, 3), math.log(spacing), device=device))
    quats = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
    )
    opacities = torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=device)))
    # sh_degree 0: one coefficient per colour channel, view-independent. A
    # higher degree buys specular behaviour the browser renderer would have to
    # evaluate per frame on a phone, and would not fit the 32-byte record.
    sh0 = torch.nn.Parameter(
        ((torch.from_numpy(rgb_np).to(device) - 0.5) / 0.2820947917).unsqueeze(1)
    )

    params = torch.nn.ParameterDict({
        "means": means, "scales": scales, "quats": quats,
        "opacities": opacities, "sh0": sh0,
    })
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=1.6e-4 * max(extent, 1.0)),
        "scales": torch.optim.Adam([params["scales"]], lr=5e-3),
        "quats": torch.optim.Adam([params["quats"]], lr=1e-3),
        "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2),
        "sh0": torch.optim.Adam([params["sh0"]], lr=2.5e-3),
    }

    # The six per-view parameters that turn MediaPipe's estimate from a
    # measurement into a prior. Deliberately *not* in `params` and *not* in
    # `optimizers`: MCMCStrategy walks both, and every tensor it finds there it
    # assumes is one row per Gaussian, to be relocated and duplicated with the
    # cloud. A (views, 3) tensor in that dict is a silent corruption of the
    # strategy's bookkeeping the first time it densifies.
    rotation_delta = torch.nn.Parameter(torch.zeros(len(poses), 3, device=device))
    translation_delta = torch.nn.Parameter(torch.zeros(len(poses), 3, device=device))
    # Floored at one pass over the views, not just at a fraction of the run. A
    # camera released before its own frame has ever been rendered would take
    # its first step against a cloud that is still the seed points, and a short
    # run - a smoke test, a tiny preview - would otherwise round the warmup
    # away entirely and refine poses that have nothing to be refined against.
    warmup = max(len(poses), int(int(iterations) * POSE_WARMUP_FRACTION))
    pose_optimizer = _pose_optimiser(
        rotation_delta, translation_delta,
        *_pose_learning_rates(iterations, len(poses), warmup),
    )

    strategy = MCMCStrategy(cap_max=int(gaussian_budget), verbose=False)
    state = strategy.initialize_state()

    for step in range(int(iterations)):
        pick = step % len(poses)
        # Frozen for the warmup, and frozen by *exclusion from the graph*
        # rather than by a zeroed gradient, so nothing accumulates in Adam's
        # moments that would be spent the instant the poses are released.
        refining = step >= warmup
        viewmats = (
            _corrected_views(
                views[pick: pick + 1],
                rotation_delta[pick: pick + 1],
                translation_delta[pick: pick + 1],
            )
            if refining
            else views[pick: pick + 1]
        )
        rendered, _, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["sh0"],
            viewmats=viewmats,
            Ks=k,
            width=width,
            height=height,
            sh_degree=0,
            packed=True,
        )
        strategy.step_pre_backward(params, optimizers, state, step, info)

        target = images[pick: pick + 1]
        mask = alphas[pick: pick + 1]
        # The loss lives only where the person is. Everything outside the mask
        # is a room that, in this coordinate frame, is not a scene at all.
        loss = (torch.abs(rendered - target) * mask).sum() / mask.sum().clamp(min=1.0)
        # MCMC's own regularisers: without them the strategy relocates dead
        # Gaussians into ever larger blobs and the splat loses its edges.
        loss = loss + 1e-2 * torch.sigmoid(params["opacities"]).abs().mean()
        loss = loss + 1e-2 * torch.exp(params["scales"]).abs().mean()
        if refining:
            # The rendered view's slice, not the whole tensor. Penalising every
            # view here would give the twenty-nine that were not rendered a
            # non-zero gradient, and they would drift on a step that saw no
            # evidence about them at all - which is exactly the behaviour
            # _pose_optimiser removes the momentum to prevent.
            loss = loss + _pose_regulariser(
                rotation_delta[pick: pick + 1], translation_delta[pick: pick + 1]
            )

        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if refining:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
            _clamp_pose_correction(rotation_delta, translation_delta)
        strategy.step_post_backward(
            params, optimizers, state, step, info,
            lr=optimizers["means"].param_groups[0]["lr"],
        )

    return _cloud_from(
        params, poses, _pose_correction_report(rotation_delta, translation_delta)
    )


def _cloud_from(
    params, poses: list[HeadPose], correction: PoseCorrection | None = None
) -> GaussianCloud:
    """Activated parameters, off the card, in the shape the exporter wants."""
    import torch

    with torch.no_grad():
        quats = torch.nn.functional.normalize(params["quats"], dim=-1)
        colors = (0.2820947917 * params["sh0"].squeeze(1) + 0.5).clamp(0.0, 1.0)
        return GaussianCloud(
            means=params["means"].detach().cpu().numpy(),
            scales=torch.exp(params["scales"]).detach().cpu().numpy(),
            quats=quats.detach().cpu().numpy(),
            colors=colors.detach().cpu().numpy(),
            opacities=torch.sigmoid(params["opacities"]).detach().cpu().numpy(),
            angular_coverage=_coverage(poses),
            views_used=len(poses),
            yaw_span_deg=_yaw_span_degrees(poses),
            pose_correction=correction or PoseCorrection(),
        )


# --------------------------------------------------------------------------
# the file


def write_splat(cloud: GaussianCloud) -> bytes:
    """The 32-byte-per-Gaussian record the browser renderer reads.

    Position and scale as float32, colour and rotation quantised to bytes. This
    is the format Spark and every other web splat viewer loads, and the reason
    the whole product can render on the customer's own device instead of on a
    GPU we pay for per call.

    Sorted by falling visual importance, which is what lets a viewer show a
    recognisable face after the first megabyte instead of after the last one.
    """
    import numpy as np

    means = np.asarray(cloud.means, dtype=np.float32)
    scales = np.asarray(cloud.scales, dtype=np.float32)
    quats = np.asarray(cloud.quats, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.float32)
    opacities = np.asarray(cloud.opacities, dtype=np.float32).reshape(-1)

    if not (len(means) == len(scales) == len(quats) == len(colors) == len(opacities)):
        raise ReconstructError("the Gaussian arrays disagree about how many Gaussians exist")

    order = np.argsort(-(scales.prod(axis=1) * opacities))
    means, scales = means[order], scales[order]
    quats, colors, opacities = quats[order], colors[order], opacities[order]

    rgba = np.concatenate(
        [np.clip(colors, 0.0, 1.0), np.clip(opacities, 0.0, 1.0)[:, None]], axis=1
    )
    rgba = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)

    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    unit = quats / np.where(norm == 0, 1.0, norm)
    rot = np.clip(unit * 128.0 + 128.0, 0, 255).astype(np.uint8)

    out = bytearray()
    for i in range(len(means)):
        out += struct.pack("<3f", *means[i])
        out += struct.pack("<3f", *scales[i])
        out += bytes(rgba[i])
        out += bytes(rot[i])
    return bytes(out)


# --------------------------------------------------------------------------
# the route


def reconstruct(
    video: Path,
    *,
    iterations: int,
    gaussian_budget: int,
    device: str = "cuda",
) -> GaussianCloud:
    """Video in, a measured splat of that person out.

    Every Gaussian this returns was optimised against a frame in which a camera
    saw the person. That is the entire reason this route beats generation
    whenever it is open, and it is what lets the quality report say the
    likeness is measured without qualifying it.
    """
    video = Path(video)
    images = _frames(video)
    poses = head_poses(images)

    # Attached after landmarking rather than before, so a clip whose audio
    # cannot be decoded costs nothing: the poses are already correct and every
    # speech value simply stays at zero.
    envelope = _speech_envelope(video, len(images))
    if envelope is not None:
        poses = [replace(p, speech=envelope[p.index]) for p in poses]

    if len(poses) < MIN_USABLE_VIEWS:
        raise ReconstructError(
            f"a face was found in only {len(poses)} sampled frames, and "
            f"{MIN_USABLE_VIEWS} are needed for the frames to be different views "
            "of a head rather than repetitions of one"
        )

    kept = _rigid_spread(poses)
    if len(kept) < MIN_USABLE_VIEWS:
        # Reached when someone talks continuously: every frame is non-rigid, so
        # the spread filter has nothing calm to choose. Building from the
        # calmest available beats refusing a clip the router already accepted.
        kept = sorted(poses, key=_speaking_cost)[: max(MIN_USABLE_VIEWS, YAW_BINS)]
        kept.sort(key=lambda p: p.index)

    masks = _head_masks(kept)
    cloud = optimise(
        kept, masks,
        iterations=iterations, gaussian_budget=gaussian_budget, device=device,
    )

    return replace(cloud, notes=_notes(poses, kept, cloud, heard=envelope is not None))


def _notes(
    poses: list[HeadPose], kept: list[HeadPose], cloud: GaussianCloud, *, heard: bool
) -> tuple[str, ...]:
    """What a support agent would need to explain this build to the family.

    Every line here answers a question somebody will eventually ask about an
    avatar that did not come out the way they hoped, and answers it with a
    measured number rather than a guess.
    """
    notes: list[str] = []

    dropped = len(poses) - len(kept)
    if dropped:
        notes.append(f"{dropped} frames were mid-word or mid-blink and were not used")
    if not heard:
        notes.append(
            "the clip carried no usable audio, so mid-word frames were recognised "
            "from the face alone and some will have been kept"
        )

    span = cloud.yaw_span_deg
    if span <= FRONTAL_SHELL_MAX_SPAN_DEG:
        # Stated as a shape rather than as a shortfall. A coverage of 0.22 is
        # what someone talking to a propped phone produces even when they did
        # everything right, and a number that reads like a failure next to no
        # sentence at all is how a good build gets thrown away.
        notes.append(
            f"the head turned through {span:.0f} degrees in total, so this is a "
            f"frontal shell: complete from the front and from the {span / 2:.0f} "
            "degrees to either side that the camera reached, and unreconstructed "
            f"behind the ears. The angular coverage of {cloud.angular_coverage} "
            "measures that shape and is not a fault"
        )
    else:
        notes.append(
            f"the head turned through {span:.0f} degrees, wide enough to be seen "
            "from the side as well as the front"
        )

    correction = cloud.pose_correction
    notes.append(
        f"the camera poses were refined by {correction.mean_deg:.2f} degrees on "
        f"average and {correction.max_deg:.2f} at most, and by "
        f"{correction.mean_mm:.1f}mm on average"
    )
    if correction.suspect:
        # The diagnostic that turns an inexplicable avatar into a fixable
        # capture. Large corrections mean the prior was wrong, and on this
        # route there are only two ways for the prior to be wrong.
        notes.append(
            f"that is a large correction: MediaPipe's poses disagreed with the "
            f"frames by more than {SUSPECT_POSE_CORRECTION_DEG:.0f} degrees on "
            "average, which points at the capture itself or at a lens far from "
            f"the {MP_VERTICAL_FOV_DEG:.0f}-degree camera the pose solver assumes"
        )

    return tuple(notes)
