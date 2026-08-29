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

import math
import os
import struct
from dataclasses import dataclass, field
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


def _rigid_spread(poses: list[HeadPose]) -> list[HeadPose]:
    """The calmest frames, spread across the angles the head actually reached.

    Two filters at once, because they pull against each other. Taking only the
    most neutral frames biases towards whatever direction the person happened
    to be facing when they stopped talking; taking an even spread of angles
    admits mid-word frames that a rigid model cannot explain. So: bucket by
    yaw, then keep the calmest few in each bucket.
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
        members.sort(key=lambda p: p.expression)
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

    strategy = MCMCStrategy(cap_max=int(gaussian_budget), verbose=False)
    state = strategy.initialize_state()

    for step in range(int(iterations)):
        pick = step % len(poses)
        rendered, _, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["sh0"],
            viewmats=views[pick: pick + 1],
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

        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        strategy.step_post_backward(
            params, optimizers, state, step, info,
            lr=optimizers["means"].param_groups[0]["lr"],
        )

    return _cloud_from(params, poses)


def _cloud_from(params, poses: list[HeadPose]) -> GaussianCloud:
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
    images = _frames(Path(video))
    poses = head_poses(images)
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
        kept = sorted(poses, key=lambda p: p.expression)[: max(MIN_USABLE_VIEWS, YAW_BINS)]
        kept.sort(key=lambda p: p.index)

    masks = _head_masks(kept)
    cloud = optimise(
        kept, masks,
        iterations=iterations, gaussian_budget=gaussian_budget, device=device,
    )

    notes = []
    dropped = len(poses) - len(kept)
    if dropped:
        notes.append(f"{dropped} frames were mid-word or mid-blink and were not used")
    if cloud.angular_coverage < 0.5:
        notes.append("the head turned through less than half of the way round")

    return GaussianCloud(
        means=cloud.means, scales=cloud.scales, quats=cloud.quats,
        colors=cloud.colors, opacities=cloud.opacities,
        angular_coverage=cloud.angular_coverage,
        views_used=cloud.views_used,
        notes=tuple(notes),
    )
