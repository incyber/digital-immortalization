"""The generated route: an album, and a splat that is honest about being partly invented.

WHICH GENERATOR, AND WHY
========================
TRELLIS (Microsoft, MIT) - specifically TRELLIS-image-large, whose weights are
also MIT on the model hub.

The three MIT candidates were compared on what they emit rather than on their
demo reels, because the emitted representation decides how much of the pipeline
downstream has to be rebuilt:

  TRELLIS   emits 3D Gaussians directly from its structured-latent decoder
            (`outputs['gaussian']`), at a density that reaches our STANDARD
            budget, and it is the only one of the three whose native output is
            the artefact this product ships. Chosen.
  LGM       also emits Gaussians, but a fixed ~65k of them, synthesised from
            four generated views. 65k is below even our PREVIEW budget of
            200k, so a likeness would be soft everywhere and there is no knob
            that fixes it - the count is the architecture.
  TripoSR   emits a triplane NeRF. Converting that to Gaussians means baking a
            radiance field into blobs, which loses exactly the high-frequency
            skin detail that decides whether a family recognises someone. It
            is the fastest of the three and the wrong output.

WHAT THE OTHER PHOTOGRAPHS ARE FOR
==================================
TRELLIS sees one image. Everything else it produces is a prior over faces, not
this face. The remaining photographs are therefore used to pull the generated
surface back towards the person wherever a camera actually looked - and *only*
appearance is corrected, never geometry.

That restriction is the same one avatar/splat/routes.py refuses to build
around: photographs taken years apart have no single 3D truth, so optimising
positions across them would fit a person who never existed at any age. Colour
and opacity, though, are per-observation and safe: where a photograph saw the
cheek, the cheek gets that skin.

LICENCE, AND ONE TRAP THAT IS EASY TO WALK INTO
===============================================
TRELLIS's own code and weights are MIT. Its *optional* setup flags are not:

  --mipgaussian     installs diff-gaussian-rasterization from mip-splatting,
                    which is the Inria rasteriser under the Gaussian-Splatting
                    License: research and evaluation only.
  --diffoctreerast  is explicitly a derivative work of the same Inria software
                    and carries the same restriction.
  --nvdiffrast      is the NVIDIA Source Code License, non-commercial.

None of the three is installed in the image, and this module never needs them:
`trellis.renderers` resolves its submodules lazily through __getattr__, so the
restricted rasterisers are only imported if a preview render is requested, and
we do not request one. We ask the pipeline for `formats=["gaussian"]` and
nothing else - which also skips mesh extraction and its GPL-licensed repair
tools - and we rasterise with gsplat (Apache-2.0) when we need pixels.

`formats=["gaussian"]` is therefore a licence boundary, not an optimisation.
Do not add "mesh" to it.

As in reconstruct.py, nothing heavy is imported at module scope.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from reconstruct import GaussianCloud, head_poses

# Where the TRELLIS weights are baked into the image.
TRELLIS_MODEL_DIR = os.environ.get("TRELLIS_MODEL_DIR", "/models/TRELLIS-image-large")

# Square edge every photograph is resized to for the appearance pass. TRELLIS
# reconstructs from a 518px conditioning image, so correcting at 512 matches
# the detail the geometry can actually carry.
CORRECTION_EDGE = 512

# The camera TRELLIS's own canonical frame is defined against: the object sits
# in a unit cube at the origin and the conditioning view looks at it from this
# distance through this field of view. Taken from the project's render_utils
# defaults, because those are the cameras its outputs are consistent with.
TRELLIS_FOV_DEG = 40.0
TRELLIS_RADIUS = 2.0

# Radii tried when fitting the anchor. The unit-cube convention fixes the
# object's size but not how much of the frame a cropped photograph fills, and
# that single scalar is the dominant misalignment between a generated splat and
# a photograph of the real person. Four renders is a cheap way to remove it.
RADIUS_CANDIDATES = (1.6, 1.8, 2.0, 2.3)

# Steps of appearance correction. Short on purpose: this is not a
# reconstruction, it is a recolouring of a fixed geometry against a handful of
# views, and running it long enough to matter would be running it long enough
# to start explaining pose error as skin tone.
CORRECTION_STEPS = 600

# Seeded so the same album rebuilds to the same splat. A family who asks why
# the second build looks different should not be told that it is a diffusion
# model and these things happen.
SEED = 20260829


class GenerateError(RuntimeError):
    """A generation that cannot be completed, with a reason worth reading."""


# --------------------------------------------------------------------------
# TRELLIS


def _pipeline(device: str = "cuda"):
    """Load the image-to-3D pipeline, or say precisely what is missing."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise GenerateError("no CUDA device: TRELLIS cannot run on this machine")
    if not Path(TRELLIS_MODEL_DIR).exists():
        raise GenerateError(f"the TRELLIS weights are missing at {TRELLIS_MODEL_DIR}")

    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(TRELLIS_MODEL_DIR)
    pipeline.to(device)
    return pipeline


def _trellis_gaussians(pipeline, anchor: Path):
    """One photograph in, 3D Gaussians out, with no restricted code touched.

    `formats=["gaussian"]` is the licence boundary described in the module
    docstring: asking for a mesh would import FlexiCubes and the mesh
    post-processing chain, and asking for a preview render would import the
    Inria rasteriser. Neither is needed to obtain the Gaussians themselves.
    """
    from PIL import Image

    image = Image.open(anchor).convert("RGB")
    # The pipeline's own preprocessing: background removal and a centred crop
    # of the subject. Used rather than ours so the conditioning image is
    # framed exactly as the model was trained to expect.
    image = pipeline.preprocess_image(image)

    outputs = pipeline.run(image, seed=SEED, formats=["gaussian"])
    gaussians = outputs.get("gaussian") or []
    if not gaussians:
        raise GenerateError("TRELLIS returned no Gaussians for the anchor photograph")
    return gaussians[0]


def _cloud_from_trellis(gaussian) -> tuple:
    """TRELLIS's activated parameters as plain tensors, in our conventions."""
    import torch

    with torch.no_grad():
        means = gaussian.get_xyz.detach()
        scales = gaussian.get_scaling.detach()
        quats = torch.nn.functional.normalize(gaussian.get_rotation.detach(), dim=-1)
        opacities = gaussian.get_opacity.detach().reshape(-1)
        # sh degree 0, so the features are one coefficient per channel and the
        # colour is the same constant every viewer uses to decode it.
        features = gaussian.get_features.detach()
        colors = (0.2820947917 * features.reshape(features.shape[0], -1)[:, :3] + 0.5)
    return means, scales, quats, opacities.clamp(0.0, 1.0), colors.clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# where each photograph was standing


def _square(image, edge: int = CORRECTION_EDGE):
    """Centre-crop to a square and resize, so one intrinsic covers every photo."""
    import cv2

    height, width = image.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    return cv2.resize(
        image[top: top + side, left: left + side], (edge, edge), interpolation=cv2.INTER_AREA
    )


def _relative_views(anchor_pose, poses, radius: float):
    """Cameras for the photographs, in TRELLIS's frame, from head pose alone.

    The same inversion reconstruct.py is built on, used for a different
    purpose. A photograph in which the head is turned 30 degrees further than
    the anchor is, to a fixed object, a camera moved 30 degrees around it. So:

        Q = R_photo @ R_anchor^T          the head's rotation change, in
                                          camera coordinates
        world-to-camera = [Q | (0, 0, r)] the object rotated by Q, viewed from
                                          distance r

    Only the *relative* rotation is used, which is what makes this sound: the
    absolute alignment between MediaPipe's canonical head and TRELLIS's
    canonical cube is unknown and cancels, because both are anchored on the
    same photograph.
    """
    import numpy as np

    r_anchor = np.asarray(anchor_pose.view, dtype=np.float32)[:3, :3]
    views = []
    for pose in poses:
        q = np.asarray(pose.view, dtype=np.float32)[:3, :3] @ r_anchor.T
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = q
        view[:3, 3] = np.array([0.0, 0.0, radius], dtype=np.float32)
        views.append(view)
    return np.stack(views) if views else np.zeros((0, 4, 4), dtype=np.float32)


def _intrinsics(edge: int, fov_deg: float = TRELLIS_FOV_DEG):
    import numpy as np

    focal = (edge / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array(
        [[focal, 0.0, edge / 2.0], [0.0, focal, edge / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _render(params, view, k, edge: int):
    """One view of the fixed geometry, through gsplat rather than any Inria code.

    The alpha channel comes back with the colours because the correction needs
    it: a generated splat has no background, so the region it does not cover
    renders black against a photograph of somebody's living room. Comparing
    those pixels would teach the optimiser that this person's skin is the
    colour of a dark wall.
    """
    from gsplat import rasterization

    rendered, alphas, _ = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"],
        opacities=params["opacities"],
        colors=params["colors"],
        viewmats=view,
        Ks=k,
        width=edge,
        height=edge,
    )
    return rendered, alphas


def _masked_error(rendered, alphas, target):
    """Photometric error where the splat actually is, and nowhere else.

    The mask is detached on purpose. Opacity is one of the two things being
    optimised here, so a mask that carried gradient would offer the optimiser a
    cheaper way to reduce the loss than getting the colour right: fade out.
    """
    import torch

    mask = (alphas.detach() > 0.5).to(rendered.dtype)
    return (torch.abs(rendered - target) * mask).sum() / mask.sum().clamp(min=1.0)


# --------------------------------------------------------------------------
# appearance correction


def correct_appearance(
    tensors: tuple,
    anchor_pose,
    photo_poses: list,
    *,
    device: str = "cuda",
    steps: int = CORRECTION_STEPS,
) -> tuple:
    """Pull the generated surface's colour towards the person, where they were seen.

    Geometry - positions, scales, rotations - is frozen throughout, and that is
    the whole design. Photographs from different years disagree about the shape
    of a face, and optimising shape across them produces an average of several
    ages: a plausible stranger. Colour and opacity are per-observation and can
    be corrected without inventing a person who never existed.

    A photograph whose head pose cannot be estimated is skipped rather than
    guessed at. The count of what was actually used is returned, because "we
    corrected against twelve of your photographs" is a true sentence and "we
    used your photographs" is a vaguer one.
    """
    import numpy as np
    import torch

    means, scales, quats, opacities, colors = tensors
    params = {
        "means": means.to(device),
        "scales": scales.to(device),
        "quats": quats.to(device),
        "opacities": torch.nn.Parameter(
            torch.logit(opacities.to(device).clamp(1e-4, 1 - 1e-4))
        ),
        "colors": torch.nn.Parameter(torch.logit(colors.to(device).clamp(1e-4, 1 - 1e-4))),
    }

    def activated() -> dict:
        return {
            "means": params["means"],
            "scales": params["scales"],
            "quats": params["quats"],
            "opacities": torch.sigmoid(params["opacities"]),
            "colors": torch.sigmoid(params["colors"]),
        }

    k = torch.from_numpy(_intrinsics(CORRECTION_EDGE)).to(device)[None]
    anchor_image = torch.from_numpy(
        _square(anchor_pose.image).astype("float32") / 255.0
    ).to(device)[None]

    # Fit the one scalar the canonical-cube convention leaves free. Everything
    # else about the alignment is fixed by the anchor; the framing of a crop is
    # not, and an unfitted radius shows up as skin tone bleeding onto the
    # background rather than as an obviously wrong picture.
    with torch.no_grad():
        anchor_view = torch.from_numpy(
            _relative_views(anchor_pose, [anchor_pose], TRELLIS_RADIUS)
        ).to(device)
        best_radius, best_error = TRELLIS_RADIUS, None
        for radius in RADIUS_CANDIDATES:
            anchor_view[0, 2, 3] = radius
            rendered, alphas = _render(activated(), anchor_view, k, CORRECTION_EDGE)
            error = _masked_error(rendered, alphas, anchor_image)
            if best_error is None or error < best_error:
                best_radius, best_error = radius, error

    usable = [p for p in photo_poses if p is not None]
    if not usable:
        return activated(), 0, best_radius

    views = torch.from_numpy(
        _relative_views(anchor_pose, usable, best_radius)
    ).to(device)
    targets = torch.stack([
        torch.from_numpy(_square(p.image).astype("float32") / 255.0) for p in usable
    ]).to(device)

    optimizer = torch.optim.Adam(
        [params["colors"], params["opacities"]], lr=1e-2
    )
    rng = np.random.default_rng(SEED)
    for _ in range(int(steps)):
        pick = int(rng.integers(0, len(usable)))
        rendered, alphas = _render(activated(), views[pick: pick + 1], k, CORRECTION_EDGE)
        loss = _masked_error(rendered, alphas, targets[pick: pick + 1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        return activated(), len(usable), best_radius


# --------------------------------------------------------------------------
# the route


def _cap(state: dict, budget: int) -> dict:
    """Trim to the download the customer was promised, least visible first."""
    import torch

    count = state["means"].shape[0]
    if count <= budget:
        return state
    importance = state["scales"].prod(dim=1) * state["opacities"].reshape(-1)
    keep = torch.topk(importance, k=int(budget)).indices
    return {name: value[keep] for name, value in state.items()}


def generate(
    anchor: Path,
    photos: list[Path],
    *,
    gaussian_budget: int,
    device: str = "cuda",
    steps: int = CORRECTION_STEPS,
) -> GaussianCloud:
    """Photographs in, a splat that looks right and is partly invented out.

    The coverage returned is 1.0 and that is not a compliment: a generated
    splat renders from every direction *by construction*, because the model
    invented the directions nobody photographed. What fraction of it is
    measured is not decided here and cannot be - avatar/splat/build.py derives
    it from the route, precisely so that no worker can report an album as if it
    were a video.
    """
    import cv2

    anchor = Path(anchor)
    if not anchor.exists():
        raise GenerateError("the anchor photograph was not downloaded")

    pipeline = _pipeline(device)
    tensors = _cloud_from_trellis(_trellis_gaussians(pipeline, anchor))

    images, sources = [], [anchor, *[p for p in photos if Path(p) != anchor]]
    for path in sources:
        image = cv2.imread(str(path))
        if image is not None:
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    poses = head_poses(images)
    anchor_pose = next((p for p in poses if p.index == 0), None)
    if anchor_pose is None:
        raise GenerateError(
            "no face could be found in the anchor photograph, so the other "
            "photographs cannot be placed relative to it"
        )

    state, corrected, _ = correct_appearance(
        tensors, anchor_pose, [p for p in poses if p.index != 0],
        device=device, steps=steps,
    )
    state = _cap(state, int(gaussian_budget))

    notes = [
        (
            f"{corrected} of the {len(sources) - 1} other photographs corrected the "
            "generated surface; the rest could not be placed"
        )
    ]
    if corrected == 0:
        notes.append(
            "no photograph beyond the anchor could be used, so nothing corrected "
            "what the model invented"
        )

    return GaussianCloud(
        means=state["means"].detach().cpu().numpy(),
        scales=state["scales"].detach().cpu().numpy(),
        quats=state["quats"].detach().cpu().numpy(),
        colors=state["colors"].detach().cpu().numpy(),
        opacities=state["opacities"].reshape(-1).detach().cpu().numpy(),
        # By construction, and stated as such in the report the customer reads.
        angular_coverage=1.0,
        views_used=corrected + 1,
        notes=tuple(notes),
    )
