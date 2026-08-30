# syntax=docker/dockerfile:1.7
# The splat worker: gsplat and TRELLIS in one image, and none of our code.
#
# Its own image rather than a task on infra/serverless. That one carries
# LivePortrait and CosyVoice and is already about 15GB; this one needs a CUDA
# *devel* base because gsplat compiles kernels, and its jobs run for minutes
# rather than seconds. Sharing would make every sixty-second voice job pay the
# cold start of a 3D generator it never calls.
#
# This image is publishable, on the same terms as the other one. It contains
# open-source projects, their weights, and a bootstrap that can only download,
# verify and exec. The handler, the reconstruction route and the generation
# route travel in a hash-verified archive from the operator's own object
# storage and never enter a container registry.
#
# LICENCE POSTURE, which is the reason several obvious lines are absent:
#
#   gsplat (Nerfstudio)          Apache-2.0. An independent implementation.
#   TRELLIS + TRELLIS-image-large  MIT, code and weights.
#   DINOv2 (TRELLIS's conditioner) Apache-2.0 since the 2023 relicence.
#   MediaPipe + its task bundles   Apache-2.0.
#   spconv  Apache-2.0.  xformers  BSD.  utils3d, open3d, rembg  MIT.
#   FlexiCubes                   Apache-2.0, but only at the submodule commit
#                                TRELLIS pins today - see the clone below.
#
# Deliberately NOT installed, and asserted absent below:
#
#   diff-gaussian-rasterization  the Inria rasteriser. "Research and evaluation
#                                purposes only." TRELLIS's --mipgaussian flag
#                                installs it; we do not use that flag.
#   diffoctreerast               explicitly a derivative work of the same Inria
#                                software, under the same licence.
#   nvdiffrast                   NVIDIA Source Code License, non-commercial.
#   insightface, smplx           forbidden across this whole system.
#
#   The mesh half of TRELLIS's basic dependency list - pymeshfix (GPL-3.0),
#   python-igraph (GPL-2.0), pyvista, xatlas, trimesh - is also absent.
#
#   plyfile is GPL-3.0-or-later - every release of it, checked across the
#   whole version history on PyPI, not just the newest. An earlier revision of
#   this file called it BSD and installed it. It is imported at module scope by
#   trellis/representations/gaussian/gaussian_model.py, so the Gaussian path
#   does not avoid it by accident; it is removed deliberately, below.
#
#   kaolin is Apache-2.0 overall, but kaolin/__init__.py eagerly imports
#   kaolin.non_commercial, which is NVIDIA Source Code Licensed - the same
#   "research or evaluation purposes only" terms as nvdiffrast. FlexiCubes
#   imports one pure-Python helper from it. Also removed deliberately, below.
#
#   open3d IS installed, and is the one member of the mesh half that stays.
#   trellis/pipelines/__init__.py imports TrellisTextTo3DPipeline eagerly, and
#   that module imports open3d at module scope, so even the image pipeline
#   cannot avoid it. Its licence is MIT, read at source, so it is permissible
#   where the GPL and non-commercial packages are not.
#
# WHAT THE GAUSSIAN PATH ACTUALLY REACHES, resolved rather than assumed:
#
#   trellis.renderers resolves lazily through __getattr__, so the Inria and
#   nvdiffrast rasterisers are genuinely unreachable at import. But
#   trellis/__init__.py imports representations eagerly, representations
#   imports .mesh eagerly, and TRELLIS-image-large's pipeline.json lists
#   slat_decoder_mesh, which Pipeline.from_pretrained loads unconditionally.
#   The mesh *representation* is therefore always imported, whatever
#   formats=["gaussian"] does at inference time. That is why FlexiCubes must be
#   present, and why the two restricted imports behind it are patched out
#   rather than wished away.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip python3.10-dev \
      git git-lfs ffmpeg curl build-essential ninja-build \
      libgl1 libglib2.0-0 libegl1 libgles2 \
 && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# torch, from the CUDA index.
#
# Its own layer with --index-url, which replaces PyPI entirely rather than
# adding to it - the same shape as infra/serverless/Dockerfile, and for the
# same reason: putting a build dependency in this layer is how that image's
# first attempt failed.
#
# xformers is installed here, from the same index, rather than from PyPI with a
# version pin. TRELLIS's *sparse* attention only implements the xformers and
# flash-attn backends - ATTN_BACKEND=sdpa raises at import in
# trellis/modules/sparse/attention, whatever the dense path supports - so
# xformers is not optional, and resolving it against torch on this index is
# what stops the pair drifting apart on a rebuild.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
      torch torchvision xformers

# --------------------------------------------------------------------------
# gsplat, compiled here rather than at first use.
#
# The extension is JIT-compiled on first import by default, which would put a
# multi-minute nvcc run inside the first customer's build and repeat it on
# every cold worker. TORCH_CUDA_ARCH_LIST is required because the build machine
# has no GPU to detect: L4 is 8.9, A100 is 8.0, A10/A40 are 8.6, H100 is 9.0.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    MAX_JOBS=4
# setuptools and packaging first, explicitly. gsplat's build imports
# `packaging` while compiling its CUDA extension, and this base image ships a
# pip with no setuptools alongside it - so the install fails inside the build
# rather than at resolution, which is why the error names a missing module
# instead of a missing dependency.
RUN pip install --no-cache-dir "setuptools>=70" wheel packaging ninja

RUN pip install --no-cache-dir "gsplat==1.5.3" \
 && python -c "from gsplat.cuda._backend import _C; print('gsplat kernels built')"

# --------------------------------------------------------------------------
# MediaPipe, and the two task bundles the reconstruction route needs.
#
# In the same environment as everything else here, unlike the other worker
# where CosyVoice forced a second virtualenv. numpy is pinned below 2 for the
# whole image because MediaPipe's wheels are built against 1.x.
RUN pip install --no-cache-dir "mediapipe>=0.10.14,<0.11" "numpy<2"

RUN mkdir -p /models \
 && curl -sSL -o /models/face_landmarker.task \
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
 && curl -sSL -o /models/selfie_multiclass.tflite \
      "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

# --------------------------------------------------------------------------
# TRELLIS.
#
# The dependency list is TRELLIS's own --basic set with the mesh half removed,
# written out rather than run through setup.sh: setup.sh takes the whole set,
# and the whole set is where the GPL mesh-repair tools and the optional
# non-commercial rasterisers come from. safetensors and utils3d are added
# because the Gaussian representation imports them directly.
#
# --recurse-submodules is not optional and its absence is what broke the three
# builds before this one. FlexiCubes is a git submodule, not a pip package:
# trellis/representations/mesh/cube2mesh.py does
# `from .flexicubes.flexicubes import FlexiCubes` at module scope, so without
# the submodule the import fails with a ModuleNotFoundError naming a *trellis*
# module, which reads like a missing dependency and is not one. No amount of
# pip install could ever have fixed it. TRELLIS's own README says to clone this
# way; this file did not.
#
# Pinned to a reviewed commit rather than to main, because the submodule
# pointer carries licence weight and moves: TRELLIS commit eccd70c,
# "Update FlexiCubes submodule due to license change", replaced the original
# nv-tlabs FlexiCubes - NVIDIA Source Code License, "research or evaluation
# purposes only" - with MaxtirError's Apache-2.0 fork. Building an older ref
# would silently reintroduce a non-commercial dependency. Moving this ARG
# forward means re-reading both LICENSE files before shipping.
ARG TRELLIS_REF=442aa1e1afb9014e80681d3bf604e8d728a86ee7
#
# The clone is a partial one - --filter=blob:none with --no-checkout - because
# --depth 1 --branch cannot resolve a bare commit SHA, and a SHA is the whole
# point. The exact commits of both repositories are written to
# /opt/TRELLIS/PROVENANCE before the .git directories are removed, so the image
# can still say what it is built from without carrying git history or lazily
# fetching a blob from GitHub at runtime.
RUN git clone --filter=blob:none --no-checkout \
      https://github.com/microsoft/TRELLIS.git /opt/TRELLIS \
 && git -C /opt/TRELLIS checkout --detach ${TRELLIS_REF} \
 && git -C /opt/TRELLIS submodule update --init --recursive --depth 1 \
 && test -f /opt/TRELLIS/trellis/representations/mesh/flexicubes/flexicubes.py \
 && { echo "TRELLIS    $(git -C /opt/TRELLIS rev-parse HEAD)"; \
      echo "FlexiCubes $(git -C /opt/TRELLIS/trellis/representations/mesh/flexicubes rev-parse HEAD)"; \
    } > /opt/TRELLIS/PROVENANCE \
 && find /opt/TRELLIS -name .git -prune -exec rm -rf {} + \
 && cat /opt/TRELLIS/PROVENANCE

# Every line below was resolved by walking the module-scope import closure of
# `import trellis` at the pinned commit and then removing one package at a time
# in a scratch virtualenv until the import broke. Licence for each read at
# source - the LICENSE file in the project's own repository - not from a
# summary or a badge:
#
#   pillow MIT-CMU.  tqdm MPL-2.0 AND MIT.  scipy, pandas BSD-3-Clause.
#   opencv-python-headless Apache-2.0.  rembg MIT.  onnxruntime MIT.
#   transformers Apache-2.0.  safetensors Apache-2.0.  huggingface_hub
#   Apache-2.0.  spconv Apache-2.0.  numpy BSD-3-Clause.  boto3 Apache-2.0.
#   runpod MIT.  open3d MIT.  utils3d MIT.
#
#   easydict is LGPL-3.0, which is weak copyleft rather than GPL: it is used
#   unmodified and only through import, so it does not reach the calling work.
#   It is flagged here because it is the one non-permissive licence that
#   remains, and because trellis/pipelines/samplers/flow_euler.py imports it at
#   module scope, so it cannot simply be dropped.
#
# imageio, imageio-ffmpeg and a second copy of ninja used to be on this line.
# None is reachable from `import trellis`, and no file in this worker imports
# them; ninja is already installed in the gsplat layer above.
RUN pip install --no-cache-dir \
      pillow tqdm easydict scipy \
      "opencv-python-headless>=4.10,<5" rembg onnxruntime \
      "transformers>=4.44" safetensors huggingface_hub \
      "open3d>=0.18" "spconv-cu120" "numpy<2" boto3 runpod \
 && pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# --------------------------------------------------------------------------
# The two module-scope imports that would otherwise drag a GPL-3.0 package and
# an NVIDIA-Source-Code-Licensed one into a commercially shipped image.
#
# Neither is reachable from anything this worker runs:
#
#   plyfile  is used only by Gaussian.save_ply and Gaussian.load_ply.
#            generate.py reads get_xyz / get_scaling / get_rotation /
#            get_opacity / get_features off the Gaussian and serialises the
#            splat itself; it never calls either method.
#   kaolin   supplies one pure-Python shape validator, check_tensor, used only
#            inside FlexiCubes.__call__ - mesh extraction, which
#            formats=["gaussian"] never performs.
#
# Both are replaced by objects that raise on any use, so this is a boundary
# that fails loudly rather than a stub that quietly returns something wrong.
# If either upstream line ever changes, the assertion below fails the build and
# the dependency gets re-audited instead of being silently reinstated. That is
# the same guarantee the verification step at the bottom provides, moved to the
# two places where a restricted package could actually get back in.
RUN python - <<'PY'
import pathlib, sys

ABSENT = '''

class _RestrictedDependencyAbsent:
    """Stands in for a name whose package is barred on licence grounds."""

    def __init__(self, name, package, licence, reason):
        self._name, self._package = name, package
        self._licence, self._reason = licence, reason

    def _raise(self, attr):
        raise NotImplementedError(
            f"{self._name}.{attr} needs {self._package} ({self._licence}), "
            f"which is deliberately not installed in this image. {self._reason}"
        )

    def __getattr__(self, attr):
        self._raise(attr)

    def __call__(self, *args, **kwargs):
        self._raise("__call__")
'''

PATCHES = [
    (
        "trellis/representations/gaussian/gaussian_model.py",
        "from plyfile import PlyData, PlyElement",
        ABSENT + '''
_PLY = ("plyfile", "GPL-3.0-or-later",
        "It is reached only from Gaussian.save_ply/load_ply; this worker reads "
        "the activated tensors off the Gaussian and never serialises a .ply "
        "through TRELLIS.")
PlyData = _RestrictedDependencyAbsent("PlyData", *_PLY)
PlyElement = _RestrictedDependencyAbsent("PlyElement", *_PLY)
''',
    ),
    (
        "trellis/representations/mesh/flexicubes/flexicubes.py",
        "from kaolin.utils.testing import check_tensor",
        ABSENT + '''
check_tensor = _RestrictedDependencyAbsent(
    "check_tensor", "kaolin",
    "Apache-2.0 overall, but kaolin/__init__.py eagerly imports "
    "kaolin.non_commercial, which is NVIDIA Source Code Licensed - research "
    "or evaluation purposes only",
    "It is reached only from FlexiCubes.__call__, i.e. mesh extraction, which "
    "formats=['gaussian'] never performs.",
)
''',
    ),
]

root = pathlib.Path("/opt/TRELLIS")
for rel, old, new in PATCHES:
    path = root / rel
    src = path.read_text()
    if old not in src:
        sys.exit(
            f"licence patch is stale: {rel} no longer contains {old!r}. "
            "Re-audit that dependency before changing this line."
        )
    path.write_text(src.replace(old, new, 1))
    print(f"removed the restricted import from {rel}")
PY

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/TRELLIS-image-large', \
  local_dir='/models/TRELLIS-image-large')"

# DINOv2 is TRELLIS's image conditioner and is fetched through torch.hub at
# first use. Cached here so a cold worker does not need GitHub to be up in
# order to build somebody's father.
ENV TORCH_HOME=/models/torch
RUN python -c "\
import torch; \
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)"

# --------------------------------------------------------------------------
# The licence assertion, in the image rather than in a document.
#
# The other worker asserts that InsightFace was never downloaded. Same idea:
# if any restricted implementation is ever pulled in transitively, this fails
# the build instead of shipping. The handler's health task reports the same
# list at runtime.
#
# plyfile and kaolin are on this list now. They are not exotic transitive
# risks: both were reached directly by `import trellis`, and the obvious way to
# make a future build go green - pip install the module the traceback names -
# is exactly what this list has to prevent. A build that fails here is working.
RUN python -c "\
import importlib.util, sys; \
restricted = ['diff_gaussian_rasterization', 'diffoctreerast', 'nvdiffrast', \
              'insightface', 'smplx', 'pymeshfix', 'igraph', \
              'plyfile', 'kaolin', 'trimesh', 'xatlas', 'pyvista']; \
found = [n for n in restricted if importlib.util.find_spec(n)]; \
sys.exit(f'restricted modules present in a shipped image: {found}') if found else \
print('no restricted modules present')"

# The Gaussian path must import with none of them, which is the claim the
# licence posture rests on. Checked at build time so a TRELLIS update that
# moves an import into the hot path is caught here rather than in an audit.
#
# `import trellis` is kept deliberately, and it is the broadest of the three
# lines rather than the narrowest. It is not redundant with the two below:
# trellis/__init__.py imports models, modules, pipelines, renderers,
# representations and utils, so it is the line that would fail if a TRELLIS
# update moved a restricted import into any of them. Narrowing this step to
# just the pipeline would still import the whole package as a side effect of
# importing a submodule, but it would stop *stating* that as the requirement -
# and the check exists to state it.
#
# The two specific imports stay for the same reason they were written: they
# name the two objects generate.py actually resolves, so the check fails if
# either is moved or renamed rather than merely if the package breaks.
RUN cd /opt/TRELLIS && python -c "\
import trellis; \
from trellis.pipelines import TrellisImageTo3DPipeline; \
from trellis.representations import Gaussian; \
print('trellis gaussian path imports clean')"

# The import above proves nothing restricted is *needed*. This proves nothing
# restricted was quietly satisfied by the patch either: the two placeholders
# must still be placeholders, and must still raise. If a future TRELLIS moves
# save_ply or check_tensor into the Gaussian path, the worker fails loudly at
# runtime rather than shipping a licence breach quietly - and this line is what
# guarantees the placeholder is still the thing standing there.
WORKDIR /opt/TRELLIS
RUN python - <<'PY'
import sys
import trellis.representations.gaussian.gaussian_model as gm
from trellis.representations.mesh.flexicubes.flexicubes import check_tensor

# The exact call shapes of the real, never-taken call sites upstream.
cases = [
    ("PlyElement.describe", lambda: gm.PlyElement.describe(None, "vertex")),
    ("PlyData(...)",        lambda: gm.PlyData([None])),
    ("PlyData.read",        lambda: gm.PlyData.read("/dev/null")),
    ("check_tensor",        lambda: check_tensor(None, (None, 3), throw=False)),
]
for name, call in cases:
    try:
        call()
    except NotImplementedError:
        print(f"  {name} raises as expected")
    else:
        sys.exit(
            f"licence placeholder for {name} did not raise: a restricted "
            "package may have been reinstated"
        )
print("licence placeholders intact")
PY

# --------------------------------------------------------------------------
# The only file of ours in the image, and it does nothing but fetch the rest.
COPY bootstrap/bootstrap.py /opt/bootstrap.py

ENV APP_DIR=/opt/app \
    APP_ENTRYPOINT=handler.py \
    TRELLIS_MODEL_DIR=/models/TRELLIS-image-large \
    FACE_MODEL_PATH=/models/face_landmarker.task \
    SEG_MODEL_PATH=/models/selfie_multiclass.tflite \
    ATTN_BACKEND=xformers \
    SPCONV_ALGO=native \
    PYTHONPATH=/opt/TRELLIS

WORKDIR /opt/app
CMD ["python", "-u", "/opt/bootstrap.py"]
