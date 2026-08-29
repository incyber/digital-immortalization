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
#   spconv, FlexiCubes             Apache-2.0. plyfile, xformers  BSD.
#   rembg MIT, its u2net weights   Apache-2.0.
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
#   open3d is the exception, and it is installed. It was excluded with the
#   others on the reasoning that nothing here extracts a mesh, but
#   trellis.representations imports it eagerly, so the Gaussian path does not
#   in fact avoid it - the build proved that, which is what the build is for.
#   Its licence is MIT, read at source, so it is permissible where the GPL
#   pair are not.
#   We never extract a mesh, so none of it is imported: trellis/utils/__init__
#   is empty and trellis.renderers resolves lazily, so nothing restricted is
#   reachable from the Gaussian path this worker uses.
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
# non-commercial rasterisers come from. plyfile and safetensors are added
# because the Gaussian representation imports them directly.
ARG TRELLIS_REF=main
RUN git clone --depth 1 --branch ${TRELLIS_REF} \
      https://github.com/microsoft/TRELLIS.git /opt/TRELLIS

RUN pip install --no-cache-dir \
      pillow imageio imageio-ffmpeg tqdm easydict scipy ninja \
      "opencv-python-headless>=4.10,<5" rembg onnxruntime \
      "transformers>=4.44" safetensors plyfile huggingface_hub \
      "spconv-cu120" "numpy<2" boto3 runpod \
 && pip install --no-cache-dir \
      "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

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
RUN python -c "\
import importlib.util, sys; \
restricted = ['diff_gaussian_rasterization', 'diffoctreerast', 'nvdiffrast', \
              'insightface', 'smplx', 'pymeshfix', 'igraph']; \
found = [n for n in restricted if importlib.util.find_spec(n)]; \
sys.exit(f'restricted modules present in a shipped image: {found}') if found else \
print('no restricted modules present')"

# The Gaussian path must import with none of them, which is the claim the
# licence posture rests on. Checked at build time so a TRELLIS update that
# moves an import into the hot path is caught here rather than in an audit.
# open3d, MIT. See the note above: the Gaussian path imports it whether or not
# a mesh is ever produced.
RUN pip install --no-cache-dir "open3d>=0.18"

RUN cd /opt/TRELLIS && python -c "\
import trellis; \
from trellis.pipelines import TrellisImageTo3DPipeline; \
from trellis.representations import Gaussian; \
print('trellis gaussian path imports clean')"

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
