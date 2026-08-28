# The GPU worker: LivePortrait, CosyVoice and the face detector in one image.
#
# One image rather than three because RunPod Serverless bills per running
# worker, and the two models never run at the same time. Three endpoints would
# mean three cold starts, three sets of safety settings to keep correct, and no
# isolation benefit for work that all happens at signup.
#
# Weights are baked in, unlike the compose images which mount them. A
# serverless worker starts from nothing, so anything not in the image is
# downloaded on every cold start - 2.6GB of weights would dominate the run.
#
# Nothing here listens on a port for the outside world. The platform starts the
# process when a job is queued and stops it when the job ends.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip python3.10-dev \
      git git-lfs ffmpeg curl build-essential sox libsox-dev \
      libgl1 libglib2.0-0 libegl1 libgles2 \
 && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# torch, from the CUDA index.
#
# Its own layer with --index-url, which replaces PyPI entirely rather than
# adding to it. Everything else installs from PyPI afterwards; putting build
# dependencies such as flit_core in this layer is how the first attempt at the
# compose image failed.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
      torch torchvision torchaudio

# --------------------------------------------------------------------------
# LivePortrait
ARG LIVEPORTRAIT_REF=main
RUN git clone --depth 1 --branch ${LIVEPORTRAIT_REF} \
      https://github.com/KlingAIResearch/LivePortrait.git /opt/LivePortrait

WORKDIR /opt/LivePortrait
RUN pip install --no-cache-dir -r requirements_base.txt \
 && pip install --no-cache-dir \
      onnxruntime-gpu \
      "opencv-python-headless>=4.10,<5" \
      "numpy<2" \
      requests runpod

# The six MIT weights only. The InsightFace detection models live in the same
# repository and are deliberately not fetched: LivePortrait's LICENSE requires
# them to be replaced for commercial use, facegeom does that, and an image that
# does not contain them cannot quietly fall back to them.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('KlingTeam/LivePortrait', \
  local_dir='/opt/LivePortrait/pretrained_weights', \
  allow_patterns=['liveportrait/**'])" \
 && test ! -d /opt/LivePortrait/pretrained_weights/insightface

COPY liveportrait/facegeom_shim.py /opt/LivePortrait/src/utils/facegeom_shim.py
COPY liveportrait/patch_cropper.py /opt/LivePortrait/patch_cropper.py

# Rewritten at build time so a broken patch fails the build rather than the
# first paid GPU second.
RUN python patch_cropper.py \
 && python -c "import ast; ast.parse(open('src/utils/cropper.py').read())" \
 && grep -q facegeom_shim src/utils/cropper.py

# --------------------------------------------------------------------------
# CosyVoice
ARG COSY_REF=main
RUN git clone --depth 1 --branch ${COSY_REF} --recursive \
      https://github.com/QwenAudio/CosyVoice.git /opt/CosyVoice

WORKDIR /opt/CosyVoice
# requirements.txt is not installable as one unit: it pins deepspeed and
# tensorrt-cu12, pip fails the whole file when one entry is unsatisfiable, and
# the previous failure left hyperpyyaml missing - which surfaced at call time
# instead of at build time. The needed subset is listed explicitly.
RUN pip install --no-cache-dir \
      conformer diffusers "HyperPyYAML>=1.2" inflect librosa soundfile \
      "modelscope>=1.20" networkx "numpy<2" omegaconf onnx \
      openai-whisper "protobuf>=4.25" "pydantic>=2.7" pyworld rich \
      "transformers>=4.51" x-transformers hydra-core lightning \
      matplotlib gdown wget pyarrow tensorboard

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='/models/CosyVoice2-0.5B')"

# --------------------------------------------------------------------------
# Face geometry, in its own virtualenv.
#
# MediaPipe and CosyVoice do not agree about numpy. Sharing one environment
# means one of them runs on a version it was never tested against; a second
# virtualenv costs disk instead. --system-site-packages is deliberately not
# used, for the same reason.
RUN python -m venv /opt/facegeom-venv \
 && /opt/facegeom-venv/bin/pip install --no-cache-dir \
      "mediapipe>=1.0,<2" \
      "opencv-contrib-python-headless>=4.10" \
      "fastapi>=0.115" "uvicorn[standard]>=0.34" "python-multipart>=0.0.20" \
      "numpy>=1.26" "loguru>=0.7"

COPY facegeom/server.py facegeom/facegeom_core.py /opt/facegeom/
RUN mkdir -p /models && curl -sSL -o /models/face_landmarker.task \
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# --------------------------------------------------------------------------
COPY serverless/handler.py /opt/handler.py

ENV FACE_MODEL_PATH=/models/face_landmarker.task \
    FACE_SERVICE_URL=http://127.0.0.1:7001 \
    FACE_PYTHON=/opt/facegeom-venv/bin/python \
    COSY_MODEL_DIR=/models/CosyVoice2-0.5B \
    LP_WEIGHTS=/opt/LivePortrait/pretrained_weights \
    PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS

WORKDIR /opt/LivePortrait
CMD ["python", "-u", "/opt/handler.py"]
