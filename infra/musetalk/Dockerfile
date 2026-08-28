# MuseTalk as a live lip-sync service.
#
# Licences, checked against the sources rather than a summary: MuseTalk's code
# is MIT and its own disclaimer states the trained model is "available for any
# purpose, even commercially". sd-vae-ft-mse is MIT, whisper-tiny is MIT, and
# the BiSeNet face parser is MIT. Nothing here carries a non-commercial clause,
# which is the whole reason this project self-hosts.
#
# mmpose, mmcv and mmdet are deliberately absent. Upstream needs them only to
# find the face box during preparation, and mmcv compiles from source against a
# matching CUDA toolchain - the most common way a build like this fails.
# bbox.py computes the same box from the MediaPipe detector this project
# already uses, so three heavy packages become none.
#
# tensorflow, gradio and the training extras in requirements.txt are also
# absent. None are reachable from the inference path.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip python3.10-dev \
      git ffmpeg curl build-essential \
      libgl1 libglib2.0-0 libegl1 libgles2 \
 && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

# Its own layer, with --index-url replacing PyPI entirely. Everything else
# installs from PyPI afterwards.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
      torch torchvision

ARG MUSETALK_REF=main
RUN git clone --depth 1 --branch ${MUSETALK_REF} \
      https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk

WORKDIR /opt/MuseTalk

# The inference subset. requirements.txt pins tensorflow 2.12 and gradio, which
# together are several gigabytes and are not imported by anything this service
# calls.
RUN pip install --no-cache-dir \
      "diffusers==0.30.2" "accelerate==0.28.0" "transformers==4.39.2" \
      "numpy<2" "opencv-python-headless>=4.10,<5" soundfile librosa einops \
      omegaconf "huggingface_hub>=0.30" \
      "fastapi>=0.115" "uvicorn[standard]>=0.34" "python-multipart>=0.0.20" \
      websockets requests

# Weights land under /opt/MuseTalk/models because upstream resolves several of
# them relative to the working directory: the VAE joins "models" with its type,
# and the face parser defaults to ./models/face-parse-bisent.
RUN python -c "\
from huggingface_hub import snapshot_download as d; \
d('TMElyralab/MuseTalk', local_dir='models', allow_patterns=['musetalkV15/*']); \
d('stabilityai/sd-vae-ft-mse', local_dir='models/sd-vae', \
  allow_patterns=['config.json','diffusion_pytorch_model.bin']); \
d('openai/whisper-tiny', local_dir='models/whisper', \
  allow_patterns=['config.json','pytorch_model.bin','preprocessor_config.json']); \
d('ManyOtherFunctions/face-parse-bisent', local_dir='models/face-parse-bisent', \
  allow_patterns=['79999_iter.pth','resnet18-5c106cde.pth'])" \
 && test -f models/musetalkV15/unet.pth \
 && test -f models/face-parse-bisent/79999_iter.pth

# The face detector, in its own virtualenv. MediaPipe and the diffusion stack
# disagree about numpy; two environments cost disk instead of correctness.
RUN python -m venv /opt/facegeom-venv \
 && /opt/facegeom-venv/bin/pip install --no-cache-dir \
      "mediapipe>=1.0,<2" "opencv-contrib-python-headless>=4.10" \
      "fastapi>=0.115" "uvicorn[standard]>=0.34" "python-multipart>=0.0.20" \
      "numpy>=1.26" "loguru>=0.7"

COPY facegeom/server.py facegeom/facegeom_core.py /opt/facegeom/
RUN mkdir -p /models && curl -sSL -o /models/face_landmarker.task \
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

COPY musetalk/server.py musetalk/bbox.py /opt/MuseTalk/

ENV FACE_MODEL_PATH=/models/face_landmarker.task \
    FACE_SERVICE_URL=http://127.0.0.1:7001 \
    MUSETALK_AVATARS=/avatars

EXPOSE 7100
CMD ["python", "-u", "server.py"]
