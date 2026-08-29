# The application: gateway, the realtime agents it spawns, and Piper.
#
# One container on purpose. The gateway starts each call's agent as a child
# process and reads the avatar's assets off local disk, so splitting them
# across hosts is not a configuration change, it is a rewrite. One machine with
# one volume is the honest shape of this system today.
#
# Piper runs as a second process rather than being imported. It is GPL, and
# speaking to it over HTTP keeps it an aggregation rather than a derived work -
# the same boundary the compose setup already draws.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg curl ca-certificates \
      libgl1 libglib2.0-0 libegl1 libgles2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
RUN pip install --no-cache-dir .

# The speech model, baked in. Downloading it on the first call would put a
# multi-hundred-megabyte fetch inside somebody's first sentence.
ARG STT_MODEL=small
RUN python -c "\
from faster_whisper import WhisperModel; WhisperModel('${STT_MODEL}', device='cpu', compute_type='int8')"

# The voice, likewise.
RUN mkdir -p /voices && cd /voices \
 && curl -sSL -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx" \
 && curl -sSL -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json"

COPY infra/piper/server.py /opt/piper/server.py
RUN pip install --no-cache-dir piper-tts

COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Built avatars live here. Without a volume mounted at this path every restart
# loses them, and the customer is told to upload their photographs again.
VOLUME ["/app/assets"]

ENV ASSETS_DIR=/app/assets \
    TTS_BACKEND=http \
    TTS_URL=http://127.0.0.1:7002 \
    STT_BACKEND=faster \
    STT_MODEL=small \
    RENDERER_BACKEND=viseme \
    COOKIES_SECURE=true \
    PIPER_VOICE=/voices/en_GB-alba-medium.onnx

EXPOSE 8000
CMD ["/entrypoint.sh"]
