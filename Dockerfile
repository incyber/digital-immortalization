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

# The voices, likewise.
#
# One per supported language, and exactly the identifiers in
# services/voices.py - the voice follows the avatar's language, so a voice
# that is present for one locale and missing for another is a call that ends
# in a 500 the moment somebody speaks. The previous single en_GB-alba-medium
# was reachable from no locale at all: nothing in VOICES names it.
RUN mkdir -p /voices && cd /voices \
 && for v in \
      en/en_US/hfc_female/medium/en_US-hfc_female-medium \
      es/es_ES/davefx/medium/es_ES-davefx-medium \
      pt/pt_BR/faber/medium/pt_BR-faber-medium \
      fr/fr_FR/siwis/medium/fr_FR-siwis-medium \
      de/de_DE/thorsten/medium/de_DE-thorsten-medium \
      it/it_IT/riccardo/x_low/it_IT-riccardo-x_low \
    ; do \
      curl -fsSL -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/$v.onnx" \
   && curl -fsSL -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/$v.onnx.json" ; \
    done

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
    PIPER_PORT=7002 \
    PIPER_VOICES_DIR=/voices \
    PIPER_VOICE=en_US-hfc_female-medium

# PIPER_PORT is set because infra/piper/server.py defaults to 5050 while
# TTS_URL above and the entrypoint's health probe both say 7002. Unset, the
# gateway talks to a port nothing is listening on and every call is silent.
#
# PIPER_VOICE is a voice *name*: the server joins it to PIPER_VOICES_DIR and
# appends .onnx. A full path here resolved to /voices//voices/....onnx.onnx.
# It is only the fallback for a request that names no voice; the application
# always names one.

EXPOSE 8000
CMD ["/entrypoint.sh"]
