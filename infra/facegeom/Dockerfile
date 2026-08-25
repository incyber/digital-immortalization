# Face geometry service.
#
# Containerised rather than run in-process for two reasons. MediaPipe 1.x
# aborts the whole process on Apple Silicon - it initialises a Metal helper
# regardless of the requested delegate and calls abort() when the service is
# unavailable - so it cannot be developed against locally any other way. And
# in production this is ingestion-time work that must not share a process, or
# a GPU, with the renderers.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "mediapipe>=1.0,<2" \
      "opencv-contrib-python-headless>=4.10" \
      "fastapi>=0.115" \
      "uvicorn[standard]>=0.34" \
      "numpy>=1.26" \
      "python-multipart>=0.0.20" \
      "loguru>=0.7"

WORKDIR /app
COPY server.py facegeom_core.py /app/

# Fetched at build time so a cold container does not pay for it on first call.
RUN mkdir -p /models && curl -sSL -o /models/face_landmarker.task \
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

ENV FACE_MODEL_PATH=/models/face_landmarker.task
EXPOSE 7001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7001"]
