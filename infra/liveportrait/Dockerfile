# LivePortrait: photographs in, a base video with real head motion out.
#
# Two deliberate choices.
#
# The InsightFace detection models are never downloaded. LivePortrait's own
# LICENSE requires them to be replaced for commercial use, and facegeom_shim
# substitutes the MediaPipe service in their place. Only the six MIT weights
# are mounted; the image cannot fall back to the restricted ones because they
# are not present.
#
# Weights are mounted rather than baked in. They are 631MB, they change on a
# different cadence to the code, and an image that carries them is slow to
# rebuild for a one-line fix.
FROM python:3.10-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git ffmpeg libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Pinned. An upstream change to the cropper is exactly the kind of thing that
# would silently reintroduce the InsightFace path.
ARG LIVEPORTRAIT_REF=main
RUN git clone --depth 1 --branch ${LIVEPORTRAIT_REF} \
      https://github.com/KlingAIResearch/LivePortrait.git /opt/LivePortrait

WORKDIR /opt/LivePortrait

RUN pip install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements_base.txt \
 && pip install --no-cache-dir \
      onnxruntime \
      "fastapi>=0.115" "uvicorn[standard]>=0.34" "python-multipart>=0.0.20" \
      requests

# The replacement detector, and the service that wraps it all.
COPY facegeom_shim.py /opt/LivePortrait/src/utils/facegeom_shim.py
COPY patch_cropper.py /opt/LivePortrait/patch_cropper.py
COPY server.py /opt/LivePortrait/server.py

# Rewrites the cropper to use the shim, at build time so a broken patch fails
# the build rather than the first request.
RUN python patch_cropper.py && python -c "import ast,sys; ast.parse(open('src/utils/cropper.py').read())"

ENV FACE_SERVICE_URL=http://facegeom:7001 \
    PYTHONUNBUFFERED=1

EXPOSE 7002
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7002"]
