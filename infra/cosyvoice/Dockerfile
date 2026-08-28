# CosyVoice 2: the person's own voice, cloned from a short reference.
#
# Chosen over ElevenLabs for one structural reason, not price. Hosted cloning
# services require the subject to record a consent statement, and a dead person
# cannot. Self-hosting removes the policy question because there is no vendor
# to have a policy.
#
# Apache-2.0 on both the code (QwenAudio/CosyVoice) and the weights
# (FunAudioLLM/CosyVoice2-0.5B), and 150ms streaming latency by its own
# measurement.
FROM python:3.10-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git git-lfs ffmpeg libgl1 libglib2.0-0 curl build-essential sox libsox-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

ARG COSY_REF=main
RUN git clone --depth 1 --branch ${COSY_REF} --recursive \
      https://github.com/QwenAudio/CosyVoice.git /opt/CosyVoice || \
    git clone --depth 1 --recursive \
      https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice

WORKDIR /opt/CosyVoice

# torch from PyPI: --index-url would replace PyPI entirely and break build
# dependencies, which is how the LivePortrait image failed first time.
# torchaudio is present for CosyVoice's own imports but its loader is not
# used: recent versions delegate decoding to TorchCodec, which has no wheel
# for this platform and fails at call time rather than at import. The server
# reads the reference with soundfile instead.
RUN pip install --no-cache-dir torch torchaudio \
 && pip install --no-cache-dir \
      "fastapi>=0.115" "uvicorn[standard]>=0.34" "python-multipart>=0.0.20" \
      modelscope huggingface_hub soundfile librosa \
 # requirements.txt is not usable as-is: it pins onnxruntime-gpu, deepspeed
 # and tensorrt-cu12, none of which resolve without CUDA, and pip fails the
 # whole file when one entry is unsatisfiable - which previously left
 # hyperpyyaml missing and surfaced as a runtime error rather than a build
 # failure. The CPU-viable subset is listed explicitly instead.
 && pip install --no-cache-dir \
      conformer diffusers "HyperPyYAML>=1.2" inflect librosa \
      "modelscope>=1.20" networkx "numpy<2" omegaconf onnx onnxruntime \
      openai-whisper "protobuf>=4.25" "pydantic>=2.7" pyworld rich \
      "transformers>=4.51" x-transformers hydra-core lightning \
      matplotlib gdown wget pyarrow tensorboard
 # WeTextProcessing is omitted: it builds pywrapfst from source and needs
 # OpenFST headers that are not present here. It provides Chinese text
 # normalisation, which this deployment does not use.

COPY server.py /opt/CosyVoice/server.py

ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS \
    COSY_MODEL_DIR=/models/CosyVoice2-0.5B \
    PYTHONUNBUFFERED=1

EXPOSE 7003
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7003"]
