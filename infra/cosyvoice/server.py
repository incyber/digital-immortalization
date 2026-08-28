"""CosyVoice 2 as a service: text plus a reference recording, speech out.

Zero-shot cloning. The reference is the customer's own recording of the person,
already validated and loudness-normalised at upload, so nothing here has to
decide whether a file is usable.

Voices are cached by avatar rather than re-derived per utterance. Deriving the
speaker embedding is the expensive part; synthesis afterwards is not, and a
call makes one request per sentence.
"""

import io
import os
import time
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

MODEL_DIR = Path(os.environ.get("COSY_MODEL_DIR", "/models/CosyVoice2-0.5B"))
REFERENCE_DIR = Path(os.environ.get("COSY_REFERENCES", "/references"))
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CosyVoice 2")

# What the model expects a prompt to be sampled at.
PROMPT_RATE = 16000


def _load_prompt(path: Path):
    """The reference recording as a (1, N) float tensor at PROMPT_RATE."""
    import librosa
    import numpy as np
    import torch

    audio, rate = __import__("soundfile").read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != PROMPT_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=PROMPT_RATE)
    return torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)

_model = None
_lock = Lock()


def model():
    """Loaded once, on first use.

    Not at import: a container that cannot reach its weights should start and
    report why through /health rather than crash-looping.
    """
    global _model
    with _lock:
        if _model is None:
            from cosyvoice.cli.cosyvoice import CosyVoice2

            _model = CosyVoice2(str(MODEL_DIR), load_jit=False, load_trt=False, fp16=False)
        return _model


@app.get("/health")
async def health():
    return {
        "ok": MODEL_DIR.exists(),
        "model_dir": str(MODEL_DIR),
        "loaded": _model is not None,
        "references": len(list(REFERENCE_DIR.glob("*.wav"))),
    }


@app.post("/references/{avatar_id}")
async def put_reference(avatar_id: str, file: UploadFile = File(...)):  # noqa: B008
    """Store the reference recording for one avatar."""
    if not avatar_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="bad avatar id")

    path = REFERENCE_DIR / f"{avatar_id}.wav"
    path.write_bytes(await file.read())
    return {"avatar_id": avatar_id, "bytes": path.stat().st_size}


@app.post("/speak")
async def speak(
    text: str = Form(...),
    avatar_id: str = Form(...),
    reference_text: str = Form(""),
):
    """Synthesise `text` in the voice of `avatar_id`."""
    reference = REFERENCE_DIR / f"{avatar_id}.wav"
    if not reference.exists():
        raise HTTPException(
            status_code=404, detail=f"no voice reference stored for {avatar_id}"
        )

    try:
        import soundfile as sf
        import torch

        # Loaded with soundfile rather than CosyVoice's load_wav, which goes
        # through torchaudio and now delegates to TorchCodec - a package with
        # no wheel for this platform, and which fails at call time rather than
        # at import. The reference is already mono 16k from upload, so there is
        # nothing for the heavier path to do.
        prompt = _load_prompt(reference)
        started = time.perf_counter()

        chunks = []
        for out in model().inference_zero_shot(
            text, reference_text, prompt, stream=False
        ):
            chunks.append(out["tts_speech"])

        if not chunks:
            raise HTTPException(status_code=500, detail="synthesis produced no audio")

        audio = torch.cat(chunks, dim=1)
        buffer = io.BytesIO()
        sf.write(buffer, audio.squeeze(0).numpy(), model().sample_rate, format="WAV")

        elapsed = time.perf_counter() - started
        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={"X-Synthesis-Seconds": f"{elapsed:.2f}"},
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc
