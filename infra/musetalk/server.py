"""MuseTalk as a live service: a warm GPU that speaks in the person's face.

Why this is a long-lived service and not a serverless job. The models are about
2GB on the card and take tens of seconds to load. A call renders 25 frames a
second for minutes at a time, so reloading per sentence is not slow, it is
impossible. This is the one component the spend document names as needing a
rented machine rather than pay-per-job.

The stream is continuous. Frames leave at a fixed rate whether or not the
avatar is speaking: when there is audio the model renders the mouth, and when
there is not the prepared cycle frame is sent unchanged. That matters more than
it sounds. A stream that stops between sentences reads as a frozen call, and
switching sources at the moment speech begins puts a visible cut exactly where
the eye is already looking.

Preparation runs once per avatar and is cached. It is the expensive half:
every frame of the base clip is cropped, encoded to a latent, and given a
blending mask. After that a frame costs one small UNet pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

sys.path.insert(0, "/opt/MuseTalk")

from bbox import musetalk_bbox

AVATARS = Path(os.environ.get("MUSETALK_AVATARS", "/avatars"))
FACE_SERVICE_URL = os.environ.get("FACE_SERVICE_URL", "http://127.0.0.1:7001")

FPS = 25
# Audio the model is fed at a time. Shorter means the first frame of a reply
# leaves sooner; too short and the whisper encoder has too little context and
# the mouth shapes get mushy. 0.6s is fifteen frames.
CHUNK_SECONDS = 0.6
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_SECONDS) * BYTES_PER_SAMPLE

# How many frames may sit rendered but unsent before rendering pauses. A call
# that falls behind should drop to idle and recover, not accumulate a growing
# delay between what is heard and what is seen.
MAX_QUEUED_FRAMES = FPS * 2

JPEG_QUALITY = 88

app = FastAPI(title="MuseTalk live")

_models: dict = {}
_avatars: dict[str, PreparedAvatar] = {}


# ---------------------------------------------------------------------------
# models


def models() -> dict:
    """Loaded once, on first use rather than at import.

    A container that cannot reach its weights should start and say so through
    /health, not crash-loop while something upstream retries it.
    """
    if _models:
        return _models

    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.utils import load_all_model
    from transformers import WhisperModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Relative paths, resolved against /opt/MuseTalk. Upstream builds several
    # weight paths with os.path.join("models", ...) internally - the VAE and
    # the face parser both do - so the working directory is load-bearing and
    # absolute paths here would only fix half of them.
    vae, unet, pe = load_all_model(
        unet_model_path="models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="models/musetalkV15/musetalk.json",
        device=device,
    )
    pe = pe.to(device=device, dtype=dtype)
    vae.vae = vae.vae.to(device=device, dtype=dtype)
    unet.model = unet.model.to(device=device, dtype=dtype)

    whisper = (
        WhisperModel.from_pretrained("models/whisper")
        .to(device=device, dtype=dtype)
        .eval()
    )
    whisper.requires_grad_(False)

    _models.update(
        device=device,
        dtype=dtype,
        vae=vae,
        unet=unet,
        pe=pe,
        whisper=whisper,
        audio=AudioProcessor(feature_extractor_path="models/whisper"),
        parser=FaceParsing(
            left_cheek_width=90, right_cheek_width=90,
        ),
        timesteps=torch.tensor([0], device=device),
    )
    return _models


# ---------------------------------------------------------------------------
# preparation


class PreparedAvatar:
    """Everything a call needs, computed once from the base clip.

    Stored as a forward-then-reverse cycle so playback loops without a jump:
    the last frame of the cycle is adjacent to the first.
    """

    def __init__(self, avatar_id: str):
        self.avatar_id = avatar_id
        self.dir = AVATARS / avatar_id
        self.frames: list[np.ndarray] = []
        self.boxes: list[tuple[int, int, int, int]] = []
        self.latents: list[torch.Tensor] = []
        self.masks: list[np.ndarray] = []
        self.mask_boxes: list = []
        self.idle_jpegs: list[bytes] = []

    # -- disk ----------------------------------------------------------
    def exists(self) -> bool:
        return (self.dir / "latents.pt").exists()

    def load(self) -> PreparedAvatar:
        meta = json.loads((self.dir / "meta.json").read_text())
        self.boxes = [tuple(b) for b in meta["boxes"]]
        self.mask_boxes = [tuple(b) for b in meta["mask_boxes"]]
        self.latents = torch.load(self.dir / "latents.pt", map_location=models()["device"])
        self.frames = [
            cv2.imread(str(self.dir / "frames" / f"{i:08d}.png"))
            for i in range(meta["count"])
        ]
        self.masks = [
            cv2.imread(str(self.dir / "masks" / f"{i:08d}.png"), cv2.IMREAD_GRAYSCALE)
            for i in range(meta["count"])
        ]
        self._encode_idle()
        return self

    def _encode_idle(self) -> None:
        self.idle_jpegs = [
            cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
            for f in self.frames
        ]

    # -- build ---------------------------------------------------------
    def build(self, video: Path, bbox_shift: int = 0) -> PreparedAvatar:
        from musetalk.utils.blending import get_image_prepare_material

        m = models()
        frames = _decode(video)
        if not frames:
            raise HTTPException(status_code=400, detail="the base clip has no frames")

        forward_boxes = []
        forward_latents = []
        kept = []
        for frame in frames:
            landmarks = _landmarks(frame)
            if landmarks is None:
                # A frame with no detectable face is dropped rather than
                # guessed at. The cycle is a loop, so a gap in it is a jump.
                continue
            box = musetalk_bbox(landmarks, frame.shape[0], frame.shape[1], bbox_shift)
            x1, y1, x2, y2 = box
            crop = cv2.resize(
                frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4
            )
            forward_latents.append(m["vae"].get_latents_for_unet(crop))
            forward_boxes.append(box)
            kept.append(frame)

        if not kept:
            raise HTTPException(
                status_code=400, detail="no frame of the base clip contains a face"
            )

        # Forward then reverse: the clip plays out and back, so it loops with
        # no cut. Anything that indexes one of these lists must index them all.
        self.frames = kept + kept[::-1]
        self.boxes = forward_boxes + forward_boxes[::-1]
        self.latents = forward_latents + forward_latents[::-1]

        self.masks = []
        self.mask_boxes = []
        for frame, box in zip(self.frames, self.boxes, strict=True):
            mask, crop_box = get_image_prepare_material(
                frame, list(box), fp=m["parser"], mode="jaw"
            )
            self.masks.append(mask)
            self.mask_boxes.append(crop_box)

        self._save()
        self._encode_idle()
        return self

    def _save(self) -> None:
        if self.dir.exists():
            shutil.rmtree(self.dir)
        (self.dir / "frames").mkdir(parents=True)
        (self.dir / "masks").mkdir(parents=True)

        for i, (frame, mask) in enumerate(zip(self.frames, self.masks, strict=True)):
            cv2.imwrite(str(self.dir / "frames" / f"{i:08d}.png"), frame)
            cv2.imwrite(str(self.dir / "masks" / f"{i:08d}.png"), mask)

        torch.save(self.latents, self.dir / "latents.pt")
        (self.dir / "meta.json").write_text(
            json.dumps(
                {
                    "count": len(self.frames),
                    "boxes": [list(b) for b in self.boxes],
                    "mask_boxes": [list(b) for b in self.mask_boxes],
                }
            )
        )

    # -- render --------------------------------------------------------
    def render(self, audio: bytes, start: int) -> list[bytes]:
        """One audio chunk in, the JPEG frames that lip-sync it out."""
        from musetalk.utils.blending import get_image_blending
        from musetalk.utils.utils import datagen

        m = models()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav = Path(handle.name)
        _write_wav(wav, audio)

        try:
            features, length = m["audio"].get_audio_feature(str(wav), weight_dtype=m["dtype"])
            chunks = m["audio"].get_whisper_chunk(
                features, m["device"], m["dtype"], m["whisper"], length, fps=FPS,
                audio_padding_length_left=2, audio_padding_length_right=2,
            )
        finally:
            wav.unlink(missing_ok=True)

        count = len(self.frames)
        latents = [self.latents[(start + i) % count] for i in range(len(chunks))]

        out: list[bytes] = []
        index = start
        for whisper_batch, latent_batch in datagen(chunks, latents, batch_size=8):
            audio_features = m["pe"](whisper_batch.to(m["device"]))
            latent_batch = latent_batch.to(device=m["device"], dtype=m["unet"].model.dtype)
            predicted = m["unet"].model(
                latent_batch, m["timesteps"], encoder_hidden_states=audio_features
            ).sample
            decoded = m["vae"].decode_latents(
                predicted.to(device=m["device"], dtype=m["vae"].vae.dtype)
            )

            for face in decoded:
                slot = index % count
                x1, y1, x2, y2 = self.boxes[slot]
                resized = cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
                blended = get_image_blending(
                    self.frames[slot].copy(), resized, [x1, y1, x2, y2],
                    self.masks[slot], self.mask_boxes[slot],
                )
                out.append(
                    cv2.imencode(
                        ".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )[1].tobytes()
                )
                index += 1
        return out


# ---------------------------------------------------------------------------
# helpers


def _decode(video: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def _landmarks(frame: np.ndarray) -> np.ndarray | None:
    """The 478-point mesh, from the licence-clean detector.

    Over HTTP to the local face service rather than in-process: MediaPipe and
    the diffusion stack disagree about numpy, and this keeps them apart.
    """
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    response = requests.post(
        f"{FACE_SERVICE_URL}/detect",
        files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
        timeout=30,
    )
    response.raise_for_status()
    faces = response.json().get("faces") or []
    if not faces:
        return None
    return np.array(faces[0]["landmarks"], dtype=np.float32)


def _write_wav(path: Path, pcm: bytes) -> None:
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(BYTES_PER_SAMPLE)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def avatar(avatar_id: str) -> PreparedAvatar:
    if avatar_id in _avatars:
        return _avatars[avatar_id]
    prepared = PreparedAvatar(avatar_id)
    if not prepared.exists():
        raise HTTPException(status_code=404, detail=f"{avatar_id} has not been prepared")
    _avatars[avatar_id] = prepared.load()
    return _avatars[avatar_id]


# ---------------------------------------------------------------------------
# routes


@app.get("/health")
async def health():
    return {
        "ok": Path("/opt/MuseTalk/models/musetalkV15/unet.pth").exists(),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded": bool(_models),
        "prepared": sorted(p.name for p in AVATARS.glob("*") if p.is_dir()),
    }


@app.post("/avatars/{avatar_id}/prepare")
async def prepare(avatar_id: str, file: UploadFile = File(...), bbox_shift: int = 0):  # noqa: B008
    """Build the cached materials from a base clip. Once per avatar."""
    if not avatar_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="bad avatar id")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        handle.write(await file.read())
        video = Path(handle.name)

    started = time.perf_counter()
    try:
        prepared = await asyncio.to_thread(
            PreparedAvatar(avatar_id).build, video, bbox_shift
        )
    finally:
        video.unlink(missing_ok=True)

    _avatars[avatar_id] = prepared
    return {
        "avatar_id": avatar_id,
        "frames": len(prepared.frames),
        "seconds": round(time.perf_counter() - started, 1),
    }


@app.websocket("/avatars/{avatar_id}/stream")
async def stream(websocket: WebSocket, avatar_id: str):
    """A continuous 25fps stream of frames, speaking or not.

    Audio arrives as binary PCM16 mono at 16kHz. Frames leave as JPEG. The
    client never has to decide which source to show, because there is only one.
    """
    await websocket.accept()

    try:
        prepared = avatar(avatar_id)
    except HTTPException as exc:
        await websocket.close(code=1008, reason=str(exc.detail))
        return

    pending = bytearray()
    rendered: asyncio.Queue[bytes] = asyncio.Queue()
    index = 0
    stop = asyncio.Event()

    async def receive() -> None:
        nonlocal pending
        while not stop.is_set():
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                stop.set()
                return
            if (data := message.get("bytes")) is not None:
                pending.extend(data)
            # A flush ends an utterance: render whatever is left rather than
            # holding it back until enough arrives to fill a whole chunk.
            elif (text := message.get("text")) is not None:
                op = json.loads(text).get("op")
                # A flush ends an utterance: render what is left rather than
                # holding it back until enough arrives to fill a whole chunk.
                if op == "flush" and pending:
                    await render_chunk(bytes(pending))
                    pending = bytearray()
                # Barge-in. Everything already rendered is now wrong, because
                # the words it was rendered for are no longer being said.
                elif op == "cancel":
                    pending = bytearray()
                    while not rendered.empty():
                        rendered.get_nowait()

    async def render_chunk(audio: bytes) -> None:
        nonlocal index
        if rendered.qsize() > MAX_QUEUED_FRAMES:
            # Behind. Dropping this chunk costs a moment of idle mouth; not
            # dropping it costs a permanently growing gap between sound and
            # picture.
            return
        frames = await asyncio.to_thread(prepared.render, audio, index)
        index += len(frames)
        for frame in frames:
            await rendered.put(frame)

    async def pump() -> None:
        nonlocal pending, index
        interval = 1.0 / FPS
        next_at = time.monotonic()
        idle_index = 0

        while not stop.is_set():
            if len(pending) >= CHUNK_BYTES:
                chunk, pending = bytes(pending[:CHUNK_BYTES]), pending[CHUNK_BYTES:]
                asyncio.create_task(render_chunk(chunk))

            try:
                frame = rendered.get_nowait()
            except asyncio.QueueEmpty:
                # Silence: the prepared frame, already encoded. The cycle keeps
                # advancing so the head does not stall between sentences.
                frame = prepared.idle_jpegs[idle_index % len(prepared.idle_jpegs)]
                idle_index += 1
                index = idle_index

            await websocket.send_bytes(frame)

            next_at += interval
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))

    try:
        await asyncio.gather(receive(), pump())
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        with suppress(RuntimeError):
            await websocket.close()


if __name__ == "__main__":
    import uvicorn

    AVATARS.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        ["/opt/facegeom-venv/bin/python", "-m", "uvicorn", "server:app",
         "--host", "127.0.0.1", "--port", "7001", "--log-level", "warning"],
        cwd="/opt/facegeom",
    )
    uvicorn.run(app, host="0.0.0.0", port=7100)
