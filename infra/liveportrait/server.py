"""LivePortrait as a service.

One job: turn a customer's photograph into a short video of that person with
real head motion, blinks and micro-expression. That clip becomes the base the
lip-sync renderer animates during a call.

Driven by a generic motion clip rather than generated from nothing. The
distinction matters for this product: a driving video warps the face you gave
it, so identity is preserved by construction. A generative image-to-video model
re-invents the face on every run, and a recreation of somebody's parent that
drifts between takes is worse than one that is merely imperfect.
"""

import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

sys.path.insert(0, "/opt/LivePortrait")

WEIGHTS = Path(os.environ.get("LP_WEIGHTS", "/weights"))
DRIVING = Path(os.environ.get("LP_DRIVING", "/driving/idle.mp4"))
OUTPUT_DIR = Path(tempfile.gettempdir()) / "liveportrait"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LivePortrait")


@app.get("/health")
async def health():
    missing = [
        name
        for name in (
            "liveportrait/base_models/appearance_feature_extractor.pth",
            "liveportrait/base_models/motion_extractor.pth",
            "liveportrait/base_models/spade_generator.pth",
            "liveportrait/base_models/warping_module.pth",
            "liveportrait/landmark.onnx",
            "liveportrait/retargeting_models/stitching_retargeting_module.pth",
        )
        if not (WEIGHTS / name).exists()
    ]
    return {
        "ok": not missing,
        "missing_weights": missing,
        "driving_clip": DRIVING.exists(),
        # Recorded so it is visible in any health dump that the restricted
        # models are absent rather than merely unused.
        "insightface_present": (WEIGHTS / "insightface").exists(),
    }


@app.post("/animate")
async def animate(
    background: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008
):
    """Photograph in, animated clip out."""
    if not DRIVING.exists():
        raise HTTPException(status_code=503, detail="no driving clip is configured")

    job = uuid.uuid4().hex[:12]
    workdir = OUTPUT_DIR / job
    workdir.mkdir(parents=True, exist_ok=True)

    source = workdir / "source.jpg"
    source.write_bytes(await file.read())

    command = [
        sys.executable, "inference.py",
        "-s", str(source),
        "-d", str(DRIVING),
        "-o", str(workdir),
        # CPU here. On the GPU box this becomes the default device; the flag
        # exists so the same image runs on a developer machine at all.
        "--flag_force_cpu",
    ]

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        command, cwd="/opt/LivePortrait", capture_output=True, text=True, timeout=1800
    )

    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-800:]
        raise HTTPException(status_code=500, detail=f"animation failed: {tail}")

    produced = sorted(workdir.rglob("*.mp4"))
    if not produced:
        raise HTTPException(status_code=500, detail="animation produced no video")

    # Largest output is the animation; LivePortrait also writes a concatenated
    # comparison video which is smaller and not what we want.
    clip = max(produced, key=lambda p: p.stat().st_size)
    background.add_task(_cleanup, workdir, keep=clip)
    return FileResponse(clip, media_type="video/mp4", filename="base.mp4")


def _cleanup(workdir: Path, keep: Path) -> None:
    for path in workdir.rglob("*"):
        if path.is_file() and path != keep:
            path.unlink(missing_ok=True)
