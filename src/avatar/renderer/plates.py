"""Per-avatar assets shared by every renderer backend.

Two ways in:

  build_assets()      Real source material. A 10-30s idle clip of the person,
                      face visible throughout. This is the production path and
                      the one the onboarding flow drives.

  synthetic_assets()  A drawn placeholder. No source material, no face
                      detection, no weights. Exists so the call pipeline is
                      runnable and testable before any customer has uploaded
                      anything, and so CI needs no image fixtures.

Both produce the same AvatarAssets, so nothing downstream knows which was used.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Number of mouth openness steps. Six is enough for the envelope-driven
# renderer to read as speech and small enough that plate selection stays a
# lookup rather than a search.
PLATE_COUNT = 6


class NoFaceDetected(ValueError):
    """Raised when a source clip has no usable face.

    Surfaced to the customer at upload time with the frame count that failed,
    because a rejection before payment is a filter and a rejection after it is
    a refund.
    """


@dataclass
class AvatarAssets:
    """Everything a renderer needs for one avatar.

    idle_frames  the looping base video, RGB uint8, (H, W, 3)
    mouth_box    (x, y, w, h) region replaced per frame while speaking
    plates       PLATE_COUNT mouth crops, index 0 closed to PLATE_COUNT-1 open
    """

    idle_frames: list[np.ndarray]
    mouth_box: tuple[int, int, int, int]
    plates: list[np.ndarray]
    fps: int
    size: tuple[int, int]

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "assets.npz",
            idle=np.stack(self.idle_frames),
            plates=np.stack(self.plates),
        )
        (out_dir / "assets.json").write_text(
            json.dumps(
                {"mouth_box": list(self.mouth_box), "fps": self.fps, "size": list(self.size)}
            )
        )

    @classmethod
    def load(cls, out_dir: Path) -> AvatarAssets:
        out_dir = Path(out_dir)
        blob = np.load(out_dir / "assets.npz")
        meta = json.loads((out_dir / "assets.json").read_text())
        return cls(
            idle_frames=[f for f in blob["idle"]],
            mouth_box=tuple(meta["mouth_box"]),  # type: ignore[arg-type]
            plates=[p for p in blob["plates"]],
            fps=meta["fps"],
            size=tuple(meta["size"]),  # type: ignore[arg-type]
        )


def decode_video(path: Path, size: tuple[int, int], fps: int) -> list[np.ndarray]:
    """Decode a clip to raw RGB frames at a fixed size and rate.

    ffmpeg rather than OpenCV's VideoCapture because it handles the container
    and codec variety real customer uploads arrive in, and because rate and
    scale conversion happen in one pass.
    """
    w, h = size
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-vf", f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ],
        capture_output=True,
        check=True,
    )
    frame_bytes = w * h * 3
    count = len(proc.stdout) // frame_bytes
    if count == 0:
        raise ValueError(f"{path} decoded to zero frames")
    buf = np.frombuffer(proc.stdout[: count * frame_bytes], dtype=np.uint8)
    return [f for f in buf.reshape(count, h, w, 3)]


def detect_mouth_box(frames: list[np.ndarray], sample: int = 12) -> tuple[int, int, int, int]:
    """Locate the mouth region by averaging face detections across the clip.

    OpenCV's bundled Haar cascade is used deliberately over a learned detector:
    it ships inside the pinned opencv-python-headless wheel, so it adds no
    weight file carrying its own licence. The design isolates every such
    dependency, and the cheapest way to isolate one is not to take it.
    """
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    step = max(1, len(frames) // sample)
    boxes = []
    for frame in frames[::step]:
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        found = cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)
        if len(found):
            boxes.append(max(found, key=lambda b: b[2] * b[3]))

    if not boxes:
        raise NoFaceDetected(
            f"no face found in {len(frames[::step])} sampled frames; "
            "the source clip must show the face throughout"
        )

    fx, fy, fw, fh = np.mean(np.array(boxes), axis=0).astype(int)
    # Mouth occupies roughly the lower third of a frontal face box. Widened
    # 15% so plate edges land on cheek rather than lip.
    mw = int(fw * 0.5 * 1.15)
    mh = int(fh * 0.33)
    mx = fx + (fw - mw) // 2
    my = fy + int(fh * 0.62)
    h, w = frames[0].shape[:2]
    return (max(0, mx), max(0, my), min(mw, w - mx), min(mh, h - my))


def extract_plates(
    frames: list[np.ndarray], mouth_box: tuple[int, int, int, int]
) -> list[np.ndarray]:
    """Pick PLATE_COUNT mouth crops spanning closed to open.

    Openness is scored by vertical gradient energy inside the mouth box: an
    open mouth introduces strong horizontal edges (lip boundaries, teeth, the
    dark interior) that a closed mouth does not. Crops are then sampled at
    even quantiles of that score, so the set spans the range actually present
    in the clip rather than an assumed one.
    """
    x, y, w, h = mouth_box
    crops = [f[y : y + h, x : x + w] for f in frames]
    energy = [
        float(np.abs(np.diff(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY).astype(np.int16), axis=0)).mean())
        for c in crops
    ]
    order = np.argsort(energy)
    picks = np.linspace(0, len(order) - 1, PLATE_COUNT).astype(int)
    return [crops[order[p]].copy() for p in picks]


def build_assets(
    source_video: Path,
    out_dir: Path | None = None,
    size: tuple[int, int] = (512, 512),
    fps: int = 25,
) -> AvatarAssets:
    """Full ingestion of a real source clip."""
    frames = decode_video(Path(source_video), size, fps)
    mouth_box = detect_mouth_box(frames)
    assets = AvatarAssets(
        idle_frames=frames,
        mouth_box=mouth_box,
        plates=extract_plates(frames, mouth_box),
        fps=fps,
        size=size,
    )
    if out_dir is not None:
        assets.save(Path(out_dir))
    return assets


def synthetic_assets(
    size: tuple[int, int] = (512, 512), fps: int = 25, seconds: float = 4.0
) -> AvatarAssets:
    """A drawn stand-in avatar, generated rather than uploaded.

    Face detection is skipped because the mouth box is known by construction.
    The idle loop carries a slow sway so the video track does not read as a
    still image, which is the same reason the real path needs a clip rather
    than a photograph.
    """
    w, h = size
    n = int(fps * seconds)
    cx, cy = w // 2, h // 2
    mouth_box = (cx - w // 8, cy + h // 8, w // 4, h // 10)

    frames: list[np.ndarray] = []
    for i in range(n):
        # Full sway cycle over the loop length, so the loop point is seamless.
        phase = 2 * np.pi * i / n
        dx, dy = int(np.sin(phase) * 4), int(np.cos(phase) * 2)
        f = np.full((h, w, 3), (18, 20, 28), dtype=np.uint8)
        cv2.ellipse(f, (cx + dx, cy + dy), (w // 4, int(h * 0.32)), 0, 0, 360, (196, 168, 148), -1)
        for side in (-1, 1):
            eye = (cx + dx + side * w // 10, cy + dy - h // 12)
            cv2.ellipse(f, eye, (w // 34, h // 46), 0, 0, 360, (250, 250, 250), -1)
            cv2.circle(f, eye, w // 70, (44, 38, 34), -1)
        cv2.line(f, (cx + dx, cy + dy - h // 40), (cx + dx, cy + dy + h // 22), (168, 140, 122), 2)
        frames.append(f)

    plates = []
    x, y, mw, mh = mouth_box
    for step in range(PLATE_COUNT):
        # Openness scales the dark interior vertically; step 0 is a closed line.
        plate = frames[0][y : y + mh, x : x + mw].copy()
        open_h = int((mh / 2 - 1) * step / (PLATE_COUNT - 1))
        cv2.ellipse(
            plate, (mw // 2, mh // 2), (mw // 2 - 2, max(1, open_h)), 0, 0, 360, (58, 30, 34), -1
        )
        if open_h > 3:
            cv2.ellipse(
                plate, (mw // 2, mh // 2 - open_h // 2), (mw // 3, max(1, open_h // 3)),
                0, 0, 360, (238, 236, 230), -1,
            )
        plates.append(plate)

    return AvatarAssets(
        idle_frames=frames, mouth_box=mouth_box, plates=plates, fps=fps, size=size
    )
