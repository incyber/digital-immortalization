"""Client for the face geometry service.

The detector runs in its own container rather than in this process. Two
reasons, both load-bearing:

  MediaPipe 1.x aborts the whole process on Apple Silicon. It initialises a
  Metal helper regardless of the requested delegate and calls abort() when the
  service is unavailable, so an in-process import takes the gateway down on a
  developer machine.

  In production this is ingestion-time work. Keeping it out of the renderer
  process means it never competes for the GPU, and it can scale separately -
  face detection is bursty at signup, rendering is sustained during calls.

Replaces the detectors both renderers ship: MuseTalk's mmpose/DWPose and
LivePortrait's InsightFace, neither of which may be used commercially.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import httpx
import numpy as np
from loguru import logger


class FaceServiceError(RuntimeError):
    pass


class NoFace(ValueError):
    """No face in the image. Never a guess."""


@dataclass(frozen=True)
class Face:
    """One detected face, in source-image pixel coordinates."""

    bbox: tuple[int, int, int, int]
    mouth_box: tuple[int, int, int, int]
    # Lip separation over face height. A real measurement from landmarks,
    # unlike the image-statistics proxies tried before it, which scored the
    # shadow under the lower lip and ranked closed mouths as most open.
    mouth_openness: float
    # -1 fully left, 0 facing the camera, 1 fully right.
    yaw: float
    landmarks: np.ndarray
    mask: np.ndarray | None = None

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    @property
    def is_frontal(self) -> bool:
        return abs(self.yaw) < 0.25


class FaceClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                return (await client.get(f"{self._base_url}/health")).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def detect_all(self, image_bytes: bytes, *, want_mask: bool = False) -> list[Face]:
        """Every face in the image, largest first. Empty when there is none."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(
                    f"{self._base_url}/detect",
                    params={"include_mask": str(want_mask).lower()},
                    files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                )
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            raise FaceServiceError(f"face service unreachable: {exc}") from exc

        return [_parse(face) for face in body.get("faces", [])]

    async def detect(self, image_bytes: bytes, *, want_mask: bool = False) -> Face:
        """The largest face, or raise."""
        faces = await self.detect_all(image_bytes, want_mask=want_mask)
        if not faces:
            raise NoFace("no face found")
        return faces[0]


def _parse(payload: dict) -> Face:
    mask = None
    encoded = payload.get("mask_png_b64")
    if encoded:
        raw = np.frombuffer(base64.b64decode(encoded), np.uint8)
        mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning("face mask failed to decode; continuing without it")

    return Face(
        bbox=tuple(payload["bbox"]),  # type: ignore[arg-type]
        mouth_box=tuple(payload["mouth_box"]),  # type: ignore[arg-type]
        mouth_openness=float(payload["mouth_openness"]),
        yaw=float(payload["yaw"]),
        landmarks=np.asarray(payload["landmarks"], dtype=np.float32),
        mask=mask,
    )
