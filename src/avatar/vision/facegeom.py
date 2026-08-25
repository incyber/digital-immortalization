"""Face geometry: one licence-clean detector for the whole system.

Both production renderers ship a face detector that cannot be used
commercially. MuseTalk's preprocessing pulls mmpose/DWPose; LivePortrait's
cropper calls InsightFace's buffalo_l, and its own LICENSE says the models
must be replaced for commercial use. Rather than solve that twice, everything
that needs to find a face calls this.

MediaPipe is Apache-2.0 including its published task bundles, and it runs on
CPU fast enough that this never has to touch the GPU the renderers need.

Deliberately in vision/ rather than renderer/: this is ingestion-time work.
Nothing on the turn path should import it.

One trap worth stating, because every tutorial online gets it wrong: in
MediaPipe 1.x the legacy `mp.solutions` namespace is gone. `mp.solutions.
face_mesh` raises AttributeError. The Tasks API below is the only path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# The published task bundle. Downloaded once into .tools/ rather than vendored,
# because it is 3MB of weights and does not belong in git.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(".tools/face_landmarker.task")

# MediaPipe's canonical face mesh. These indices are stable across versions and
# are what let a bounding box be derived from landmarks rather than guessed.
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP = 13
LOWER_LIP = 14
CHIN = 152
FOREHEAD = 10

# The outline MediaPipe defines for the face oval, used to build a soft mask
# without a second segmentation model.
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
)


class NoFace(ValueError):
    """Raised when no face is found. Never returns a guess."""


@dataclass(frozen=True)
class FaceGeometry:
    """Everything the renderers need to know about one face.

    bbox and mouth_box are integer pixel rectangles in the source image.
    landmarks are (N, 2) image coordinates. mask is a soft uint8 face mask.
    """

    bbox: tuple[int, int, int, int]
    mouth_box: tuple[int, int, int, int]
    landmarks: np.ndarray
    mask: np.ndarray

    @property
    def mouth_openness(self) -> float:
        """Lip separation relative to face height.

        A real measurement from the landmarks, unlike the image-statistics
        proxies tried earlier, which scored the shadow under the lower lip and
        ranked closed mouths as open.
        """
        upper = self.landmarks[UPPER_LIP]
        lower = self.landmarks[LOWER_LIP]
        gap = float(abs(lower[1] - upper[1]))
        height = max(1.0, float(self.bbox[3]))
        return gap / height

    @property
    def yaw(self) -> float:
        """How far the head is turned, from -1 (full left) to 1 (full right).

        Compares eye-to-nose distances. Zero is facing the camera, which is
        what a base frame wants.
        """
        left = self.landmarks[LEFT_EYE_OUTER]
        right = self.landmarks[RIGHT_EYE_OUTER]
        centre = self.landmarks[FOREHEAD]
        left_span = abs(centre[0] - left[0])
        right_span = abs(right[0] - centre[0])
        total = left_span + right_span
        if total < 1e-6:
            return 0.0
        return float((right_span - left_span) / total)


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Fetch the task bundle if it is not already present."""
    if path.exists():
        return path

    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"downloading the face landmark model to {path}")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


@lru_cache(maxsize=1)
def _landmarker():
    """One detector per process. Construction is expensive; detection is not."""
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )

    # CPU delegate explicitly. The default tries Metal on macOS and aborts the
    # process with "Check failed: service_ Service is unavailable" rather than
    # falling back. CPU is also correct on the server: this is ingestion-time
    # work and must not compete with the renderers for GPU.
    return FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(ensure_model()),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.IMAGE,
            num_faces=4,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
    )


def _to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    return np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float32)


def _bbox_from(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _mouth_box_from(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """A mouth rectangle derived from the lips, not a fraction of the face box.

    This is the difference the module makes: the earlier geometry assumed the
    mouth sat at a fixed proportion of a detected face rectangle, which drifts
    as soon as the head tilts.
    """
    lips = points[[MOUTH_LEFT, MOUTH_RIGHT, UPPER_LIP, LOWER_LIP]]
    cx = float(lips[:, 0].mean())
    cy = float(lips[:, 1].mean())

    span = float(abs(points[MOUTH_RIGHT][0] - points[MOUTH_LEFT][0]))
    w = max(8.0, span * 1.6)
    h = max(6.0, span * 1.0)

    x = int(max(0, cx - w / 2))
    y = int(max(0, cy - h / 2))
    return (x, y, int(min(w, width - x)), int(min(h, height - y)))


def _mask_from(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Soft face mask from the landmark outline.

    Replaces MuseTalk's face-parsing network, whose weights are trained on
    CelebAMask-HQ and restricted to non-commercial research.
    """
    outline = points[list(FACE_OVAL)].astype(np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(outline), 255)
    blur = max(3, (min(width, height) // 40) | 1)
    return cv2.GaussianBlur(mask, (blur, blur), 0)


def detect(image_bgr: np.ndarray) -> FaceGeometry:
    """The largest face in the image. Raises NoFace rather than guessing."""
    faces = detect_all(image_bgr)
    if not faces:
        raise NoFace("no face found")
    return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])


def detect_all(image_bgr: np.ndarray) -> list[FaceGeometry]:
    """Every face found, largest first."""
    import mediapipe as mp

    if image_bgr is None or image_bgr.size == 0:
        return []

    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    result = _landmarker().detect(frame)
    if not result.face_landmarks:
        return []

    faces: list[FaceGeometry] = []
    for landmarks in result.face_landmarks:
        points = _to_pixels(landmarks, width, height)
        faces.append(
            FaceGeometry(
                bbox=_bbox_from(points, width, height),
                mouth_box=_mouth_box_from(points, width, height),
                landmarks=points,
                mask=_mask_from(points, width, height),
            )
        )

    faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
    return faces
