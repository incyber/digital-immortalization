"""Drop-in replacement for LivePortrait's InsightFace face analyser.

LivePortrait's own LICENSE states that the InsightFace detection models must
be removed and replaced for commercial use. This is that replacement. It
presents the same tiny surface the cropper actually consumes and gets its
geometry from the MediaPipe service instead.

The surface is smaller than it looks. Everything InsightFace produces is read
as `landmark_2d_106`, and `parse_pt2_from_pt106` reduces those 106 points to
three: a left eye centre from indices [33, 35, 40, 39], a right eye centre
from [87, 89, 94, 93], and a lip centre from [52] and [61]. Nine indices decide
the crop; the rest only need to be plausible so that any bounding-box maths
downstream stays sane.

So this fills those nine from the corresponding MediaPipe landmarks and packs
the face outline into the remainder. The MIT `landmark.onnx` runner that
LivePortrait applies afterwards refines whatever it is given, which is why a
seed of this precision is enough.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import requests

FACE_SERVICE_URL = os.environ.get("FACE_SERVICE_URL", "http://facegeom:7001")

# MediaPipe canonical mesh indices.
MP_LEFT_EYE = (33, 160, 158, 133)     # outer, upper, upper, inner
MP_RIGHT_EYE = (362, 385, 387, 263)   # inner, upper, upper, outer
MP_UPPER_LIP = 13
MP_LOWER_LIP = 14
MP_FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
)

# The nine indices in the 106-point layout that actually decide the crop.
IF_LEFT_EYE = (33, 35, 40, 39)
IF_RIGHT_EYE = (87, 89, 94, 93)
IF_UPPER_LIP = 52
IF_LOWER_LIP = 61


@dataclass
class ShimFace:
    """What the cropper reads off a detected face."""

    landmark_2d_106: np.ndarray
    bbox: np.ndarray
    det_score: float = 1.0


def _to_106(mp_points: np.ndarray) -> np.ndarray:
    """Map a 478-point MediaPipe mesh onto the 106-point layout.

    Only the nine load-bearing indices need to be right. The face outline is
    resampled into the remaining slots so that anything computing extents over
    the whole array still sees the face rather than a cluster of zeros.
    """
    out = np.zeros((106, 2), dtype=np.float32)

    outline = mp_points[list(MP_FACE_OVAL)]
    resampled = np.array(
        [outline[int(i * (len(outline) - 1) / 105)] for i in range(106)],
        dtype=np.float32,
    )
    out[:] = resampled

    for slot, mp_index in zip(IF_LEFT_EYE, MP_LEFT_EYE, strict=True):
        out[slot] = mp_points[mp_index]
    for slot, mp_index in zip(IF_RIGHT_EYE, MP_RIGHT_EYE, strict=True):
        out[slot] = mp_points[mp_index]

    out[IF_UPPER_LIP] = mp_points[MP_UPPER_LIP]
    out[IF_LOWER_LIP] = mp_points[MP_LOWER_LIP]
    return out


class FaceAnalysisShim:
    """Same call shape as FaceAnalysisDIY, different geometry underneath."""

    def __init__(self, *args, **kwargs):
        self._url = kwargs.get("face_service_url", FACE_SERVICE_URL).rstrip("/")

    def prepare(self, *args, **kwargs):
        return None

    def warmup(self):
        try:
            requests.get(f"{self._url}/health", timeout=5).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"face geometry service unavailable at {self._url}: {exc}"
            ) from exc

    def get(self, img_bgr, **kwargs) -> list[ShimFace]:
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return []

        response = requests.post(
            f"{self._url}/detect",
            files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            timeout=60,
        )
        response.raise_for_status()
        faces = response.json().get("faces", [])
        if not faces:
            return []

        max_faces = kwargs.get("max_face_num", 0)
        if max_faces:
            faces = faces[:max_faces]

        out: list[ShimFace] = []
        for face in faces:
            points = np.asarray(face["landmarks"], dtype=np.float32)
            x, y, w, h = face["bbox"]
            out.append(
                ShimFace(
                    landmark_2d_106=_to_106(points),
                    bbox=np.array([x, y, x + w, y + h], dtype=np.float32),
                )
            )
        return out
