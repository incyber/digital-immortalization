"""MuseTalk's face box, without mmpose.

Upstream derives the box from DWPose wholebody keypoints, which drags in
mmpose, mmcv and mmdet. mmcv compiles from source against a matching CUDA
toolchain, and it is the single most common reason a build of this kind fails.
This project already has a face detector - MediaPipe, Apache-2.0, behind the
facegeom service - so the box is computed from that instead and the three
packages are never installed.

The formula is upstream's, kept deliberately literal so it can be compared:

    half   = landmark[29]                        a point on the lower nose
    dist   = max_y(face) - half.y                nose to chin
    top    = max(0, half.y - dist)               the same distance back up
    box    = (min_x, top, max_x, max_y)

That produces a square-ish region centred on the mouth with as much face above
the nose as below it, which is what the model was trained to inpaint.

The mapping from the 68-point layout upstream reads to MediaPipe's 478-point
mesh is by anatomy, not by index arithmetic: the two schemes share no numbering.
"""

from __future__ import annotations

import numpy as np

# Lower nose bridge, immediately above the tip. This is upstream's landmark 29 -
# the point the whole box is measured from.
MP_NOSE_MID = 195

# The face outline. Upstream takes the horizontal extent and the chin from the
# 68 facial points; on the mesh those extremes are all on the oval.
MP_FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
)


def musetalk_bbox(
    landmarks: np.ndarray,
    frame_height: int,
    frame_width: int,
    bbox_shift: int = 0,
    extra_margin: int = 10,
) -> tuple[int, int, int, int]:
    """The (x1, y1, x2, y2) region MuseTalk inpaints.

    bbox_shift moves the top edge: positive lowers it towards the mouth,
    negative raises it towards the eyes. It is the one knob upstream exposes
    for per-subject tuning and it is passed through unchanged.

    extra_margin extends the bottom edge, which v1.5 does to keep the jaw
    inside the region when the mouth opens wide.
    """
    outline = landmarks[list(MP_FACE_OVAL)]

    half_y = float(landmarks[MP_NOSE_MID][1]) + bbox_shift
    max_y = float(outline[:, 1].max())

    # Symmetric about the nose: as much face above it as there is below.
    top = max(0.0, half_y - (max_y - half_y))

    x1 = int(max(0.0, outline[:, 0].min()))
    x2 = int(min(float(frame_width), outline[:, 0].max()))
    y1 = int(top)
    y2 = int(min(float(frame_height), max_y + extra_margin))

    if x2 - x1 <= 0 or y2 - y1 <= 0:
        raise ValueError(f"degenerate face box ({x1}, {y1}, {x2}, {y2})")
    return x1, y1, x2, y2
