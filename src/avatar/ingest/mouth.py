"""Mouth plates built from the person's own face.

The naive version drew a dark ellipse on the lips. It reads as a sticker the
moment it moves, because nothing about it is theirs: not the shape, not the
colour, not the shadow under the lower lip.

What this does instead is deform their actual mouth. A jaw drop is mostly a
downward displacement of the lower lip and chin, so each plate warps those
pixels down by an increasing amount and fills the gap that opens with interior
tone taken from the darkest part of their own face. Every pixel stays theirs:
their lip colour, their stubble, their skin.

Two things were tried and removed, recorded here so they are not tried again:

  Selecting real open-mouth crops from the set. Photographs of a person are
  almost entirely closed-mouth - across a real 37-image set the mouths were
  closed in all but a handful - so there is no range to select from.

  Ranking crops by an openness measure to find the ones with teeth. Every
  measure tried scored the shadow beneath the lower lip and the stubble line
  rather than the opening, and the highest-scoring crops were verifiably
  closed mouths. Borrowing "teeth" from one produces a pale smear across the
  lips, which is worse than not having teeth at all.

The honest limit: this is a plausible articulation of a real mouth, not a
measurement of how that person speaks. Reconstructing that needs a model
trained for it, which is the GPU renderer. This is what can be done truthfully
from photographs alone, and it keeps the texture real, which is the part a
drawn shape can never get right.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from avatar.ingest.validate import detect_faces

# Where the mouth sits inside a detected face box, as fractions.
MOUTH_X, MOUTH_Y = 0.25, 0.60
MOUTH_W, MOUTH_H = 0.50, 0.34

# Fraction of the mouth box the jaw travels at the widest plate. Beyond about a
# third the chin detaches from the face and it stops looking like a jaw.
MAX_JAW_DROP = 0.34


@dataclass
class MouthSample:
    """One mouth found in one photograph, normalised to a common size.

    Collected across the whole set so the base mouth can be chosen for
    sharpness rather than taken from whichever photograph happened to be
    picked for the head.
    """

    colour: np.ndarray   # BGR
    sharpness: float


def mouth_box_for(face: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    fx, fy, fw, fh = face
    return (
        fx + int(fw * MOUTH_X),
        fy + int(fh * MOUTH_Y),
        int(fw * MOUTH_W),
        int(fh * MOUTH_H),
    )


def measure_sharpness(crop_bgr: np.ndarray) -> float:
    """Detail in this mouth crop, normalised for size.

    Used to choose which photograph's mouth to build the plates from. The
    sharpest one carries the lip edge and stubble that make the warp read as a
    real mouth rather than a blur being stretched.
    """
    if crop_bgr.size == 0:
        return 0.0
    grey = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    normed = cv2.resize(grey, (96, 64), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(normed, cv2.CV_64F).var())


def collect_mouths(images: list[bytes], size: tuple[int, int] = (192, 128)) -> list[MouthSample]:
    """Every mouth in the set, normalised so they can be compared.

    Normalising by the detected face box rather than by pixels is what makes a
    mouth photographed from two metres comparable with one photographed from
    thirty centimetres.
    """
    samples: list[MouthSample] = []
    for data in images:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(grey)
        if not faces:
            continue

        x, y, w, h = mouth_box_for(max(faces, key=lambda f: f[2] * f[3]))
        x, y = max(0, x), max(0, y)
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            continue

        resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        samples.append(MouthSample(colour=resized, sharpness=measure_sharpness(resized)))

    return samples


def match_tone(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Recolour source to sit in reference's lighting.

    Borrowed pixels come from a different photograph, with different white
    balance. Without this the mouth is visibly a patch from another day.
    """
    src = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    for channel in range(3):
        s_mean, s_std = src[:, :, channel].mean(), src[:, :, channel].std() + 1e-6
        r_mean, r_std = ref[:, :, channel].mean(), ref[:, :, channel].std()
        src[:, :, channel] = (src[:, :, channel] - s_mean) * (r_std / s_std) + r_mean

    return cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _jaw_warp(closed_bgr: np.ndarray, drop: float) -> np.ndarray:
    """Displace the lower half of the mouth downward.

    A jaw opening is mostly a rotation of the mandible, which at the scale of a
    mouth crop is a downward shift that grows from the lip line to the chin.
    Applying it with remap keeps the person's own lip texture, colour and
    stubble, which is the whole point: the pixels stay theirs.
    """
    h, w = closed_bgr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    lip_line = h * 0.46
    # Zero above the lips, rising smoothly to full displacement at the bottom.
    below = np.clip((ys - lip_line) / max(1.0, h - lip_line), 0.0, 1.0)
    # Tapered at the corners so the lips stretch rather than shear.
    across = 1.0 - np.clip(np.abs(xs - w / 2) / (w / 2), 0.0, 1.0) ** 2

    shift = below * across * (drop * h)
    return cv2.remap(
        closed_bgr,
        xs,
        ys - shift,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _interior(closed_bgr: np.ndarray) -> np.ndarray:
    """What shows inside an open mouth.

    Built from the darkest tones already present in their own face, so the
    interior sits in the same light as the rest of the frame. Slightly warm,
    because a mouth interior is not neutral grey.
    """
    lab = cv2.cvtColor(closed_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dark = float(np.percentile(lab[:, :, 0], 4))
    lab[:, :, 0] = dark * 0.55
    lab[:, :, 1] = np.clip(lab[:, :, 1].mean() + 6.0, 0, 255)   # a little red
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def build_mouth_plates(
    base_mouth_bgr: np.ndarray,
    count: int,
) -> list[np.ndarray]:
    """Plates from closed to open, all built from this person's face.

    Plate zero is the base frame untouched. Each subsequent plate opens the jaw
    further and reveals interior behind it.
    """
    h, w = base_mouth_bgr.shape[:2]
    interior = _interior(base_mouth_bgr)

    plates = [base_mouth_bgr.copy()]
    for step in range(1, count):
        fraction = step / (count - 1)
        drop = MAX_JAW_DROP * fraction

        warped = _jaw_warp(base_mouth_bgr, drop)

        # The gap the jaw opened, as a soft mask centred on the lip line.
        gap_h = max(2, int(h * drop * 0.9))
        gap = np.zeros((h, w), np.float32)
        centre_y = int(h * 0.47 + gap_h * 0.45)
        cv2.ellipse(
            gap,
            (w // 2, centre_y),
            (max(2, int(w * 0.30)), max(1, gap_h // 2)),
            0, 0, 360, 1.0, -1,
        )
        gap = cv2.GaussianBlur(gap, (0, 0), sigmaX=max(1.0, w * 0.02))[:, :, None]

        plate = (warped * (1.0 - gap) + interior * gap).astype(np.uint8)
        plates.append(plate)

    return plates
