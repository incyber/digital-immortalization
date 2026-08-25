"""Whether a set of photographs can train a usable likeness.

Checked at upload, not at training time, for a commercial reason: a training
run costs GPU minutes and takes long enough that the customer has left the
page. Rejecting a bad set before payment is a filter; rejecting it after is a
refund and an apology to someone who has just uploaded pictures of their dead
parent.

The thresholds come from current FLUX LoRA practice - 15 usable images is the
floor, 20-30 the working range, and past roughly 40 you get overfitting rather
than fidelity. What matters more than the count is coverage, which is why a
set of thirty head-on portraits is refused while twenty varied ones pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

# Below this a LoRA does not converge on a stable identity.
MIN_USABLE = 15
# The range this pipeline is tuned for.
RECOMMENDED_MIN = 20
RECOMMENDED_MAX = 30
# Past here, extra images add training time and overfitting, not likeness.
MAX_ACCEPTED = 40

# FLUX trains at 1024; below 512 on the short edge the face carries too little
# detail to be worth a training slot.
MIN_SHORT_EDGE = 512

# A face smaller than this fraction of the frame is a person in a landscape,
# not a portrait.
MIN_FACE_FRACTION = 0.04

# Laplacian variance below this reads as motion blur or heavy upscaling.
MIN_SHARPNESS = 60.0

# Half-body coverage is the difference between a talking head and the
# neck-and-torso avatar this product promises. Enforced as a minimum.
MIN_HALF_BODY = 5


class Verdict(str, Enum):
    OK = "ok"
    REJECTED = "rejected"


class Reason(str, Enum):
    TOO_SMALL = "resolution below 512px on the short edge"
    NO_FACE = "no face detected"
    MANY_FACES = "more than one face in frame"
    BLURRY = "too blurry or heavily upscaled"
    FACE_TOO_SMALL = "face occupies too little of the frame"


@dataclass
class PhotoVerdict:
    """One image, and why it was kept or dropped."""

    filename: str
    verdict: Verdict
    reasons: list[Reason] = field(default_factory=list)
    # Fraction of the frame height spanned by the detected face. Used to tell
    # a head-and-shoulders crop from a half-body shot.
    face_height_fraction: float = 0.0

    @property
    def is_half_body(self) -> bool:
        """A face occupying less than a third of the frame implies the torso
        is in shot. Crude, but it separates the two framings reliably enough
        to hold a customer to the shot list."""
        return 0.0 < self.face_height_fraction < 0.33


@dataclass
class SetVerdict:
    """Whether the set as a whole can train a likeness."""

    photos: list[PhotoVerdict]
    problems: list[str] = field(default_factory=list)

    @property
    def usable(self) -> list[PhotoVerdict]:
        return [p for p in self.photos if p.verdict is Verdict.OK]

    @property
    def half_body_count(self) -> int:
        return sum(1 for p in self.usable if p.is_half_body)

    @property
    def acceptable(self) -> bool:
        return not self.problems


def inspect_photo(filename: str, image_bytes: bytes) -> PhotoVerdict:
    """Judge one image.

    Uses the Haar cascade bundled inside the pinned OpenCV wheel, for the same
    reason the renderer does: it adds no weight file carrying its own licence.
    It over-rejects on extreme angles, which is the safe direction here - a
    dropped good photo costs the customer one retake, an accepted bad one
    costs a training run.
    """
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return PhotoVerdict(filename, Verdict.REJECTED, [Reason.NO_FACE])

    height, width = frame.shape[:2]
    reasons: list[Reason] = []

    if min(height, width) < MIN_SHORT_EDGE:
        reasons.append(Reason.TOO_SMALL)

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cv2.Laplacian(grey, cv2.CV_64F).var() < MIN_SHARPNESS:
        reasons.append(Reason.BLURRY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)

    fraction = 0.0
    if len(faces) == 0:
        reasons.append(Reason.NO_FACE)
    elif len(faces) > 1:
        # Two faces means the model cannot tell which person it is learning.
        reasons.append(Reason.MANY_FACES)
    else:
        _, _, fw, fh = faces[0]
        fraction = fh / height
        if (fw * fh) / (width * height) < MIN_FACE_FRACTION:
            reasons.append(Reason.FACE_TOO_SMALL)

    verdict = Verdict.REJECTED if reasons else Verdict.OK
    return PhotoVerdict(filename, verdict, reasons, face_height_fraction=fraction)


def inspect_set(photos: list[PhotoVerdict]) -> SetVerdict:
    """Judge the set, given per-image verdicts.

    Separate from inspect_photo so the browser can show per-image feedback as
    each upload lands, then a single set-level answer once they all have.
    """
    result = SetVerdict(photos=photos)
    usable = result.usable

    if len(usable) < MIN_USABLE:
        result.problems.append(
            f"only {len(usable)} usable images; at least {MIN_USABLE} are needed, "
            f"and {RECOMMENDED_MIN}-{RECOMMENDED_MAX} gives the best likeness"
        )

    if len(usable) > MAX_ACCEPTED:
        result.problems.append(
            f"{len(usable)} images is more than the {MAX_ACCEPTED} this trains on; "
            "past that the model overfits rather than improving"
        )

    if usable and result.half_body_count < MIN_HALF_BODY:
        result.problems.append(
            f"only {result.half_body_count} images show head and shoulders or more; "
            f"at least {MIN_HALF_BODY} are needed for a half-body avatar rather "
            "than a floating head"
        )

    return result
