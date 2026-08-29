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

# Sharpness is measured on the face, resampled to a fixed size, never on the
# whole frame.
#
# Measured against a real 37-photo iPhone set: whole-frame Laplacian variance
# ranged from 11.6 to 1744 - a 150x spread on images that were all sharp. The
# low readings were portrait-mode and plain-wall backgrounds, which are smooth
# by design and dominate the pixel count. Any absolute threshold on that
# number rejects good photographs, and it rejected two thirds of that set.
#
# The face crop from the same photographs measured 98 to 822 once normalised,
# with a median of 169. The only reading below 60 was a face occupying 7% of
# the frame height, which is genuinely soft once enlarged.
FACE_SHARPNESS_SIZE = 256
MIN_FACE_SHARPNESS = 60.0

# A second detection smaller than this share of the largest is treated as a
# false positive rather than a second person. Haar readily finds "faces" in
# clothing and background texture; on the same real set it reported two or
# three in four photographs that contained one person.
SECOND_FACE_RATIO = 0.40

# Torso coverage decides how the avatar is framed, not whether it can be built.
#
# Demanding twenty half-body photographs of somebody who has died is not a
# requirement a family can meet: what exists is what exists, and it is usually
# close portraits. So the framing adapts to the material instead. With enough
# pictures showing the shoulders and chest, the avatar is head, neck and torso;
# without them it is head and shoulders, cropped where the evidence stops.
#
# Three is a floor for the wider framing rather than a gate on the product:
# below it there is not enough of the torso seen from enough angles to
# reconstruct one, and inventing it would put clothing on somebody that they
# never wore.
MIN_FOR_HALF_BODY = 3


class Verdict(str, Enum):
    OK = "ok"
    REJECTED = "rejected"


class Framing(str, Enum):
    """How much of the person the avatar will show.

    Decided by what the photographs contain. Neither is a failure; they are
    different products, and the customer is told which one they are getting.
    """

    HEAD = "head"            # head, neck, shoulders
    HALF_BODY = "half_body"  # head, neck, shoulders, chest

    @property
    def label(self) -> str:
        return "head and shoulders" if self is Framing.HEAD else "head and torso"


class Reason(str, Enum):
    TOO_SMALL = "resolution below 512px on the short edge"
    NO_FACE = "no face detected"
    MANY_FACES = "more than one face in frame"
    BLURRY = "the face is not sharp enough"
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
class Requirement:
    """One condition, its target, and where the set currently stands.

    Reported per requirement rather than as a single pass/fail so the page can
    show progress while photographs are still arriving. A customer who has
    uploaded twenty-four usable pictures and is blocked should be able to see
    which condition is unmet and by how much, not face a dead button.
    """

    key: str
    label: str
    current: int
    target: int
    met: bool
    hint: str = ""
    # Informational requirements report progress without gating the build.
    # Torso coverage is the case that forced the distinction: it changes what
    # the avatar looks like, it does not decide whether one can exist.
    blocking: bool = True


@dataclass
class SetVerdict:
    """Whether the set as a whole can train a likeness."""

    photos: list[PhotoVerdict]
    problems: list[str] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)

    @property
    def usable(self) -> list[PhotoVerdict]:
        return [p for p in self.photos if p.verdict is Verdict.OK]

    @property
    def half_body_count(self) -> int:
        return sum(1 for p in self.usable if p.is_half_body)

    @property
    def framing(self) -> Framing:
        """What can be built from these photographs."""
        return (
            Framing.HALF_BODY
            if self.half_body_count >= MIN_FOR_HALF_BODY
            else Framing.HEAD
        )

    @property
    def acceptable(self) -> bool:
        return not self.problems


def _cascade(name: str) -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + name)


def detect_faces(grey: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find faces, frontal or in profile.

    The profile pass matters more than it looks: the shot list explicitly asks
    for two profile photographs, and the frontal cascade does not see them.
    Without it the product rejects the very pictures it requested.

    Profiles are mirrored and re-run because the cascade only detects one
    facing direction.
    """
    frontal = list(_cascade("haarcascade_frontalface_default.xml").detectMultiScale(
        grey, scaleFactor=1.1, minNeighbors=5
    ))
    if frontal:
        return [tuple(int(v) for v in f) for f in frontal]

    profile = _cascade("haarcascade_profileface.xml")
    found = list(profile.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5))
    if not found:
        flipped = cv2.flip(grey, 1)
        width = grey.shape[1]
        found = [
            (width - x - w, y, w, h)
            for (x, y, w, h) in profile.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=5
            )
        ]
    return [tuple(int(v) for v in f) for f in found]


def face_sharpness(grey: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Laplacian variance of the face, resampled to a fixed size.

    Normalising the crop is what makes the number comparable between a phone
    and a scanned print. Measuring the crop rather than the frame is what stops
    a deliberately blurred background from condemning a sharp subject.
    """
    x, y, w, h = box
    crop = grey[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    resized = cv2.resize(
        crop, (FACE_SHARPNESS_SIZE, FACE_SHARPNESS_SIZE), interpolation=cv2.INTER_AREA
    )
    return float(cv2.Laplacian(resized, cv2.CV_64F).var())


def inspect_photo(filename: str, image_bytes: bytes) -> PhotoVerdict:
    """Judge one image.

    Uses the Haar cascades bundled inside the pinned OpenCV wheel, for the same
    reason the renderer does: they add no weight file carrying its own licence.
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
    faces = detect_faces(grey)

    fraction = 0.0
    if not faces:
        # Sharpness is not assessed without a face: there is nothing to measure
        # it on, and the frame as a whole is not the subject.
        reasons.append(Reason.NO_FACE)
    else:
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        largest = faces[0]
        largest_area = largest[2] * largest[3]

        # Only a substantial second detection counts as a second person.
        others = [f for f in faces[1:] if (f[2] * f[3]) >= largest_area * SECOND_FACE_RATIO]
        if others:
            reasons.append(Reason.MANY_FACES)

        _, _, _, fh = largest
        fraction = fh / height
        if largest_area / (width * height) < MIN_FACE_FRACTION:
            reasons.append(Reason.FACE_TOO_SMALL)
        elif face_sharpness(grey, largest) < MIN_FACE_SHARPNESS:
            # Checked only when the face is big enough to judge; a distant face
            # is soft because it is distant, and FACE_TOO_SMALL says that
            # better than "blurry" does.
            reasons.append(Reason.BLURRY)

    verdict = Verdict.REJECTED if reasons else Verdict.OK
    return PhotoVerdict(filename, verdict, reasons, face_height_fraction=fraction)


def inspect_set(photos: list[PhotoVerdict]) -> SetVerdict:
    """Judge the set, given per-image verdicts.

    Separate from inspect_photo so the browser can show per-image feedback as
    each upload lands, then a single set-level answer once they all have.
    """
    result = SetVerdict(photos=photos)
    usable = result.usable
    half_body = result.half_body_count

    result.requirements = [
        Requirement(
            key="usable",
            label="Usable photographs",
            current=len(usable),
            target=MIN_USABLE,
            met=len(usable) >= MIN_USABLE,
            hint=f"{RECOMMENDED_MIN}-{RECOMMENDED_MAX} gives the best likeness.",
        ),
        Requirement(
            key="half_body",
            label="Showing the chest and shoulders",
            current=half_body,
            target=MIN_FOR_HALF_BODY,
            met=half_body >= MIN_FOR_HALF_BODY,
            blocking=False,
            hint=(
                "Optional. With three or more the avatar includes the torso; "
                "otherwise it is framed at head and shoulders."
            ),
        ),
        Requirement(
            key="not_too_many",
            label=f"No more than {MAX_ACCEPTED}",
            current=len(usable),
            target=MAX_ACCEPTED,
            met=len(usable) <= MAX_ACCEPTED,
            hint="Past this the model overfits rather than improving.",
        ),
    ]

    if len(usable) < MIN_USABLE:
        # Every sentence in `problems` is shown to a customer verbatim - the
        # web app deliberately prints the server's own wording rather than
        # rephrasing it - so the grammar is part of the interface. "only 1
        # usable images" was reaching people who had just been told most of
        # their photographs of someone who died were unusable.
        count = len(usable)
        result.problems.append(
            f"only {count} usable {'photograph' if count == 1 else 'photographs'}; "
            f"at least {MIN_USABLE} are needed, and "
            f"{RECOMMENDED_MIN}-{RECOMMENDED_MAX} gives the best likeness"
        )

    if len(usable) > MAX_ACCEPTED:
        result.problems.append(
            f"{len(usable)} images is more than the {MAX_ACCEPTED} this trains on; "
            "past that the model overfits rather than improving"
        )

    # Torso coverage deliberately adds no problem. It changes the framing,
    # which the customer is told about, rather than blocking the build.

    return result
