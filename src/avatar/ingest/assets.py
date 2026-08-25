"""Turns a set of photographs into something the renderer can animate.

This is the step between "the photographs are acceptable" and "you can call
them". It produces the same AvatarAssets the renderer already consumes, so
nothing downstream changes: an idle loop, a mouth region, and a set of mouth
plates from closed to open.

What it does not do is invent a person. The base frame is one of the
customer's own photographs, chosen for being the sharpest well-framed one; the
idle motion is a slow drift applied to that frame; and the mouth plates are
derived from the same face rather than drawn. The result is their face, moving
plausibly, which is a different and more honest thing than a synthesised
likeness.

The photorealistic renderer replaces the plate generation here without
touching the interface. Until then this is what makes an avatar callable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from avatar.ingest.validate import Framing, detect_faces, face_sharpness
from avatar.renderer.plates import PLATE_COUNT, AvatarAssets

# The rendered track is square; the face sits in the upper portion so there is
# room for shoulders beneath it.
OUTPUT_SIZE = 512
IDLE_FPS = 25
IDLE_SECONDS = 4.0

# How much of the output height the face should occupy, per framing. A head
# and shoulders crop puts the face larger in frame than a torso crop does.
FACE_TARGET = {Framing.HEAD: 0.42, Framing.HALF_BODY: 0.28}


class NoUsablePhoto(ValueError):
    """Raised when no photograph in the set can serve as a base frame."""


@dataclass
class Candidate:
    image: np.ndarray
    box: tuple[int, int, int, int]
    sharpness: float
    frontality: float

    @property
    def score(self) -> float:
        # Sharpness decides most of it; frontality breaks ties, because a base
        # frame turned away from the camera looks wrong the moment it speaks.
        return self.sharpness * (0.5 + self.frontality)


def _frontality(grey: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """How symmetric the face is, as a proxy for facing the camera.

    A profile has a bright side and a dark side; a frontal face does not. Cheap
    and good enough to rank candidates, which is all it is used for.
    """
    x, y, w, h = box
    crop = grey[y : y + h, x : x + w]
    if crop.size == 0 or w < 8:
        return 0.0
    left = crop[:, : w // 2].astype(np.float32)
    right = np.fliplr(crop[:, w - w // 2 :]).astype(np.float32)
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    difference = float(np.abs(left - right).mean())
    return max(0.0, 1.0 - difference / 64.0)


def choose_base_frame(images: list[bytes]) -> Candidate:
    """Pick the photograph the avatar will be built from."""
    candidates: list[Candidate] = []
    for data in images:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(grey)
        if not faces:
            continue
        box = max(faces, key=lambda f: f[2] * f[3])
        candidates.append(
            Candidate(
                image=frame,
                box=box,
                sharpness=face_sharpness(grey, box),
                frontality=_frontality(grey, box),
            )
        )

    if not candidates:
        raise NoUsablePhoto("no photograph in this set has a detectable face")
    return max(candidates, key=lambda c: c.score)


def crop_for_framing(candidate: Candidate, framing: Framing) -> np.ndarray:
    """Crop and scale so the face sits at the right size for the framing.

    The crop is clamped to the image, then letterboxed rather than stretched -
    a face that has been squashed to fill a square stops looking like the
    person, which defeats the point.
    """
    frame = candidate.image
    height, width = frame.shape[:2]
    _, fy, fw, fh = candidate.box
    fx = candidate.box[0]

    target = FACE_TARGET[framing]
    # Height of the crop that puts the face at `target` of the output.
    crop_h = int(fh / target)
    crop_w = crop_h

    # Horizontally centred on the face; vertically placed so there is headroom
    # above and body below.
    cx = fx + fw // 2
    cy = fy + fh // 2
    top = cy - int(crop_h * 0.38)
    left = cx - crop_w // 2

    # Clamp, then pad whatever falls outside rather than shifting the subject.
    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, top + crop_h - height)
    pad_right = max(0, left + crop_w - width)

    if pad_top or pad_left or pad_bottom or pad_right:
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
        top += pad_top
        left += pad_left

    crop = frame[top : top + crop_h, left : left + crop_w]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)


def locate_mouth(base_rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Mouth region within the cropped base frame.

    Re-detected on the crop rather than mapped from the original, so the box is
    correct in the coordinate space it is used in.
    """
    grey = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    faces = detect_faces(grey)
    height, width = base_rgb.shape[:2]

    if faces:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    else:
        # The crop is built around a face that was detected once already; if the
        # cascade loses it after resizing, fall back to where it must be.
        fw, fh = int(width * 0.42), int(height * 0.42)
        fx, fy = (width - fw) // 2, int(height * 0.18)

    mw = int(fw * 0.5)
    mh = int(fh * 0.30)
    mx = fx + (fw - mw) // 2
    my = fy + int(fh * 0.62)

    mx = max(0, min(mx, width - 1))
    my = max(0, min(my, height - 1))
    return (mx, my, min(mw, width - mx), min(mh, height - my))


def build_plates(base_rgb: np.ndarray, mouth_box: tuple[int, int, int, int]) -> list[np.ndarray]:
    """Mouth crops from closed to open, derived from this face.

    The closed plate is the photograph untouched. Wider plates darken and
    deepen an elliptical opening within the existing lips, so the colour and
    skin texture stay the person's own rather than being painted on.
    """
    x, y, w, h = mouth_box
    closed = base_rgb[y : y + h, x : x + w].copy()
    plates = [closed]

    for step in range(1, PLATE_COUNT):
        plate = closed.copy()
        openness = step / (PLATE_COUNT - 1)

        # The opening grows downward from the lip line, as a real jaw does.
        axis_y = max(1, int(h * 0.34 * openness))
        axis_x = max(2, int(w * 0.30 + w * 0.06 * openness))
        centre = (w // 2, int(h * 0.52 + h * 0.10 * openness))

        interior = plate.copy()
        cv2.ellipse(interior, centre, (axis_x, axis_y), 0, 0, 360, (48, 26, 30), -1)
        # Blended rather than pasted, so the edge is not a hard line.
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, centre, (axis_x, axis_y), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0
        mask = mask[:, :, None]
        plate = (plate * (1 - mask) + interior * mask).astype(np.uint8)

        if openness > 0.35:
            # A hint of teeth along the upper lip once the mouth is properly open.
            teeth = plate.copy()
            cv2.ellipse(
                teeth,
                (centre[0], centre[1] - axis_y // 2),
                (int(axis_x * 0.72), max(1, axis_y // 3)),
                0, 0, 360, (232, 228, 220), -1,
            )
            tmask = np.zeros((h, w), np.uint8)
            cv2.ellipse(
                tmask,
                (centre[0], centre[1] - axis_y // 2),
                (int(axis_x * 0.72), max(1, axis_y // 3)),
                0, 0, 360, 255, -1,
            )
            tmask = cv2.GaussianBlur(tmask, (5, 5), 0).astype(np.float32) / 255.0
            plate = (plate * (1 - tmask[:, :, None]) + teeth * tmask[:, :, None]).astype(np.uint8)

        plates.append(plate)

    return plates


def build_idle_loop(base_rgb: np.ndarray, fps: int, seconds: float) -> list[np.ndarray]:
    """A gently moving loop from a single photograph.

    A still frame reads as a frozen call. The drift is deliberately small - a
    couple of pixels and a fraction of a percent of scale - because anything
    larger looks like the photograph is sliding rather than the person
    breathing. The cycle completes exactly once over the loop so the join is
    seamless.
    """
    height, width = base_rgb.shape[:2]
    frames: list[np.ndarray] = []
    count = max(2, int(fps * seconds))

    for i in range(count):
        phase = 2 * np.pi * i / count
        dx = np.sin(phase) * 2.0
        dy = np.cos(phase * 0.5) * 1.4
        scale = 1.012 + 0.004 * np.sin(phase)

        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0.0, scale)
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        frames.append(
            cv2.warpAffine(
                base_rgb, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )
    return frames


def build_avatar_assets(
    images: list[bytes],
    framing: Framing = Framing.HEAD,
    fps: int = IDLE_FPS,
    seconds: float = IDLE_SECONDS,
) -> AvatarAssets:
    """Photographs in, renderable assets out."""
    candidate = choose_base_frame(images)
    base_bgr = crop_for_framing(candidate, framing)
    base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)

    mouth_box = locate_mouth(base_rgb)

    return AvatarAssets(
        idle_frames=build_idle_loop(base_rgb, fps, seconds),
        mouth_box=mouth_box,
        plates=build_plates(base_rgb, mouth_box),
        fps=fps,
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
    )
