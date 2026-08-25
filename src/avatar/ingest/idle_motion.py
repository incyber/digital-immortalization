"""Authoring our own idle head motion.

LivePortrait is driven by a motion template: a sequence of rotation,
translation, scale, expression and eyelid values. It ships several, extracted
from videos of real people.

We generate ours instead. A template is pure numbers - there is no imagery in
it - but an extracted one still encodes a specific person's mannerisms, and
shipping that means every customer's late parent moves like the same stranger.
Generating it also makes the motion tunable, which an extracted clip is not.

Ranges below are taken from measuring the shipped templates, so the output
sits in the same space the model was trained to consume:

    scale         ~1.61, drifting under 3%
    translation   about +-0.08 in x and y, z fixed at 0
    rotation      under 0.09 absolute drift from the first frame
    expression    63 values, magnitude under 0.09
    eyelids       ~0.25 open, dropping towards 0 on a blink
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 30

# Measured from the shipped templates. Staying inside these keeps the motion
# in the range the model was trained on; exceeding them makes the head shear.
BASE_SCALE = 1.61
MAX_YAW_RAD = 0.055      # ~3 degrees
MAX_PITCH_RAD = 0.035    # ~2 degrees
MAX_ROLL_RAD = 0.018     # ~1 degree
MAX_SHIFT = 0.012
EYES_OPEN = 0.26
EYES_CLOSED = 0.02
LIP_CLOSED = 0.002

# A blink is roughly 120ms and people blink every few seconds. Regular blinking
# reads as mechanical, so the interval is jittered.
BLINK_FRAMES = 4
BLINK_EVERY_S = 4.2
BLINK_JITTER_S = 1.6


@dataclass(frozen=True)
class IdleStyle:
    """How animated the idle motion is.

    Exposed because it is a product decision, not a technical one: a
    recreation of a still, formal person should not sway like an animated one.
    """

    name: str
    sway: float = 1.0      # multiplier on rotation and translation
    blink_rate: float = 1.0

    @staticmethod
    def calm() -> IdleStyle:
        return IdleStyle("calm", sway=0.55, blink_rate=0.8)

    @staticmethod
    def natural() -> IdleStyle:
        return IdleStyle("natural", sway=1.0, blink_rate=1.0)

    @staticmethod
    def animated() -> IdleStyle:
        return IdleStyle("animated", sway=1.6, blink_rate=1.25)


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rotation matrix from three small angles, applied yaw-pitch-roll."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ rx @ ry).astype(np.float32)


def _blink_frames(total: int, style: IdleStyle, rng: np.random.Generator) -> set[int]:
    """Which frames a blink starts on.

    Jittered rather than periodic: a blink exactly every four seconds is one of
    the clearest tells that something is generated.
    """
    frames: set[int] = set()
    interval = BLINK_EVERY_S / max(0.1, style.blink_rate)
    t = rng.uniform(0.4, interval)
    while t * FPS < total:
        start = int(t * FPS)
        frames.update(range(start, min(total, start + BLINK_FRAMES)))
        t += max(0.8, interval + rng.uniform(-BLINK_JITTER_S, BLINK_JITTER_S))
    return frames


def build_idle_template(
    seconds: float = 6.0,
    style: IdleStyle | None = None,
    seed: int = 0,
) -> dict:
    """A motion template of gentle, looping idle movement.

    The cycle completes a whole number of times over the clip so the loop
    point is seamless; a template that ends mid-sway visibly snaps when it
    repeats.
    """
    style = style or IdleStyle.natural()
    rng = np.random.default_rng(seed)
    total = max(2, int(seconds * FPS))

    blinks = _blink_frames(total, style, rng)

    # Whole numbers of cycles, and mutually prime so the head does not trace
    # the same path twice.
    yaw_cycles, pitch_cycles, roll_cycles = 1.0, 2.0, 3.0
    phase = rng.uniform(0, 2 * math.pi, size=3)

    motion: list[dict] = []
    eyes: list[np.ndarray] = []
    lips: list[np.ndarray] = []

    for i in range(total):
        u = i / total

        yaw = math.sin(2 * math.pi * yaw_cycles * u + phase[0]) * MAX_YAW_RAD * style.sway
        pitch = math.sin(2 * math.pi * pitch_cycles * u + phase[1]) * MAX_PITCH_RAD * style.sway
        roll = math.sin(2 * math.pi * roll_cycles * u + phase[2]) * MAX_ROLL_RAD * style.sway

        shift_x = math.sin(2 * math.pi * u + phase[0]) * MAX_SHIFT * style.sway
        shift_y = math.cos(2 * math.pi * u + phase[1]) * MAX_SHIFT * 0.6 * style.sway

        motion.append(
            {
                "scale": np.array([[BASE_SCALE]], dtype=np.float32),
                "R_d": _rotation(yaw, pitch, roll).reshape(1, 3, 3),
                # Left neutral. Expression belongs to the speech renderer; a
                # smile baked into the idle loop would play under every word.
                "exp": np.zeros((1, 21, 3), dtype=np.float32),
                "t": np.array([[shift_x, shift_y, 0.0]], dtype=np.float32),
            }
        )

        openness = EYES_CLOSED if i in blinks else EYES_OPEN
        eyes.append(np.array([[openness, openness]], dtype=np.float32))
        lips.append(np.array([[LIP_CLOSED]], dtype=np.float32))

    return {
        "n_frames": total,
        "output_fps": FPS,
        "motion": motion,
        "c_d_eyes_lst": eyes,
        "c_d_lip_lst": lips,
    }


def write_idle_template(path: Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(build_idle_template(**kwargs), fh)
    return path
