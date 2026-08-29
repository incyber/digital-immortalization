"""One frame of animation: everything a face and body are doing at an instant.

Thirty-two numbers, fixed shape, no optional fields. A renderer receives these
and nothing else, which is what lets the whole motion system be developed and
tested with no GPU and no renderer at all - the output is numbers, and numbers
can be asserted on.

Ranges are declared here rather than in the renderer because they are the
contract. A value outside its range is a bug in the director, not something for
the renderer to clamp quietly: a clamp hides the error and produces a face that
is subtly wrong rather than obviously broken.

The angular limits come from measurements already made in this project against
real motion templates - see ingest/idle_motion.py - rather than from taste.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

# The mouth shapes, named and ordered. This tuple is the single source of
# truth, and it exists because its absence was a real latent bug: the slots
# were an unnamed tuple of fifteen floats, so every consumer had to guess which
# sound each one meant. A permuted guess is a mouth making the wrong shape for
# every word - and subtle enough to survive a demo, which is the worst kind of
# wrong.
#
# The set is the standard fifteen: silence, then the visually distinct
# consonant and vowel groups. Sounds that look identical share a slot, because
# a viewer reads lips by shape and /p/, /b/ and /m/ are one shape.
VISEME_NAMES: tuple[str, ...] = (
    "sil",   # closed, at rest
    "PP",    # p, b, m - lips pressed
    "FF",    # f, v - lip against teeth
    "TH",    # th - tongue between teeth
    "DD",    # d, t, n - tongue behind teeth
    "kk",    # k, g - back of tongue, little visible
    "CH",    # ch, j, sh - rounded and forward
    "SS",    # s, z - narrow, teeth close
    "nn",    # n, l - tongue up
    "RR",    # r - rounded, slight pucker
    "aa",    # father - open
    "E",     # bed - mid
    "ih",    # bit - narrow
    "oh",    # boat - rounded
    "ou",    # boot - tight and rounded
)

VISEME_COUNT = len(VISEME_NAMES)

# Sum of all viseme weights. Slightly above one because adjacent shapes overlap
# during a transition; far above one and the mouth tears.
MAX_VISEME_SUM = 1.2


@dataclass(frozen=True)
class Channel:
    """One animation parameter, its bounds, and how fast it may change.

    slew is in units per second. It exists because the difference between a
    face and a puppet is mostly rate: every value here can be correct and the
    result still look mechanical if they arrive too quickly.
    """

    name: str
    low: float
    high: float
    slew: float


# Head rotation in radians. The limits are the ones measured against real
# motion templates in idle_motion.py; beyond them a warp stops looking like a
# head turning and starts looking like an image being pulled.
HEAD = (
    Channel("head_yaw", -0.35, 0.35, 1.8),
    Channel("head_pitch", -0.25, 0.25, 1.6),
    Channel("head_roll", -0.18, 0.18, 1.2),
)

# Gaze is relative to the head, and is deliberately allowed to move far faster
# than the head. Human gaze is ballistic - a saccade is complete in about 40ms
# - and smoothing it is the single most uncanny thing this system could do.
GAZE = (
    Channel("gaze_yaw", -0.5, 0.5, 40.0),
    Channel("gaze_pitch", -0.4, 0.4, 40.0),
)

EYES = (
    Channel("blink", 0.0, 1.0, 30.0),
    Channel("lid_upper_l", -1.0, 1.0, 2.0),
    Channel("lid_upper_r", -1.0, 1.0, 2.0),
)

BROW = (
    Channel("brow_inner_l", -1.0, 1.0, 3.0),
    Channel("brow_inner_r", -1.0, 1.0, 3.0),
    Channel("brow_outer_l", -1.0, 1.0, 3.0),
    Channel("brow_outer_r", -1.0, 1.0, 3.0),
)

MOUTH = (
    Channel("jaw_open", 0.0, 1.0, 12.0),
    Channel("mouth_smile_l", -1.0, 1.0, 1.5),
    Channel("mouth_smile_r", -1.0, 1.0, 1.5),
    Channel("mouth_press", 0.0, 1.0, 2.0),
)

BODY = (
    Channel("torso_lean", -1.0, 1.0, 1.0),
    Channel("torso_yaw", -0.4, 0.4, 1.0),
    Channel("shoulder_raise", 0.0, 1.0, 1.5),
    Channel("breath", 0.0, 1.0, 2.0),
)

CHANNELS: tuple[Channel, ...] = HEAD + GAZE + EYES + BROW + MOUTH + BODY
BY_NAME: dict[str, Channel] = {c.name: c for c in CHANNELS}


@dataclass
class PoseFrame:
    """What the likeness is doing at time t on the speech timeline.

    Timed against speech rather than the wall clock. That is the decision the
    rest of the design rests on: the model produces a sentence before the voice
    synthesises it, and the voice synthesises it before it is rendered, so a
    nod that must peak on a stressed syllable can be started before that
    syllable is heard. Scheduled on a wall clock, every nod arrives late.
    """

    t: float = 0.0

    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0

    gaze_yaw: float = 0.0
    gaze_pitch: float = 0.0

    blink: float = 0.0
    lid_upper_l: float = 0.0
    lid_upper_r: float = 0.0

    brow_inner_l: float = 0.0
    brow_inner_r: float = 0.0
    brow_outer_l: float = 0.0
    brow_outer_r: float = 0.0

    jaw_open: float = 0.0
    mouth_smile_l: float = 0.0
    mouth_smile_r: float = 0.0
    mouth_press: float = 0.0

    torso_lean: float = 0.0
    torso_yaw: float = 0.0
    shoulder_raise: float = 0.0
    breath: float = 0.0

    visemes: tuple[float, ...] = field(default=(0.0,) * VISEME_COUNT)

    # What the body is doing, if anything. Named rather than numeric because
    # the renderer plays a montage for these and blends it over everything
    # above; weight is how much of it is showing.
    body_action: str | None = None
    body_action_weight: float = 0.0

    def to_array(self) -> np.ndarray:
        """The continuous channels, in declaration order.

        For the wire, and for tests. Everything that varies smoothly is here;
        body_action is deliberately not, because it is a discrete choice and
        interpolating between two of them is meaningless.
        """
        return np.array([getattr(self, c.name) for c in CHANNELS], dtype=np.float32)

    def clamped(self) -> PoseFrame:
        """A copy with every channel inside its declared range.

        The last thing before the wire, not a substitute for producing correct
        values. Anything relying on this to be correct is already wrong.
        """
        values = {c.name: min(c.high, max(c.low, getattr(self, c.name))) for c in CHANNELS}

        visemes = tuple(max(0.0, v) for v in self.visemes)
        total = sum(visemes)
        if total > MAX_VISEME_SUM:
            visemes = tuple(v * MAX_VISEME_SUM / total for v in visemes)

        return PoseFrame(
            t=self.t,
            visemes=visemes,
            body_action=self.body_action,
            body_action_weight=min(1.0, max(0.0, self.body_action_weight)),
            **values,
        )

    def out_of_range(self) -> list[str]:
        """Which channels are outside their bounds. Empty when correct."""
        return [
            c.name
            for c in CHANNELS
            if not (c.low - 1e-6 <= getattr(self, c.name) <= c.high + 1e-6)
        ]


def channel_names() -> list[str]:
    return [c.name for c in CHANNELS]


assert len(CHANNELS) == 20, "channel table and PoseFrame have diverged"
assert {c.name for c in CHANNELS} <= {f.name for f in fields(PoseFrame)}
