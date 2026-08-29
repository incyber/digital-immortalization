"""What the body does, and when - generated per instance, never replayed.

The requirement this serves is the same one noise.py serves, one level up:
"It should be dynamically changing based on emotions, content the avatar is
saying, gestures, etc. It should behave like a human and not like a loop
video."

The obvious reading of that is variety of choice - pick a different gesture
each time and it will not look like a loop. That reading is wrong, and it is
worth being explicit about why, because it decides the shape of this module.

A likeness with twelve nod clips picked at random still reads as a machine
inside a minute. The eye does not compare *which* gesture to the last one; it
compares *this* nod to every nod it has already seen, and a replayed clip is
identical to the frame. Twelve clips is twelve loops, not none. So no gesture
here is a stored curve. nod_small is a family - amplitude, beat count, left and
right asymmetry, rise, decay all drawn per instance - and two nods in one call
are never the same nod. That property is the deliverable; the scheduler around
it is only what keeps the family from being used badly.

Two decisions carry the rest of it.

Timing is not negotiable. Every event is scheduled at onset = anchor - rise_s,
so the apex of the movement lands on the syllable it belongs to. If that onset
is already in the past the event is dropped, never compressed and never started
late: a nod that arrives after the stressed word is not a late nod, it is a
different and wrong gesture, and it reads worse than stillness. Dropping is
cheap only because it is possible at all - the speech timeline exists before
the audio does, which is the whole reason PoseFrame is timed against speech
rather than the clock.

Restraint is the other. The budget below is deliberately mean: one body action
per two and a half seconds, three per utterance. A likeness that gestures on
every clause is read as a puppet, and read that way faster than one that
gestures rarely. Under-gesturing costs nothing an audience can name.

Seeded from a session id, so two calls differ and one call replays exactly.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

# Curves are stored on a grid rather than as a closure so a director can
# resample them at any frame rate without re-deriving anything, and so a test
# can look at the values. 400Hz is far above any rate a face is rendered at;
# the cost is a few hundred floats per event.
SAMPLE_DT = 1.0 / 400.0

# Every instance of a gesture draws its own shape inside these bounds. This is
# where "not a loop video" actually lives - not in which gesture is chosen.
AMPLITUDE_RANGE = (0.6, 1.4)

# Applied to the spec's nominal rise. For nod_small this spans 110-180ms, which
# is the measured range of a human nod from onset to apex.
RISE_JITTER = (0.76, 1.24)

# Decay as a multiple of rise. Always slower than the rise: movements that fall
# as fast as they rise look struck rather than made.
DECAY_RATIO = (1.2, 2.0)

# How much smaller each repeat is than the one before. A second nod is never as
# big as the first, and three equal beats is the single clearest tell of a
# looped clip.
BEAT_FALLOFF = (0.5, 0.8)

# Gap to the next beat, as a multiple of the decay. At or above 0.82 the
# previous beat has returned to rest before the next one peaks, which keeps the
# first apex the largest - the scheduler has promised that apex to a syllable.
BEAT_GAP = (0.82, 1.05)

# Left/right asymmetry as a fraction of the main channel. A nod that is purely
# sagittal is a hinge, not a head.
ASYMMETRY_MAX = 0.2

# Multiplier for an affect label a spec says nothing about.
DEFAULT_AFFINITY = 1.0

# How far back the anti-repetition memory reaches, and how hard it bites. Each
# prior use in the window multiplies the weight by this, so a third use in
# twelve is worth an eighth of a first.
RECENCY_WINDOW = 12
RECENCY_DECAY = 0.35

# A sequence of three gestures that has already happened twice inside living
# memory is a pattern, and the third occurrence is where a viewer starts to
# predict the body. The recency weighting above cannot see this: it counts
# single gestures, and a loop is an order, not a tally. The memory is bounded
# because two identical triples an hour apart are not a repeat anyone can see,
# and an unbounded ledger would eventually leave nothing legal to choose.
PATTERN_MEMORY = 200
PATTERN_LIMIT = 2

# The budget. Both numbers are lower than they feel like they should be, which
# is the point - see the module docstring.
MIN_GAP_S = 2.5
MAX_PER_UTTERANCE = 3

# A standing candidate for doing nothing, weighted against the real ones. The
# budget caps how much can happen; this is what keeps the body from spending
# its whole allowance every time it is offered.
SILENCE_WEIGHT = 0.9


def _smootherstep(u: np.ndarray) -> np.ndarray:
    """Zero value and zero slope at both ends, unlike a raised cosine's cousin.

    The second derivative also vanishes at the ends, which matters here because
    the eye reads the onset of an acceleration, not the acceleration.
    """
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _pulse(times: np.ndarray, offset: float, rise: float, decay: float) -> np.ndarray:
    """One rise-and-fall, zero and flat outside [offset, offset + rise + decay]."""
    out = np.zeros_like(times)

    rising = (times - offset) / rise
    up = (rising >= 0.0) & (rising <= 1.0)
    out[up] = _smootherstep(rising[up])

    falling = (times - offset - rise) / decay
    down = (falling > 0.0) & (falling <= 1.0)
    out[down] = 1.0 - _smootherstep(falling[down])

    return out


@dataclass(frozen=True)
class Curve:
    """A sampled movement over one or more pose channels.

    Evaluated by cubic Hermite between samples rather than linearly, because
    linear interpolation has a slope discontinuity at every knot and the eye
    finds those in exactly the places this module cares about - the first
    twentieth of a second of a movement.

    The end tangents are held at zero, and the envelope is built to start and
    end at zero, so the curve joins the surrounding stillness with no step in
    value and none in velocity. A step in either is a visible cut.
    """

    channels: Mapping[str, np.ndarray]
    dt: float
    # Onset to perceptual apex. The scheduler subtracts this from the anchor,
    # so it is a promise about where the largest value falls.
    rise_s: float

    @property
    def duration(self) -> float:
        return (self._length - 1) * self.dt

    @property
    def _length(self) -> int:
        return len(next(iter(self.channels.values())))

    def sample(self, name: str, times: np.ndarray) -> np.ndarray:
        """The channel at each time. Zero outside the window."""
        values = self.channels[name]
        n = len(values)
        times = np.asarray(times, dtype=float)
        out = np.zeros(times.shape, dtype=float)

        position = times / self.dt
        inside = (position > 0.0) & (position < n - 1)
        if not inside.any():
            return out

        # Catmull-Rom tangents, in units per sample. The ends stay at zero:
        # that is what makes the join to stillness smooth in velocity.
        tangents = np.zeros(n)
        tangents[1:-1] = (values[2:] - values[:-2]) / 2.0

        index = np.floor(position[inside]).astype(int)
        f = position[inside] - index
        f2 = f * f
        f3 = f2 * f

        out[inside] = (
            (2.0 * f3 - 3.0 * f2 + 1.0) * values[index]
            + (f3 - 2.0 * f2 + f) * tangents[index]
            + (-2.0 * f3 + 3.0 * f2) * values[index + 1]
            + (f3 - f2) * tangents[index + 1]
        )
        return out

    def at(self, t: float) -> dict[str, float]:
        """Every channel at one time, for a director filling a PoseFrame."""
        times = np.array([t])
        return {name: float(self.sample(name, times)[0]) for name in self.channels}

    def peak(self, name: str) -> float:
        return float(np.abs(self.channels[name]).max())


@dataclass(frozen=True)
class _Shape:
    """The drawn parameters of one instance, before it is given channels."""

    rise_s: float
    decay_s: float
    amplitudes: tuple[float, ...]
    offsets: tuple[float, ...]


def _draw_shape(
    spec: GestureSpec,
    rng: np.random.Generator,
    counts: tuple[int, int],
    alternate: bool = False,
) -> _Shape:
    """Draw one instance. Every number here is why two nods are not one nod."""
    rise = spec.rise_s * rng.uniform(*RISE_JITTER)
    decay = rise * rng.uniform(*DECAY_RATIO)
    count = int(rng.integers(counts[0], counts[1] + 1))

    amplitude = rng.uniform(*AMPLITUDE_RANGE)
    sign = 1.0
    offset = 0.0
    amplitudes: list[float] = []
    offsets: list[float] = []

    for _ in range(count):
        amplitudes.append(amplitude * sign)
        offsets.append(offset)
        amplitude *= rng.uniform(*BEAT_FALLOFF)
        offset += rise + decay * rng.uniform(*BEAT_GAP)
        if alternate:
            sign = -sign

    return _Shape(rise, decay, tuple(amplitudes), tuple(offsets))


def _render(shape: _Shape, gains: Mapping[str, float]) -> Curve:
    """Turn a drawn shape into channel values.

    The grid is built with linspace so the last sample lands exactly on the end
    of the decay. Off by a sample and the curve ends on a small non-zero value,
    which is a step to rest - the visible cut this class exists to avoid.
    """
    duration = shape.offsets[-1] + shape.rise_s + shape.decay_s
    length = max(3, round(duration / SAMPLE_DT) + 1)
    times = np.linspace(0.0, duration, length)

    envelope = np.zeros(length)
    for offset, amplitude in zip(shape.offsets, shape.amplitudes, strict=True):
        envelope += amplitude * _pulse(times, offset, shape.rise_s, shape.decay_s)

    return Curve(
        channels={name: envelope * gain for name, gain in gains.items()},
        dt=duration / (length - 1),
        rise_s=shape.rise_s,
    )


@dataclass(frozen=True)
class GestureSpec:
    """One gesture as a family of movements, not as a movement.

    generate is the whole point of the type: it is called per instance and must
    return a different curve each time. Anything that returns a stored curve
    from here reintroduces the loop this module was written to remove.
    """

    name: str
    region: str
    base_weight: float
    cooldown_s: float
    rise_s: float
    affinity: Mapping[str, float]
    generate: Callable[[GestureSpec, np.random.Generator], Curve]

    def affinity_for(self, affect: str) -> float:
        return self.affinity.get(affect, DEFAULT_AFFINITY)


# Peak angles in radians, or in channel units for the body. Sized so that the
# largest amplitude draw of the largest beat still sits inside the ranges
# declared in pose.py - a gesture that needs clamping is a gesture that was
# specified wrong.
NOD_SMALL_RAD = 0.045
NOD_DEEP_RAD = 0.10
SHAKE_RAD = 0.07
SHRUG_UNITS = 0.35
LEAN_UNITS = 0.32
BEAT_UNITS = 0.06


def _nod(spec: GestureSpec, rng: np.random.Generator, pitch: float, counts: tuple[int, int]):
    shape = _draw_shape(spec, rng, counts)
    asymmetry = rng.uniform(-ASYMMETRY_MAX, ASYMMETRY_MAX)
    return _render(
        shape,
        {
            "head_pitch": pitch,
            # A real nod is slightly off-axis and each person is off-axis in
            # their own direction; drawn per instance so it is not a signature.
            "head_roll": pitch * asymmetry,
            "head_yaw": pitch * asymmetry * 0.5,
        },
    )


def _generate_nod_small(spec: GestureSpec, rng: np.random.Generator) -> Curve:
    return _nod(spec, rng, NOD_SMALL_RAD, (1, 3))


def _generate_nod_deep(spec: GestureSpec, rng: np.random.Generator) -> Curve:
    return _nod(spec, rng, NOD_DEEP_RAD, (1, 2))


def _generate_head_shake(spec: GestureSpec, rng: np.random.Generator) -> Curve:
    # Alternating, because a shake is a reversal and not a repetition. One beat
    # is not a shake, so the count starts at two.
    shape = _draw_shape(spec, rng, (2, 3), alternate=True)
    asymmetry = rng.uniform(-ASYMMETRY_MAX, ASYMMETRY_MAX)
    return _render(shape, {"head_yaw": SHAKE_RAD, "head_roll": SHAKE_RAD * asymmetry})


def _generate_shrug(spec: GestureSpec, rng: np.random.Generator) -> Curve:
    shape = _draw_shape(spec, rng, (1, 1))
    asymmetry = rng.uniform(-ASYMMETRY_MAX, ASYMMETRY_MAX)
    return _render(
        shape,
        {
            "shoulder_raise": SHRUG_UNITS,
            # The shoulders come up and the torso goes back a little with them.
            "torso_lean": -SHRUG_UNITS * 0.15,
            "torso_yaw": SHRUG_UNITS * asymmetry * 0.3,
        },
    )


def _generate_lean_in(spec: GestureSpec, rng: np.random.Generator) -> Curve:
    # A posture, not a beat: it rises over most of a second, holds through a
    # long decay, and returns. Held indefinitely it would stop reading as a
    # movement and start reading as the resting pose, so it returns.
    shape = _draw_shape(spec, rng, (1, 1))
    return _render(shape, {"torso_lean": LEAN_UNITS, "head_pitch": 0.03})


def _generate_settle_back(spec: GestureSpec, rng: np.random.Generator) -> Curve:
    shape = _draw_shape(spec, rng, (1, 1))
    return _render(shape, {"torso_lean": -LEAN_UNITS * 0.85, "head_pitch": -0.02})


def _generate_beat(spec: GestureSpec, rng: np.random.Generator) -> Curve:
    # The small stress-aligned movement that runs under speech. Cheap enough to
    # fire often, which is exactly why its cooldown is the shortest here and
    # its weight is not the largest.
    shape = _draw_shape(spec, rng, (1, 2))
    return _render(shape, {"shoulder_raise": BEAT_UNITS, "head_pitch": BEAT_UNITS * 0.4})


# Affinities are multipliers, not probabilities. They say which movements a
# mood makes more likely, and the absent labels default to 1.0 so a new affect
# label degrades to neutral rather than to silence.
GESTURES: dict[str, GestureSpec] = {
    spec.name: spec
    for spec in (
        GestureSpec(
            name="nod_small",
            region="head",
            base_weight=1.0,
            cooldown_s=4.0,
            rise_s=0.145,
            affinity={"amused": 0.9, "sad": 1.2, "emphatic": 1.0, "thoughtful": 1.1},
            generate=_generate_nod_small,
        ),
        GestureSpec(
            name="nod_deep",
            region="head",
            base_weight=0.5,
            cooldown_s=12.0,
            # Slower than the 110-180ms of a small nod, and visibly so: depth
            # and speed together is a flinch, not agreement.
            rise_s=0.19,
            affinity={"amused": 0.6, "sad": 1.5, "emphatic": 1.4, "thoughtful": 1.2},
            generate=_generate_nod_deep,
        ),
        GestureSpec(
            name="head_shake",
            region="head",
            base_weight=0.6,
            cooldown_s=8.0,
            rise_s=0.20,
            affinity={"amused": 0.7, "sad": 1.4, "emphatic": 1.2, "thoughtful": 0.9},
            generate=_generate_head_shake,
        ),
        GestureSpec(
            name="shrug",
            region="arms",
            base_weight=0.45,
            cooldown_s=25.0,
            rise_s=0.35,
            affinity={"amused": 1.6, "sad": 0.5, "emphatic": 0.8, "thoughtful": 1.3},
            generate=_generate_shrug,
        ),
        GestureSpec(
            name="lean_in",
            region="torso",
            base_weight=0.35,
            cooldown_s=40.0,
            rise_s=0.9,
            affinity={"amused": 1.3, "sad": 0.4, "emphatic": 1.5, "thoughtful": 0.8},
            generate=_generate_lean_in,
        ),
        GestureSpec(
            name="settle_back",
            region="torso",
            base_weight=0.35,
            cooldown_s=30.0,
            rise_s=0.7,
            affinity={"amused": 0.6, "sad": 1.4, "emphatic": 0.5, "thoughtful": 1.2},
            generate=_generate_settle_back,
        ),
        GestureSpec(
            name="beat",
            region="arms",
            base_weight=0.55,
            cooldown_s=1.2,
            rise_s=0.12,
            affinity={"amused": 1.2, "sad": 0.5, "emphatic": 1.8, "thoughtful": 0.7},
            generate=_generate_beat,
        ),
    )
}

REGIONS = ("head", "arms", "torso")


@dataclass(frozen=True)
class GestureEvent:
    """A chosen gesture, shaped, and pinned to the speech timeline."""

    name: str
    region: str
    anchor: float
    onset: float
    curve: Curve

    @property
    def end(self) -> float:
        return self.onset + self.curve.duration

    def at(self, t: float) -> dict[str, float]:
        """Channel values at a time on the speech timeline."""
        return self.curve.at(t - self.onset)


def _seed_from(session_id: str | int) -> int:
    """A stable seed from a session id.

    Deliberately not hash(): that is salted per process, so the same session
    would replay differently tomorrow and a complaint about how the body moved
    could never be looked at again.
    """
    digest = hashlib.blake2b(str(session_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


@dataclass
class GestureScheduler:
    """Chooses what fires and when, and refuses far more often than it agrees.

    The refusals are the interesting part. A gesture is discarded outright when
    it is inside its own cooldown, when its region is still occupied, when the
    budget is spent, or - the one that matters most - when the anchor is too
    close to now for the movement to reach its apex in time. That last case is
    counted rather than salvaged, because every salvage is a compressed or late
    gesture and both look worse than the stillness they replaced.
    """

    session_id: str | int
    specs: Mapping[str, GestureSpec] = field(default_factory=lambda: GESTURES)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(_seed_from(self.session_id))
        self._last_fired: dict[str, float] = {}
        self._recent: deque[str] = deque(maxlen=RECENCY_WINDOW)
        self._history: deque[str] = deque(maxlen=PATTERN_MEMORY)
        self._region_free_at: dict[str, float] = dict.fromkeys(REGIONS, -math.inf)
        self._last_anchor = -math.inf
        self._in_utterance = 0
        self.dropped = 0

    def begin_utterance(self) -> None:
        """Reset the per-utterance allowance. Cooldowns and regions persist."""
        self._in_utterance = 0

    def effective_weight(self, spec: GestureSpec, anchor: float, affect: str) -> float:
        """The full weighting rule, in one place so a test can read it."""
        weight = spec.base_weight * spec.affinity_for(affect)

        if anchor - self._last_fired.get(spec.name, -math.inf) < spec.cooldown_s:
            return 0.0

        weight *= RECENCY_DECAY ** self._recent.count(spec.name)

        # Nominal onset is enough for the mask; the real reservation below uses
        # the drawn curve, which can only end later than the nominal one.
        if anchor - spec.rise_s < self._region_free_at[spec.region]:
            return 0.0

        return weight

    def _would_repeat_a_pattern(self, name: str) -> bool:
        """Has this exact three-gesture sequence already happened twice?

        Kept out of effective_weight because it is a veto on an order rather
        than a weighting of a gesture, and because a candidate that survives
        everything else and dies here is worth being able to see separately.
        """
        if len(self._history) < 2:
            return False

        candidate = (self._history[-2], self._history[-1], name)
        history = tuple(self._history)
        seen = sum(
            1
            for i in range(len(history) - 2)
            if history[i : i + 3] == candidate
        )
        return seen >= PATTERN_LIMIT

    def _choose(self, anchor: float, affect: str) -> GestureSpec | None:
        names = list(self.specs)
        weights = np.array(
            [
                0.0
                if self._would_repeat_a_pattern(name)
                else self.effective_weight(self.specs[name], anchor, affect)
                for name in names
            ]
        )

        total = weights.sum() + SILENCE_WEIGHT
        draw = self._rng.uniform(0.0, total)
        cumulative = 0.0
        for name, weight in zip(names, weights, strict=True):
            cumulative += weight
            if draw < cumulative:
                return self.specs[name]
        return None

    def propose(
        self, anchor: float, affect: str = "neutral", now: float = 0.0
    ) -> GestureEvent | None:
        """A gesture whose apex lands on anchor, or nothing.

        now is where the render head is on the speech timeline. Because the
        timeline exists before the audio does, now is usually well behind the
        anchor and there is room; when there is not, the event is dropped.
        """
        if self._in_utterance >= MAX_PER_UTTERANCE:
            return None
        if anchor - self._last_anchor < MIN_GAP_S:
            return None

        spec = self._choose(anchor, affect)
        if spec is None:
            return None

        curve = spec.generate(spec, self._rng)
        onset = anchor - curve.rise_s

        # The one rule that is enforced mechanically rather than intended: no
        # compression, no late start, no exceptions.
        if onset < now - 1e-9:
            self.dropped += 1
            return None

        self._last_fired[spec.name] = anchor
        self._recent.append(spec.name)
        self._history.append(spec.name)
        self._region_free_at[spec.region] = onset + curve.duration
        self._last_anchor = anchor
        self._in_utterance += 1

        return GestureEvent(
            name=spec.name,
            region=spec.region,
            anchor=anchor,
            onset=onset,
            curve=curve,
        )


assert set(REGIONS) == {spec.region for spec in GESTURES.values()}
