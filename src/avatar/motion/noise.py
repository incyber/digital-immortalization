"""The layer that makes it not a loop.

This is the direct answer to the requirement that the likeness must not behave
like a looping video. It is worth being precise about why a loop happens and
what removes it.

The 2D path generates idle motion from sine waves whose cycles divide exactly
into the clip length - see ingest/idle_motion.py - specifically so the clip
loops without a visible jump. For a video that property is essential. For a
live likeness it is the defect: anything periodic is eventually recognised as
periodic, and once seen it cannot be unseen.

So the motion here is not periodic at all. It is a sum of Ornstein-Uhlenbeck
processes - random walks pulled back towards zero - at several time constants.
Each is bounded and stationary, so the head never drifts away and never needs
resetting, and the sum has energy at every timescale from a few tenths of a
second to half a minute. It never repeats, because there is nothing to repeat.

Blinks and saccades are scheduled rather than oscillated for the same reason. A
person blinks at irregular intervals with a refractory period, and looks at
things in fast jumps with pauses between; both are event processes, and drawing
them as waves is what makes an animated face read as a machine.

Seeded per session, so two calls never move alike and one call can be replayed
exactly when something needs debugging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Time constants, in seconds, and their relative weight. The slow ones carry
# posture, the fast ones carry the small constant motion of a living head.
# Chosen to span from within a syllable to well beyond a sentence, so no single
# rate dominates and the result has no characteristic period.
TAUS = (0.4, 1.6, 6.4, 25.0)
AMPLITUDES = (1.0, 0.7, 0.45, 0.30)

# Blinks per minute at rest. Falls while speaking and rises around pauses,
# which the director adjusts; this is the baseline.
BLINK_RATE_PER_MIN = 14.0

# No two blinks closer than this. Physiological, and without it a random
# schedule produces double blinks that read as a glitch.
BLINK_REFRACTORY_S = 0.4

# How long a blink takes, closed to open.
BLINK_DURATION_S = 0.12

# Gaze holds still for at least this long between jumps. Shorter and the eyes
# look like they are searching rather than looking.
MIN_FIXATION_S = 0.25


class PinkDrift:
    """Bounded, aperiodic drift. The anti-loop primitive.

    Sampled at arbitrary times but advanced in fixed steps, so the sequence
    depends only on the seed and the step - not on how often it happened to be
    asked. Two runs of the same session produce the same motion; that is what
    makes a complaint about how it moved reproducible.
    """

    def __init__(self, seed: int = 0, step_s: float = 1.0 / 25.0):
        self._rng = np.random.default_rng(seed)
        self._step = step_s
        self._state = np.zeros(len(TAUS))
        self._t = 0.0
        # Chosen so each process has unit stationary variance, making the
        # weights above mean what they say.
        self._sigma = np.array([np.sqrt(2.0 / tau) for tau in TAUS])
        self._weights = np.array(AMPLITUDES) / sum(AMPLITUDES)

    def advance_to(self, t: float) -> float:
        """The value at t, advancing the walk as far as needed."""
        while self._t < t - 1e-9:
            noise = self._rng.standard_normal(len(TAUS))
            decay = -self._state / np.array(TAUS)
            self._state = self._state + decay * self._step + self._sigma * np.sqrt(
                self._step
            ) * noise
            self._t += self._step
        return float(self._state @ self._weights)


@dataclass
class Blink:
    t: float
    duration: float = BLINK_DURATION_S


class BlinkScheduler:
    """Blinks as events, at irregular intervals, with a refractory period.

    Intervals are drawn from an exponential distribution - the gaps between
    unrelated events - rather than jittered around a mean, because jitter
    around a mean still has a mean that becomes audible to the eye over a long
    call.
    """

    def __init__(self, seed: int = 0, rate_per_min: float = BLINK_RATE_PER_MIN):
        self._rng = np.random.default_rng(seed ^ 0x51E7)
        self._rate = rate_per_min
        self._last = -BLINK_REFRACTORY_S
        self._next = 0.0
        self._scheduled = False

    def _draw(self, after: float, rate_per_min: float) -> float:
        interval = self._rng.exponential(60.0 / max(1e-3, rate_per_min))
        return after + max(BLINK_REFRACTORY_S, interval)

    def blinks_until(self, t: float, rate_per_min: float | None = None) -> list[Blink]:
        """Every blink starting at or before t, since the last call."""
        rate = self._rate if rate_per_min is None else rate_per_min
        if not self._scheduled:
            self._next = self._draw(0.0, rate)
            self._scheduled = True

        out = []
        while self._next <= t:
            out.append(Blink(t=self._next))
            self._last = self._next
            self._next = self._draw(self._last, rate)
        return out

    def force(self, t: float) -> Blink | None:
        """Blink now, unless one just happened.

        Used where a person reliably blinks: at the end of a sentence, and on a
        large gaze shift. Those are strong human signals and worth spending an
        explicit event on rather than waiting for the schedule to coincide.
        """
        if t - self._last < BLINK_REFRACTORY_S:
            return None
        self._last = t
        self._next = self._draw(t, self._rate)
        return Blink(t=t)


@dataclass
class Saccade:
    t: float
    yaw: float
    pitch: float


class SaccadeScheduler:
    """Gaze as jumps and holds, never as a smooth sweep.

    Smoothly interpolated gaze is the most reliably uncanny thing an animated
    face can do: real eyes are still, then somewhere else, with nothing visible
    in between.
    """

    def __init__(self, seed: int = 0, spread: float = 0.12):
        self._rng = np.random.default_rng(seed ^ 0x5ACC)
        self._spread = spread
        self._next = 0.0
        self._scheduled = False

    def saccades_until(self, t: float, hold_s: float = 1.6) -> list[Saccade]:
        if not self._scheduled:
            self._next = self._rng.exponential(hold_s)
            self._scheduled = True

        out = []
        while self._next <= t:
            out.append(
                Saccade(
                    t=self._next,
                    yaw=float(self._rng.normal(0.0, self._spread)),
                    pitch=float(self._rng.normal(0.0, self._spread * 0.6)),
                )
            )
            self._next += max(MIN_FIXATION_S, self._rng.exponential(hold_s))
        return out
