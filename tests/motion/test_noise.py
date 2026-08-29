"""Proof that the motion does not loop.

The requirement was stated plainly: "It should be dynamically changing based on
emotions, content the avatar is saying, gestures, etc. It should behave like a
human and not like a loop video."

That is testable, and this is the test. Autocorrelation over ten minutes finds
any period a viewer would eventually notice - a looping template shows a hard
spike at its own length, and this must not. Everything else here defends the
properties that keep the drift usable: bounded so the head never wanders off,
reproducible so a complaint about how it moved can be replayed.
"""

import numpy as np
import pytest

from avatar.motion.noise import (
    BLINK_REFRACTORY_S,
    MIN_FIXATION_S,
    BlinkScheduler,
    PinkDrift,
    SaccadeScheduler,
)

FPS = 25


def trace(drift: PinkDrift, seconds: float, fps: int = FPS) -> np.ndarray:
    return np.array([drift.advance_to(i / fps) for i in range(int(seconds * fps))])


def autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Normalised by the overlap at each lag, not by the sample count.

    The naive form divides every lag by N, which tapers the result towards zero
    as the lag grows purely because fewer samples overlap. That taper would
    flatter this test: a real repeat at a long lag would be scaled down and
    might slip under the threshold.
    """
    signal = signal - signal.mean()
    n = len(signal)
    correlation = np.correlate(signal, signal, mode="full")[n - 1:]
    overlap = np.arange(n, 0, -1)
    correlation = correlation / overlap
    return correlation / correlation[0]


def test_the_drift_has_no_period_a_viewer_would_notice():
    """The test that means 'not a loop video'.

    Correlation at short lags is high and should be - that is what smooth
    motion is. What betrays a loop is correlation coming *back* long after it
    has decayed, which is what a repeating clip does at its own length. So the
    window starts past the slowest component of the drift.
    """
    correlation = autocorrelation(trace(PinkDrift(seed=1), seconds=600))

    lags = slice(int(60 * FPS), int(300 * FPS))
    peak = correlation[lags].max()
    assert peak < 0.35, f"repeats at lag {(correlation[lags].argmax() / FPS) + 60:.0f}s"


def test_the_same_test_catches_a_looping_signal():
    """A test that nothing fails is not evidence of anything.

    The 2D path builds idle motion from whole-number sine cycles so a clip
    loops seamlessly. That is exactly the property being rejected here, so it
    is worth proving the measurement can see it.
    """
    t = np.arange(600 * FPS) / FPS
    looping = np.sin(2 * np.pi * t / 8.0)

    correlation = autocorrelation(looping)
    lags = slice(int(60 * FPS), int(300 * FPS))

    assert correlation[lags].max() > 0.9


def test_the_drift_stays_bounded():
    """A random walk that wanders off eventually points the head at the floor."""
    signal = trace(PinkDrift(seed=2), seconds=600)

    assert np.abs(signal).max() < 4.0
    assert abs(signal.mean()) < 0.5


def test_the_same_session_moves_the_same_way_twice():
    """So that 'it moved oddly at 40 seconds' can be looked at again."""
    assert np.allclose(trace(PinkDrift(seed=7), 20), trace(PinkDrift(seed=7), 20))


def test_two_sessions_do_not_move_alike():
    a = trace(PinkDrift(seed=1), 60)
    b = trace(PinkDrift(seed=2), 60)

    correlation = np.corrcoef(a, b)[0, 1]
    assert abs(correlation) < 0.2


def test_sampling_more_often_does_not_change_the_motion():
    """The walk advances in fixed steps, so asking twice per frame is free."""
    drift = PinkDrift(seed=3)
    dense = [drift.advance_to(i / 50) for i in range(500)]

    sparse = PinkDrift(seed=3)
    coarse = [sparse.advance_to(i / 25) for i in range(250)]

    # The extra samples fall between steps and must not perturb the values at
    # the times both series share.
    assert np.allclose(dense[0::2], coarse, atol=1e-6)


# --------------------------------------------------------------------------


def test_blinks_are_irregular_not_metronomic():
    blinks = BlinkScheduler(seed=4).blinks_until(600.0)
    gaps = np.diff([b.t for b in blinks])

    assert len(blinks) > 100
    # A jittered metronome has a tight spread; unrelated events do not.
    assert gaps.std() / gaps.mean() > 0.5


def test_no_two_blinks_closer_than_the_refractory_period():
    """Double blinks read as a glitch, not as a face."""
    blinks = BlinkScheduler(seed=5).blinks_until(600.0)
    gaps = np.diff([b.t for b in blinks])

    assert gaps.min() >= BLINK_REFRACTORY_S - 1e-9


def test_the_blink_rate_is_about_what_was_asked_for():
    blinks = BlinkScheduler(seed=6, rate_per_min=14.0).blinks_until(600.0)

    assert 10 < len(blinks) / 10.0 < 19


def test_a_forced_blink_is_refused_when_one_just_happened():
    scheduler = BlinkScheduler(seed=8)
    scheduler.blinks_until(10.0)

    assert scheduler.force(100.0) is not None
    assert scheduler.force(100.05) is None


def test_gaze_holds_still_between_jumps():
    """Eyes that never rest look like they are searching for something."""
    saccades = SaccadeScheduler(seed=9).saccades_until(600.0)
    gaps = np.diff([s.t for s in saccades])

    assert len(saccades) > 50
    assert gaps.min() >= MIN_FIXATION_S - 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_saccades_stay_within_a_plausible_range(seed):
    saccades = SaccadeScheduler(seed=seed).saccades_until(300.0)

    assert all(abs(s.yaw) < 0.6 for s in saccades)
    assert all(abs(s.pitch) < 0.5 for s in saccades)
