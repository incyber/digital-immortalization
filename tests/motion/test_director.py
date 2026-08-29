"""The director: what the face is doing, and whether it can be trusted.

Every test here runs on a laptop with no GPU, no renderer and no audio device,
because the director's output is numbers. That is the property that makes this
system reviewable at all - a change to how the likeness moves shows up as a
diff in curves rather than as a video somebody has to watch twice.

The tests that earn their place are the ones about continuity and about what
survives an interruption. A face that cuts is worse than a face that is still,
and a face that freezes when interrupted stops being a person.
"""

import time

import numpy as np
import pytest

from avatar.motion.affect import affect_from_label
from avatar.motion.director import MotionDirector
from avatar.motion.pose import CHANNELS, channel_names

FPS = 25


def director(seed: int = 1) -> MotionDirector:
    return MotionDirector(session_seed=seed, fps=FPS)


def test_every_channel_stays_in_range_over_a_long_session():
    """A value out of range is a bug here, not something to clamp downstream."""
    poses = [director().pose_at(i / FPS) for i in range(FPS * 60)]

    assert all(p.out_of_range() == [] for p in poses)


def test_nothing_is_ever_not_a_number():
    """One NaN reaching a rig freezes a face for the rest of the call."""
    trace = director().trace(60)

    assert np.isfinite(trace).all()


def test_the_same_session_moves_identically_twice():
    """So that 'it looked wrong at forty seconds' can be looked at again."""
    assert np.allclose(director(7).trace(20), director(7).trace(20))


def test_two_sessions_do_not_move_alike():
    a, b = director(1).trace(60), director(2).trace(60)
    head = channel_names().index("head_yaw")

    assert abs(np.corrcoef(a[:, head], b[:, head])[0, 1]) < 0.3


@pytest.mark.parametrize(
    "channel", [c for c in CHANNELS if c.name not in {"gaze_yaw", "gaze_pitch", "blink"}],
    ids=lambda c: c.name,
)
def test_no_channel_moves_faster_than_it_is_allowed_to(channel):
    """The rate limits are the difference between a face and a puppet."""
    trace = director().trace(60)
    index = channel_names().index(channel.name)

    per_frame = np.abs(np.diff(trace[:, index])) * FPS
    assert per_frame.max() <= channel.slew * 1.5


def test_the_head_never_cuts():
    """A linear join passes a speed test and still looks like a snap.

    Only the second difference catches it, which is why this test exists
    separately from the rate limits above.
    """
    trace = director().trace(120)
    index = channel_names().index("head_yaw")

    second = np.abs(np.diff(trace[:, index], n=2))
    assert second.max() < 0.02


def test_it_keeps_breathing_in_every_state():
    """The cheapest signal that something is alive, and its absence reads as a
    photograph."""
    from avatar.motion.director import Attitude

    for attitude in Attitude:
        d = director()
        d.set_attitude(attitude, 0.0)
        breath = d.trace(20)[:, channel_names().index("breath")]

        assert breath.std() > 0.1, f"{attitude} is not breathing"


def test_interrupting_is_effectively_instant():
    """Barge-in is a state write, not a cancellation to propagate."""
    d = director()
    d.trace(5)

    started = time.perf_counter()
    d.interrupt(5.0)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.001


def test_the_likeness_stays_alive_after_being_interrupted():
    """A statue is a visible failure, and a pose-range test would pass one."""
    d = director()
    d.trace(5)
    d.interrupt(5.0)

    after = np.array([d.pose_at(5.0 + i / FPS).to_array() for i in range(FPS * 3)])

    assert after.std(axis=0).max() > 0.01


def test_affect_changes_the_resting_face():
    """A sad person is not a neutral person plus sadness; they sit differently."""
    names = channel_names()
    sad, amused = director(3), director(3)
    sad.set_affect(affect_from_label("sad", 1.0, confidence=1.0, source="tag"))
    amused.set_affect(affect_from_label("amused", 1.0, confidence=1.0, source="tag"))

    a = sad.trace(10).mean(axis=0)
    b = amused.trace(10).mean(axis=0)

    assert a[names.index("mouth_smile_l")] < b[names.index("mouth_smile_l")]
    assert a[names.index("brow_inner_l")] > b[names.index("brow_inner_l")]


def test_listening_and_speaking_look_different():
    """Attention is mostly brow and lean, not motion."""
    from avatar.motion.director import Attitude

    names = channel_names()
    listening, speaking = director(4), director(4)
    listening.set_attitude(Attitude.LISTENING, 0.0)
    speaking.set_attitude(Attitude.SPEAKING, 0.0)

    a = listening.trace(10).mean(axis=0)
    b = speaking.trace(10).mean(axis=0)

    assert a[names.index("torso_lean")] > b[names.index("torso_lean")]


@pytest.mark.slow
def test_the_face_does_not_settle_into_a_rhythm():
    """The requirement, measured on the assembled director rather than one layer.

    Five minutes rather than ten. The drift layer is already held to the longer
    window in test_noise; what this checks is that assembling it with springs,
    posture and affect did not reintroduce a period, and that is visible well
    inside five.
    """
    trace = director(5).trace(300)
    signal = trace[:, channel_names().index("head_yaw")]
    signal = signal - signal.mean()

    n = len(signal)
    correlation = np.correlate(signal, signal, mode="full")[n - 1:] / np.arange(n, 0, -1)
    correlation /= correlation[0]

    lags = slice(int(30 * FPS), int(150 * FPS))
    assert correlation[lags].max() < 0.35
