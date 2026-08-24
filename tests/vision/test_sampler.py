"""The gate's rule is that BOTH conditions must hold. These tests pin each
half, including the cases where one permits and the other refuses."""
import numpy as np
import pytest

from avatar.vision.sampler import MotionGate

INTERVAL = 4.0
THRESHOLD = 6.0


@pytest.fixture
def gate():
    return MotionGate(interval_s=INTERVAL, threshold=THRESHOLD)


@pytest.fixture
def frame():
    return np.full((240, 320, 3), 100, dtype=np.uint8)


@pytest.fixture
def other_frame():
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def test_first_frame_always_sends(gate, frame):
    assert gate.should_send(frame, now=0.0)


def test_identical_frame_within_interval_is_dropped(gate, frame):
    gate.should_send(frame, 0.0)
    assert not gate.should_send(frame, 1.0)


def test_identical_frame_after_interval_is_still_dropped(gate, frame):
    # Interval permits, motion refuses. A still person costs nothing.
    gate.should_send(frame, 0.0)
    assert not gate.should_send(frame, 10.0)


def test_changed_frame_after_interval_sends(gate, frame, other_frame):
    gate.should_send(frame, 0.0)
    assert gate.should_send(other_frame, 10.0)


def test_changed_frame_within_interval_is_dropped(gate, frame, other_frame):
    # Motion permits, interval refuses. Rate limit outranks motion, which is
    # what stops continuous movement from producing a call per frame.
    gate.should_send(frame, 0.0)
    assert not gate.should_send(other_frame, 1.0)


def test_cost_ceiling_holds_under_constant_motion(gate, other_frame):
    # Ten minutes of a frame that changes every time. The interval alone must
    # bound the count.
    rng = np.random.default_rng(2)
    sent = 0
    for i in range(600):
        f = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        if gate.should_send(f, now=float(i)):
            sent += 1
    assert sent <= 600 / INTERVAL + 1


def test_force_bypasses_both_conditions(gate, frame):
    gate.should_send(frame, 0.0)
    gate.force(frame, 0.5)
    # force resets the clock, so the next ordinary check is rate limited again
    assert not gate.should_send(frame, 1.0)
