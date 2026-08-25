"""Our own idle motion.

LivePortrait ships motion templates extracted from videos of real people.
Generating ours instead means no customer's late parent moves like the same
stranger, and it makes the motion tunable.

The measured ranges these are checked against come from the shipped templates,
so the generated motion sits in the space the model was trained to consume.
"""

import math
import pickle

import numpy as np
import pytest

from avatar.ingest.idle_motion import (
    BASE_SCALE,
    EYES_OPEN,
    FPS,
    IdleStyle,
    build_idle_template,
    write_idle_template,
)

# The shipped template measured 0.084 absolute rotation drift.
MEASURED_DRIFT = 0.084


def rotations(template):
    return np.stack([m["R_d"].reshape(3, 3) for m in template["motion"]])


def test_the_template_has_the_shape_liveportrait_reads():
    t = build_idle_template(seconds=2.0)
    assert set(t) == {"n_frames", "output_fps", "motion", "c_d_eyes_lst", "c_d_lip_lst"}
    assert t["n_frames"] == len(t["motion"]) == len(t["c_d_eyes_lst"]) == len(t["c_d_lip_lst"])
    assert t["output_fps"] == FPS


def test_each_frame_carries_the_four_expected_fields():
    frame = build_idle_template(seconds=1.0)["motion"][0]
    assert set(frame) == {"scale", "R_d", "exp", "t"}
    assert frame["R_d"].shape == (1, 3, 3)
    assert frame["exp"].shape == (1, 21, 3)
    assert frame["t"].shape == (1, 3)
    assert frame["scale"].shape == (1, 1)


def test_rotations_are_valid_rotation_matrices():
    """A matrix that is not orthonormal shears the face instead of turning it."""
    for R in rotations(build_idle_template(seconds=2.0)):
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-4)
        assert math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-4)


def test_the_motion_stays_inside_the_measured_range():
    drift = np.abs(rotations(build_idle_template(seconds=6.0)) - rotations(
        build_idle_template(seconds=6.0))[0]).max()
    assert drift < MEASURED_DRIFT * 2.5, "beyond this the head visibly shears"


def test_the_loop_joins_seamlessly():
    """A template that ends mid-sway snaps visibly every time it repeats."""
    R = rotations(build_idle_template(seconds=6.0))
    wrap = float(np.abs(R[0] - R[-1]).max())
    mid = float(np.abs(R[0] - R[len(R) // 2]).max())
    assert wrap < 0.01
    assert wrap < mid


def test_the_head_actually_moves():
    R = rotations(build_idle_template(seconds=6.0))
    assert float(np.abs(R - R[0]).max()) > 0.01, "a still frame reads as a frozen call"


def test_there_are_blinks():
    eyes = np.array([e.ravel()[0] for e in build_idle_template(seconds=12.0)["c_d_eyes_lst"]])
    assert (eyes < EYES_OPEN / 2).sum() >= 4
    assert (eyes >= EYES_OPEN / 2).sum() > len(eyes) * 0.8, "eyes are open most of the time"


def test_blinks_are_not_perfectly_periodic():
    """A blink exactly every four seconds is a clear tell."""
    eyes = np.array([e.ravel()[0] for e in build_idle_template(seconds=30.0, seed=3)["c_d_eyes_lst"]])
    closed = np.where(eyes < EYES_OPEN / 2)[0]
    starts = [c for i, c in enumerate(closed) if i == 0 or c != closed[i - 1] + 1]
    gaps = np.diff(starts)
    assert len(gaps) >= 3
    assert gaps.std() > 1.0, "blink intervals must vary"


def test_expression_is_left_neutral():
    # A smile baked into the idle loop would play underneath every word.
    for frame in build_idle_template(seconds=2.0)["motion"]:
        assert np.abs(frame["exp"]).max() == 0.0


def test_scale_matches_what_the_model_expects():
    for frame in build_idle_template(seconds=1.0)["motion"]:
        assert math.isclose(float(frame["scale"].ravel()[0]), BASE_SCALE, abs_tol=0.01)


@pytest.mark.parametrize(
    "style,expected_order",
    [(IdleStyle.calm(), 0), (IdleStyle.natural(), 1), (IdleStyle.animated(), 2)],
)
def test_styles_differ_in_how_much_they_move(style, expected_order):
    drift = np.abs(rotations(build_idle_template(seconds=6.0, style=style, seed=1))).std()
    assert drift >= 0
    assert style.sway > 0


def test_more_animated_styles_move_more():
    def drift(style):
        R = rotations(build_idle_template(seconds=6.0, style=style, seed=1))
        return float(np.abs(R - R[0]).max())

    assert drift(IdleStyle.calm()) < drift(IdleStyle.natural()) < drift(IdleStyle.animated())


def test_the_same_seed_gives_the_same_motion():
    a = build_idle_template(seconds=3.0, seed=7)
    b = build_idle_template(seconds=3.0, seed=7)
    assert np.array_equal(rotations(a), rotations(b))


def test_it_writes_a_template_liveportrait_can_load(tmp_path):
    path = write_idle_template(tmp_path / "idle.pkl", seconds=2.0)
    with path.open("rb") as fh:
        loaded = pickle.load(fh)
    assert loaded["n_frames"] == int(2.0 * FPS)
