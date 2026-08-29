"""The pose contract.

Every channel has a declared range and a declared rate limit, and the whole
motion system is testable without a GPU precisely because its output is these
numbers. If the ranges are wrong nothing downstream can be right, so they are
asserted rather than assumed.
"""

import numpy as np
import pytest

from avatar.motion.pose import (
    BY_NAME,
    CHANNELS,
    MAX_VISEME_SUM,
    VISEME_COUNT,
    PoseFrame,
    channel_names,
)


def test_a_default_pose_is_neutral_and_in_range():
    assert PoseFrame().out_of_range() == []


def test_every_channel_is_declared_once():
    names = channel_names()
    assert len(names) == len(set(names)) == len(CHANNELS)


@pytest.mark.parametrize("channel", CHANNELS, ids=lambda c: c.name)
def test_every_channel_has_a_sane_range_and_rate(channel):
    assert channel.low < channel.high
    assert channel.slew > 0


def test_gaze_may_move_far_faster_than_the_head():
    """Real eyes are ballistic. Smoothing gaze is the most uncanny thing here."""
    assert BY_NAME["gaze_yaw"].slew > BY_NAME["head_yaw"].slew * 10


def test_out_of_range_is_reported_rather_than_silently_fixed():
    """A clamp that hides an error produces a face that is subtly wrong."""
    pose = PoseFrame(head_yaw=99.0, jaw_open=-3.0)

    assert set(pose.out_of_range()) == {"head_yaw", "jaw_open"}


def test_clamping_brings_everything_into_range():
    pose = PoseFrame(head_yaw=99.0, jaw_open=-3.0, blink=5.0).clamped()

    assert pose.out_of_range() == []
    assert pose.head_yaw == BY_NAME["head_yaw"].high
    assert pose.jaw_open == 0.0


def test_overlapping_visemes_are_scaled_rather_than_cut():
    """Adjacent mouth shapes overlap in transition; far above one and it tears."""
    pose = PoseFrame(visemes=(1.0,) * VISEME_COUNT).clamped()

    assert sum(pose.visemes) == pytest.approx(MAX_VISEME_SUM)
    assert len(pose.visemes) == VISEME_COUNT


def test_negative_visemes_are_removed():
    pose = PoseFrame(visemes=(-1.0, 0.5) + (0.0,) * (VISEME_COUNT - 2)).clamped()

    assert all(v >= 0 for v in pose.visemes)


def test_the_array_is_the_declared_channels_in_order():
    pose = PoseFrame(head_yaw=0.1, breath=0.5)
    array = pose.to_array()

    assert array.shape == (len(CHANNELS),)
    assert array.dtype == np.float32
    assert array[channel_names().index("head_yaw")] == pytest.approx(0.1)
    assert array[channel_names().index("breath")] == pytest.approx(0.5)


def test_a_body_action_is_not_interpolated():
    """It is a discrete choice; a value halfway between two montages is nothing."""
    pose = PoseFrame(body_action="nod_small", body_action_weight=1.4).clamped()

    assert pose.body_action == "nod_small"
    assert pose.body_action_weight == 1.0
    assert "body_action" not in channel_names()
