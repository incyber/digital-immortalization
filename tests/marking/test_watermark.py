"""Every published frame must be identifiable as synthetic by a machine.

EU AI Act Article 50 has applied since 2 August 2026 and requires providers of
systems generating synthetic video to mark outputs machine-readably. The
persistent banner in the call UI covers the human-facing half only.

These tests pin the properties that make a mark useful: it survives the
pipeline, it is invisible, and its absence is detectable.
"""

import numpy as np
import pytest

from avatar.marking.watermark import (
    PAYLOAD_BYTES,
    detect,
    embed,
    psnr,
    rgb_to_rgba,
)

PAYLOAD = b"avtr" + bytes([1, 0, 0, 0])


@pytest.fixture
def frame():
    rng = np.random.default_rng(7)
    # Photographic-ish rather than flat: a constant frame is the easy case.
    base = rng.integers(60, 200, (256, 256, 3), dtype=np.uint8)
    return base


def test_payload_round_trips(frame):
    assert detect(embed(frame, PAYLOAD)) == PAYLOAD


def test_unmarked_frame_returns_none(frame):
    assert detect(frame) is None


def test_mark_is_imperceptible(frame):
    marked = embed(frame, PAYLOAD)
    # 45 dB is well above the ~40 dB threshold usually treated as visually
    # lossless. If a change makes the avatar look worse, it is not shippable.
    assert psnr(frame, marked) > 45.0


def test_mark_survives_rgba_conversion(frame):
    # The publisher expands RGB to RGBA before handing frames to LiveKit. A
    # mark that does not survive that never reaches a viewer.
    marked = embed(frame, PAYLOAD)
    rgba = rgb_to_rgba(marked)
    assert detect(rgba[:, :, :3]) == PAYLOAD


def test_payload_length_is_enforced(frame):
    with pytest.raises(ValueError, match="payload must be"):
        embed(frame, b"too-long-for-the-field")


def test_different_payloads_are_distinguishable(frame):
    a = embed(frame, b"avtr" + bytes([1, 0, 0, 0]))
    b = embed(frame, b"avtr" + bytes([2, 0, 0, 0]))
    assert detect(a) != detect(b)


def test_works_on_flat_frames(frame):
    # Idle frames of a dark background are nearly flat, and must still mark.
    flat = np.full((256, 256, 3), 18, dtype=np.uint8)
    assert detect(embed(flat, PAYLOAD)) == PAYLOAD


def test_works_on_saturated_frames():
    # Pure white would overflow a naive additive scheme.
    white = np.full((256, 256, 3), 255, dtype=np.uint8)
    assert detect(embed(white, PAYLOAD)) == PAYLOAD


def test_small_frames_are_rejected_rather_than_silently_unmarked():
    tiny = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        embed(tiny, PAYLOAD)


def test_payload_size_constant_matches_reality():
    assert len(PAYLOAD) == PAYLOAD_BYTES
