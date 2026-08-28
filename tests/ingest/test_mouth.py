"""Mouth plates warped from a real mouth.

These test the warp itself rather than the pipeline: the bundled cascade does
not detect drawn faces, so anything routed through detection would prove
nothing. See test_validate for that finding.
"""

import itertools

import cv2
import numpy as np

from avatar.ingest.mouth import (
    MAX_JAW_DROP,
    _interior,
    _jaw_warp,
    build_mouth_plates,
    match_tone,
    measure_sharpness,
    mouth_box_for,
)


def a_mouth(w=192, h=128):
    """A crop with a lip line, skin above and below, and texture throughout."""
    rng = np.random.default_rng(3)
    crop = np.full((h, w, 3), (150, 170, 205), np.uint8)
    crop = np.clip(crop.astype(int) + rng.integers(-10, 10, (h, w, 3)), 0, 255).astype(np.uint8)
    # lips
    cv2.ellipse(crop, (w // 2, int(h * 0.46)), (int(w * 0.28), int(h * 0.10)),
                0, 0, 360, (120, 120, 190), -1)
    # the dark line where they meet
    cv2.line(crop, (int(w * 0.24), int(h * 0.46)), (int(w * 0.76), int(h * 0.46)),
             (70, 70, 110), 2)
    return crop


def test_the_first_plate_is_the_photograph_untouched():
    base = a_mouth()
    plates = build_mouth_plates(base, 6)
    assert np.array_equal(plates[0], base)


def test_plates_open_progressively():
    """The renderer indexes by loudness, so the ordering is load-bearing."""
    plates = build_mouth_plates(a_mouth(), 6)
    brightness = [float(p.mean()) for p in plates]
    assert all(b <= a + 1e-6 for a, b in itertools.pairwise(brightness))
    assert brightness[0] - brightness[-1] > 3.0


def test_the_requested_number_of_plates_is_returned():
    for count in (2, 4, 6, 8):
        assert len(build_mouth_plates(a_mouth(), count)) == count


def test_plates_keep_the_shape_of_the_source():
    base = a_mouth(160, 96)
    for plate in build_mouth_plates(base, 6):
        assert plate.shape == base.shape


def test_the_warp_leaves_the_top_of_the_crop_alone():
    """A jaw moves; the nose and upper lip do not."""
    base = a_mouth()
    warped = _jaw_warp(base, MAX_JAW_DROP)
    top = slice(0, int(base.shape[0] * 0.30))
    assert np.abs(base[top].astype(int) - warped[top].astype(int)).mean() < 2.0


def test_the_warp_moves_the_bottom_of_the_crop():
    base = a_mouth()
    warped = _jaw_warp(base, MAX_JAW_DROP)
    bottom = slice(int(base.shape[0] * 0.60), base.shape[0])
    assert np.abs(base[bottom].astype(int) - warped[bottom].astype(int)).mean() > 1.0


def test_no_displacement_leaves_the_crop_unchanged():
    base = a_mouth()
    assert np.abs(base.astype(int) - _jaw_warp(base, 0.0).astype(int)).mean() < 0.5


def test_the_interior_is_darker_than_the_face():
    """It is a mouth, not a hole cut to the background."""
    base = a_mouth()
    interior = _interior(base)
    assert interior.mean() < base.mean()


def test_the_interior_is_built_from_this_face_not_a_fixed_colour():
    """Two people in different light must not get the same mouth interior."""
    pale = a_mouth()
    dark = (a_mouth().astype(int) * 0.45).astype(np.uint8)
    assert abs(float(_interior(pale).mean()) - float(_interior(dark).mean())) > 5.0


def test_tone_matching_moves_a_patch_into_the_reference_lighting():
    warm = a_mouth()
    cool = np.clip(a_mouth().astype(int) + np.array([60, 0, -40]), 0, 255).astype(np.uint8)
    matched = match_tone(cool, warm)
    before = abs(float(cool.mean()) - float(warm.mean()))
    after = abs(float(matched.mean()) - float(warm.mean()))
    assert after < before


def test_sharpness_ranks_a_crisp_crop_above_a_blurred_one():
    crisp = a_mouth()
    soft = cv2.GaussianBlur(crisp, (21, 21), 0)
    assert measure_sharpness(crisp) > measure_sharpness(soft)


def test_the_mouth_box_sits_in_the_lower_half_of_the_face():
    face = (100, 100, 400, 400)
    mx, my, mw, mh = mouth_box_for(face)
    assert my > face[1] + face[3] * 0.5
    assert face[0] < mx < face[0] + face[2]
    assert mw < face[2] and mh < face[3]


def test_the_jaw_drop_is_bounded():
    # Past roughly a third the chin detaches and it stops reading as a jaw.
    assert 0.0 < MAX_JAW_DROP <= 0.40
