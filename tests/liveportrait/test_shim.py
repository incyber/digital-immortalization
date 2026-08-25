"""The InsightFace replacement.

LivePortrait's LICENSE requires the InsightFace detection models to be removed
for commercial use. These pin the nine landmark indices that decide the crop,
because getting one wrong misaligns every generated frame in a way that looks
like a bad model rather than a bad index.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path("infra/liveportrait").resolve()))

from facegeom_shim import (  # noqa: E402
    IF_LEFT_EYE,
    IF_LOWER_LIP,
    IF_RIGHT_EYE,
    IF_UPPER_LIP,
    MP_LEFT_EYE,
    MP_LOWER_LIP,
    MP_RIGHT_EYE,
    MP_UPPER_LIP,
    _to_106,
)


def a_mesh(n=478):
    """A mesh where every landmark is identifiable by its index."""
    return np.array([[float(i), float(i * 2)] for i in range(n)], dtype=np.float32)


def test_the_output_is_the_shape_liveportrait_expects():
    assert _to_106(a_mesh()).shape == (106, 2)


def test_left_eye_slots_carry_the_left_eye():
    out, mesh = _to_106(a_mesh()), a_mesh()
    for slot, source in zip(IF_LEFT_EYE, MP_LEFT_EYE, strict=True):
        assert tuple(out[slot]) == tuple(mesh[source])


def test_right_eye_slots_carry_the_right_eye():
    out, mesh = _to_106(a_mesh()), a_mesh()
    for slot, source in zip(IF_RIGHT_EYE, MP_RIGHT_EYE, strict=True):
        assert tuple(out[slot]) == tuple(mesh[source])


def test_lip_slots_carry_the_lips():
    out, mesh = _to_106(a_mesh()), a_mesh()
    assert tuple(out[IF_UPPER_LIP]) == tuple(mesh[MP_UPPER_LIP])
    assert tuple(out[IF_LOWER_LIP]) == tuple(mesh[MP_LOWER_LIP])


def test_the_eyes_are_not_the_same_point():
    # If both eye slots resolved to the same landmark the similarity transform
    # degenerates and every frame comes out rotated.
    out = _to_106(a_mesh())
    left = out[list(IF_LEFT_EYE)].mean(axis=0)
    right = out[list(IF_RIGHT_EYE)].mean(axis=0)
    assert not np.allclose(left, right)


def test_no_slot_is_left_at_the_origin():
    # A zero row drags any extent computed over the array towards (0, 0).
    out = _to_106(a_mesh())
    assert not (out == 0).all(axis=1).any()


def test_the_parser_liveportrait_uses_reads_sensible_points():
    """Reproduces parse_pt2_from_pt106 exactly, on a realistic face."""
    mesh = np.zeros((478, 2), dtype=np.float32)
    # A face roughly 200px wide with eyes above the mouth.
    for i in MP_LEFT_EYE:
        mesh[i] = (80.0, 100.0)
    for i in MP_RIGHT_EYE:
        mesh[i] = (220.0, 100.0)
    mesh[MP_UPPER_LIP] = (150.0, 200.0)
    mesh[MP_LOWER_LIP] = (150.0, 210.0)
    from facegeom_shim import MP_FACE_OVAL

    for n, i in enumerate(MP_FACE_OVAL):
        mesh[i] = (60.0 + n * 4, 60.0 + n * 4)

    out = _to_106(mesh)

    left_eye = out[list(IF_LEFT_EYE)].mean(axis=0)
    right_eye = out[list(IF_RIGHT_EYE)].mean(axis=0)
    lip_centre = (out[IF_UPPER_LIP] + out[IF_LOWER_LIP]) / 2

    assert left_eye[0] < right_eye[0], "left eye must be left of the right eye"
    assert lip_centre[1] > left_eye[1], "the mouth must sit below the eyes"


def test_a_short_mesh_is_rejected_rather_than_silently_wrong():
    with pytest.raises(IndexError):
        _to_106(a_mesh(50))
