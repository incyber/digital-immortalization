"""Face geometry client.

Detection itself is exercised against the running service in the end-to-end
suite; these pin the parsing and the derived properties, which are where a
silent mistake would be invisible.
"""

import numpy as np
import pytest

from avatar.vision.faceclient import Face, FaceClient, FaceServiceError, NoFace, _parse


def a_payload(**over):
    base = {
        "bbox": [100, 120, 300, 340],
        "mouth_box": [180, 330, 140, 80],
        "mouth_openness": 0.07,
        "yaw": -0.05,
        "landmarks": [[float(i), float(i * 2)] for i in range(478)],
    }
    base.update(over)
    return base


def test_a_face_parses_into_pixel_geometry():
    face = _parse(a_payload())
    assert face.bbox == (100, 120, 300, 340)
    assert face.mouth_box == (180, 330, 140, 80)
    assert face.landmarks.shape == (478, 2)


def test_area_comes_from_the_box():
    assert _parse(a_payload()).area == 300 * 340


@pytest.mark.parametrize("yaw,frontal", [(0.0, True), (-0.2, True), (0.3, False), (-0.78, False)])
def test_frontality_follows_yaw(yaw, frontal):
    # A base frame turned away looks wrong the moment it speaks, so this is
    # what ranks candidates.
    assert _parse(a_payload(yaw=yaw)).is_frontal is frontal


def test_a_missing_mask_is_absent_not_fabricated():
    assert _parse(a_payload()).mask is None


def test_a_corrupt_mask_does_not_break_parsing():
    # A mask that fails to decode must degrade, not raise: it is an
    # enhancement, and the geometry is still usable without it.
    face = _parse(a_payload(mask_png_b64="bm90LWFuLWltYWdl"))
    assert face.mask is None
    assert face.bbox == (100, 120, 300, 340)


async def test_an_unreachable_service_raises_clearly():
    client = FaceClient("http://localhost:9")
    with pytest.raises(FaceServiceError, match="unreachable"):
        await client.detect_all(b"not-really-an-image")


async def test_health_is_false_when_unreachable():
    assert await FaceClient("http://localhost:9").healthy() is False


def test_no_face_is_its_own_error():
    # Callers distinguish "the service is down" from "there is nobody in this
    # photograph"; conflating them tells the customer to fix the wrong thing.
    assert issubclass(NoFace, ValueError)
    assert not issubclass(NoFace, FaceServiceError)


def test_landmark_count_is_the_full_mesh():
    face = Face(
        bbox=(0, 0, 10, 10),
        mouth_box=(1, 1, 2, 2),
        mouth_openness=0.0,
        yaw=0.0,
        landmarks=np.zeros((478, 2), np.float32),
    )
    assert len(face.landmarks) == 478
