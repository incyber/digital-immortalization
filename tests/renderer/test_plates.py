import numpy as np
import pytest

from avatar.renderer.plates import (
    PLATE_COUNT,
    AvatarAssets,
    NoFaceDetected,
    detect_mouth_box,
    extract_plates,
    synthetic_assets,
)


def test_synthetic_assets_have_expected_shape():
    a = synthetic_assets(size=(256, 256), fps=25, seconds=1.0)
    assert len(a.idle_frames) == 25
    assert a.idle_frames[0].shape == (256, 256, 3)
    assert len(a.plates) == PLATE_COUNT


def test_mouth_box_lies_inside_the_frame():
    a = synthetic_assets(size=(256, 256))
    x, y, w, h = a.mouth_box
    assert x >= 0 and y >= 0
    assert x + w <= 256 and y + h <= 256


def test_plates_are_ordered_closed_to_open():
    # The renderer indexes plates by loudness, so the ordering is load-bearing:
    # brighter interior and more edge energy must increase with the index.
    a = synthetic_assets(size=(256, 256))
    energy = [float(np.abs(np.diff(p.astype(np.int16), axis=0)).mean()) for p in a.plates]
    assert energy[-1] > energy[0]


def test_save_load_round_trip(tmp_path):
    a = synthetic_assets(size=(128, 128), seconds=0.5)
    a.save(tmp_path)
    b = AvatarAssets.load(tmp_path)
    assert b.mouth_box == a.mouth_box
    assert b.fps == a.fps and b.size == a.size
    assert len(b.idle_frames) == len(a.idle_frames)
    assert np.array_equal(b.plates[3], a.plates[3])


def test_no_face_raises():
    rng = np.random.default_rng(0)
    noise = [rng.integers(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(6)]
    with pytest.raises(NoFaceDetected):
        detect_mouth_box(noise)


def test_extract_plates_returns_requested_count():
    frames = [f for f in synthetic_assets(size=(128, 128), seconds=1.0).idle_frames]
    plates = extract_plates(frames, (40, 70, 40, 16))
    assert len(plates) == PLATE_COUNT
    assert all(p.shape == (16, 40, 3) for p in plates)
