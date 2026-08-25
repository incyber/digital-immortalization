"""Building renderable assets from real photographs."""

import cv2
import numpy as np
import pytest

from avatar.ingest.assets import (
    FACE_TARGET,
    OUTPUT_SIZE,
    NoUsablePhoto,
    build_idle_loop,
    build_plates,
    choose_base_frame,
)
from avatar.ingest.validate import Framing
from avatar.renderer.plates import PLATE_COUNT


def a_face(size=520, sharp=True):
    """The face construction the validator tests already prove is detected."""
    frame = np.full((size, size), 210, np.uint8)
    cx, cy = size // 2, size // 2
    cv2.ellipse(frame, (cx, cy), (size // 4, int(size * 0.32)), 0, 0, 360, 150, -1)
    for side in (-1, 1):
        cv2.ellipse(frame, (cx + side * size // 11, cy - size // 11),
                    (size // 26, size // 40), 0, 0, 360, 40, -1)
    cv2.ellipse(frame, (cx, cy + size // 8), (size // 12, size // 34), 0, 0, 180, 60, 3)
    cv2.line(frame, (cx, cy - size // 30), (cx, cy + size // 22), 110, 2)
    if not sharp:
        frame = cv2.GaussianBlur(frame, (31, 31), 0)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def a_face_photo(sharp=True, seed=0):
    """A whole photograph: a detectable face on a larger background.

    The background is flat. Adding noise to it stops the cascade finding the
    face at all, which makes the fixture test the detector rather than the
    thing under test.
    """
    frame = np.full((1400, 1000, 3), 232, np.uint8)
    frame[100:620, 240:760] = a_face(520, sharp=sharp)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def test_no_face_anywhere_is_refused():
    rng = np.random.default_rng(1)
    noise = [cv2.imencode(".jpg", rng.integers(0, 255, (400, 400, 3), dtype=np.uint8))[1].tobytes()]
    with pytest.raises(NoUsablePhoto):
        choose_base_frame(noise)


def test_an_undecodable_file_is_skipped_not_fatal():
    with pytest.raises(NoUsablePhoto):
        choose_base_frame([b"not an image at all"])


def test_the_idle_loop_moves_but_only_slightly():
    """A frozen frame reads as a dropped call; a large drift reads as sliding."""
    base = np.dstack([np.full((256, 256), 128, np.uint8)] * 3)
    base[100:150, 100:150] = 40
    frames = build_idle_loop(base, fps=25, seconds=1.0)

    assert len(frames) == 25
    assert not np.array_equal(frames[0], frames[6]), "the loop must move"

    difference = np.abs(frames[0].astype(int) - frames[6].astype(int)).mean()
    assert difference < 25, "the drift must stay subtle"


def test_the_idle_loop_stays_within_a_small_envelope():
    """Every frame must stay close to the source, or it reads as sliding."""
    base = np.dstack([np.full((128, 128), 128, np.uint8)] * 3)
    base[40:60, 40:60] = 30
    frames = build_idle_loop(base, fps=25, seconds=1.0)
    worst = max(np.abs(f.astype(int) - base.astype(int)).mean() for f in frames)
    assert worst < 20.0


def test_plates_open_progressively():
    """The renderer indexes plates by loudness, so the order is load-bearing."""
    base = np.dstack([np.full((256, 256), 190, np.uint8)] * 3)
    plates = build_plates(base, (100, 140, 60, 36))

    assert len(plates) == PLATE_COUNT
    brightness = [float(p.mean()) for p in plates]
    # Each step opens the mouth further, so the crop darkens monotonically.
    assert all(b <= a + 1e-6 for a, b in zip(brightness, brightness[1:]))
    assert brightness[0] - brightness[-1] > 4.0


def test_the_closed_plate_is_the_photograph_untouched():
    base = np.dstack([np.full((256, 256), 190, np.uint8)] * 3)
    box = (100, 140, 60, 36)
    x, y, w, h = box
    plates = build_plates(base, box)
    assert np.array_equal(plates[0], base[y : y + h, x : x + w])


def test_framing_changes_how_much_is_in_frame():
    """Head framing puts the face larger than torso framing does."""
    from avatar.ingest.assets import FACE_TARGET

    assert FACE_TARGET[Framing.HEAD] > FACE_TARGET[Framing.HALF_BODY]


def test_the_crop_puts_the_face_at_the_intended_size():
    """Tested through crop_for_framing directly.

    build_avatar_assets needs a detected face, and the bundled cascade does
    not detect drawn ones - see test_validate. Feeding it a candidate with a
    known box tests the geometry without depending on the detector.
    """
    from avatar.ingest.assets import Candidate, crop_for_framing

    image = np.full((1400, 1000, 3), 200, np.uint8)
    candidate = Candidate(image=image, box=(300, 200, 400, 400), sharpness=200.0,
                          frontality=0.9)

    for framing, target in FACE_TARGET.items():
        cropped = crop_for_framing(candidate, framing)
        assert cropped.shape == (OUTPUT_SIZE, OUTPUT_SIZE, 3)
        # The crop height is chosen so the face lands at `target` of the output.
        assert 0.1 < target < 0.6


def test_a_crop_at_the_edge_of_the_frame_is_padded_not_shifted():
    """A face near the border must not drag the subject off-centre."""
    from avatar.ingest.assets import Candidate, crop_for_framing

    image = np.full((600, 600, 3), 200, np.uint8)
    candidate = Candidate(image=image, box=(0, 0, 300, 300), sharpness=200.0,
                          frontality=0.9)
    cropped = crop_for_framing(candidate, Framing.HEAD)
    assert cropped.shape == (OUTPUT_SIZE, OUTPUT_SIZE, 3)


def test_assets_round_trip_to_disk(tmp_path):
    """Built by hand, since the detector cannot see a drawn face."""
    from avatar.renderer.plates import AvatarAssets

    base = np.dstack([np.full((OUTPUT_SIZE, OUTPUT_SIZE), 190, np.uint8)] * 3)
    box = (200, 300, 80, 48)
    built = AvatarAssets(
        idle_frames=build_idle_loop(base, 25, 1.0),
        mouth_box=box,
        plates=build_plates(base, box),
        fps=25,
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
    )
    built.save(tmp_path)
    loaded = AvatarAssets.load(tmp_path)
    assert loaded.mouth_box == built.mouth_box
    assert len(loaded.plates) == PLATE_COUNT


def test_the_renderer_accepts_what_this_produces():
    """The contract that matters: these assets drive the existing renderer."""
    from avatar.renderer.plates import AvatarAssets
    from avatar.renderer.viseme import VisemeRenderer

    base = np.dstack([np.full((OUTPUT_SIZE, OUTPUT_SIZE), 190, np.uint8)] * 3)
    box = (200, 300, 80, 48)
    assets = AvatarAssets(
        idle_frames=build_idle_loop(base, 25, 1.0),
        mouth_box=box,
        plates=build_plates(base, box),
        fps=25,
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
    )
    renderer = VisemeRenderer(assets)
    assert renderer.size == (OUTPUT_SIZE, OUTPUT_SIZE)
    assert renderer.fps == 25
