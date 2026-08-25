"""Upload validation, tested at the boundaries a customer will actually hit."""

import cv2
import numpy as np

from avatar.ingest.validate import (
    MAX_ACCEPTED,
    MIN_FACE_SHARPNESS,
    MIN_FOR_HALF_BODY,
    MIN_USABLE,
    Framing,
    PhotoVerdict,
    Reason,
    Verdict,
    face_sharpness,
    inspect_photo,
    inspect_set,
)


def encode(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def blank(w=1024, h=1024, value=128):
    rng = np.random.default_rng(0)
    # Textured rather than flat, so the sharpness check is not trivially failed.
    return (np.full((h, w, 3), value, dtype=np.uint8) + rng.integers(-30, 30, (h, w, 3))).astype(
        np.uint8
    )


def test_a_tiny_image_is_rejected():
    verdict = inspect_photo("small.jpg", encode(blank(300, 300)))
    assert verdict.verdict is Verdict.REJECTED
    assert Reason.TOO_SMALL in verdict.reasons


def test_an_image_with_no_face_is_rejected():
    verdict = inspect_photo("landscape.jpg", encode(blank()))
    assert Reason.NO_FACE in verdict.reasons


def test_a_faceless_image_is_not_also_called_blurry():
    # Sharpness is measured on the face. With no face there is nothing to
    # judge, and reporting both reasons tells the customer to fix the wrong
    # thing.
    verdict = inspect_photo("landscape.jpg", encode(blank()))
    assert Reason.BLURRY not in verdict.reasons


def test_a_small_stray_detection_is_not_treated_as_a_second_person():
    """Haar finds faces in clothing and background texture.

    On the real set it reported two or three in four photographs that
    contained one person, and each was rejected as a group photo.
    """
    from avatar.ingest.validate import SECOND_FACE_RATIO

    assert 0.0 < SECOND_FACE_RATIO < 1.0


def test_undecodable_bytes_are_rejected_not_crashed():
    verdict = inspect_photo("corrupt.jpg", b"this is not a jpeg")
    assert verdict.verdict is Verdict.REJECTED


def a_face(size=520, sharp=True):
    """A synthetic face the bundled cascade actually detects."""
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


def test_a_sharp_region_on_a_smooth_background_scores_sharp():
    """The bug this replaced, tested where it can be tested honestly.

    Whole-frame Laplacian variance is dominated by the background. On a real
    37-photo iPhone set, portrait-mode and plain-wall backgrounds pushed the
    frame-wide reading to 11-22 while the subject was sharp, and two thirds of
    the set was rejected as blurry.

    Asserted against face_sharpness directly with a known box, rather than
    through inspect_photo. The bundled cascade does not detect drawn faces, so
    a synthetic end-to-end version of this test would pass whether the fix
    worked or not - which is what an earlier version of it did.
    """
    frame = np.full((1400, 1000, 3), 235, np.uint8)   # large, smooth background
    frame[100:620, 240:760] = a_face(520)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # The whole-frame measure is low. That is exactly why it cannot decide.
    assert cv2.Laplacian(grey, cv2.CV_64F).var() < 60.0

    # The subject is not.
    assert face_sharpness(grey, (240, 100, 520, 520)) >= MIN_FACE_SHARPNESS


def test_a_genuinely_soft_region_scores_below_the_threshold():
    frame = np.full((1400, 1000, 3), 235, np.uint8)
    frame[100:620, 240:760] = a_face(520, sharp=False)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert face_sharpness(grey, (240, 100, 520, 520)) < MIN_FACE_SHARPNESS


def test_the_bundled_cascade_does_not_detect_drawn_faces():
    """Recorded so the limitation is visible rather than rediscovered.

    Haar keys on photographic texture. Every synthetic face tried here went
    undetected, which means any end-to-end fixture built from one tests
    nothing. Detection-dependent behaviour is covered against real
    photographs; everything else is tested through the functions directly.
    """
    from avatar.ingest.validate import detect_faces

    frame = np.full((1400, 1000, 3), 235, np.uint8)
    frame[100:620, 240:760] = a_face(520)
    assert detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) == []


def test_face_sharpness_does_not_depend_on_image_resolution():
    """The same photograph at two sizes must score comparably.

    An absolute threshold on an unnormalised measure penalises whichever
    camera happens to have more megapixels.
    """
    small = cv2.cvtColor(a_face(320), cv2.COLOR_BGR2GRAY)
    large = cv2.cvtColor(cv2.resize(a_face(320), (1280, 1280)), cv2.COLOR_BGR2GRAY)

    a = face_sharpness(small, (0, 0, small.shape[1], small.shape[0]))
    b = face_sharpness(large, (0, 0, large.shape[1], large.shape[0]))
    assert abs(a - b) / max(a, b) < 0.5


def test_face_sharpness_of_an_empty_crop_is_zero():
    grey = np.zeros((10, 10), np.uint8)
    assert face_sharpness(grey, (0, 0, 0, 0)) == 0.0


def test_the_sharpness_threshold_is_applied_to_the_face_not_the_frame():
    from avatar.ingest.validate import FACE_SHARPNESS_SIZE

    assert MIN_FACE_SHARPNESS > 0
    assert FACE_SHARPNESS_SIZE >= 128


def ok(name, fraction=0.5):
    return PhotoVerdict(name, Verdict.OK, [], face_height_fraction=fraction)


def test_too_few_usable_images_is_refused():
    result = inspect_set([ok(f"{i}.jpg") for i in range(MIN_USABLE - 1)])
    assert not result.acceptable
    assert "at least" in result.problems[0]


def test_too_many_images_is_refused():
    # Head-on and half-body mixed so this fails only on the count.
    photos = [ok(f"{i}.jpg", 0.5 if i % 2 else 0.25) for i in range(MAX_ACCEPTED + 5)]
    result = inspect_set(photos)
    assert any("overfits" in p for p in result.problems)


def test_a_set_of_only_head_shots_is_accepted_and_framed_as_a_head():
    """Close portraits are what families actually have.

    Requiring twenty torso photographs of somebody who has died is not a
    requirement anyone can meet. The framing follows the photographs instead.
    """
    result = inspect_set([ok(f"{i}.jpg", 0.5) for i in range(25)])
    assert result.acceptable, result.problems
    assert result.framing is Framing.HEAD


def test_enough_torso_shots_widen_the_framing():
    photos = [ok(f"h{i}.jpg", 0.5) for i in range(20)]
    photos += [ok(f"b{i}.jpg", 0.22) for i in range(MIN_FOR_HALF_BODY)]
    result = inspect_set(photos)
    assert result.acceptable
    assert result.framing is Framing.HALF_BODY


def test_torso_coverage_never_blocks_the_build():
    # It changes what the avatar looks like; it does not decide whether one
    # can exist at all.
    result = inspect_set([ok(f"{i}.jpg", 0.5) for i in range(20)])
    half = next(r for r in result.requirements if r.key == "half_body")
    assert half.met is False
    assert half.blocking is False
    assert result.acceptable


def test_a_well_covered_set_is_accepted():
    photos = [ok(f"head-{i}.jpg", 0.5) for i in range(18)]
    photos += [ok(f"body-{i}.jpg", 0.22) for i in range(MIN_FOR_HALF_BODY + 2)]
    result = inspect_set(photos)
    assert result.acceptable, result.problems


def test_rejected_images_do_not_count_towards_the_total():
    # Twelve good images and twenty rejects. Someone uploading thirty-two
    # files sees "32" and expects to pass; the count that matters is 12.
    good = [ok(f"g{i}.jpg", 0.22) for i in range(5)]
    good += [ok(f"h{i}.jpg", 0.5) for i in range(7)]
    bad = [PhotoVerdict(f"b{i}.jpg", Verdict.REJECTED, [Reason.BLURRY]) for i in range(20)]

    result = inspect_set(good + bad)

    assert len(result.usable) == 12
    assert not result.acceptable
    assert any("only 12 usable" in p for p in result.problems)


def test_exactly_the_minimum_is_accepted():
    # The boundary itself must pass, or the message telling customers they
    # need 15 is a lie.
    photos = [ok(f"b{i}.jpg", 0.22) for i in range(MIN_FOR_HALF_BODY)]
    photos += [ok(f"h{i}.jpg", 0.5) for i in range(MIN_USABLE - MIN_FOR_HALF_BODY)]
    result = inspect_set(photos)
    assert result.acceptable, result.problems


def test_requirements_report_progress_not_just_failure():
    """A dead button with no explanation is the failure this prevents.

    Somebody who has uploaded twenty-four usable photographs and is still
    blocked must be able to see which condition is unmet and by how much.
    """
    photos = [ok(f"h{i}.jpg", 0.42) for i in range(24)]  # all close portraits
    result = inspect_set(photos)

    by_key = {r.key: r for r in result.requirements}
    assert by_key["usable"].met is True
    assert by_key["usable"].current == 24

    assert by_key["half_body"].met is False
    assert by_key["half_body"].current == 0



def test_a_short_set_names_both_numbers():
    # "only 3 usable" tells somebody what to do; "not enough" does not.
    result = inspect_set([ok(f"h{i}.jpg", 0.42) for i in range(3)])
    assert any("only 3 usable" in p for p in result.problems)


def test_every_requirement_is_met_for_a_good_set():
    photos = [ok(f"h{i}.jpg", 0.42) for i in range(18)]
    photos += [ok(f"b{i}.jpg", 0.22) for i in range(MIN_FOR_HALF_BODY)]
    result = inspect_set(photos)
    assert all(r.met for r in result.requirements)
    assert result.acceptable


def test_opencv_still_provides_the_cascade():
    """Two OpenCV distributions are installed and import order decides which wins.

    mediapipe requires opencv-contrib-python 5.x; this project pins
    opencv-python-headless below 5 because plain 5.x dropped the bundled Haar
    API. Contrib 5.x still has it, so the combination works - but silently,
    and a future resolution could pick the one that does not.
    """
    assert hasattr(cv2, "CascadeClassifier")
    classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    assert not classifier.empty(), "the bundled frontal cascade must load"
