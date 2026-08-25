"""Upload validation, tested at the boundaries a customer will actually hit."""

import cv2
import numpy as np

from avatar.ingest.validate import (
    MAX_ACCEPTED,
    MIN_HALF_BODY,
    MIN_USABLE,
    PhotoVerdict,
    Reason,
    Verdict,
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


def test_undecodable_bytes_are_rejected_not_crashed():
    verdict = inspect_photo("corrupt.jpg", b"this is not a jpeg")
    assert verdict.verdict is Verdict.REJECTED


def test_a_blurred_image_is_rejected():
    blurred = cv2.GaussianBlur(blank(), (31, 31), 0)
    assert Reason.BLURRY in inspect_photo("blurry.jpg", encode(blurred)).reasons


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


def test_a_set_of_only_head_shots_is_refused():
    # Twenty-five perfectly good portraits that cannot produce a half-body
    # avatar. This is the failure mode most datasets have.
    result = inspect_set([ok(f"{i}.jpg", 0.5) for i in range(25)])
    assert not result.acceptable
    assert any("floating head" in p for p in result.problems)


def test_a_well_covered_set_is_accepted():
    photos = [ok(f"head-{i}.jpg", 0.5) for i in range(18)]
    photos += [ok(f"body-{i}.jpg", 0.22) for i in range(MIN_HALF_BODY + 2)]
    result = inspect_set(photos)
    assert result.acceptable, result.problems


def test_rejected_images_do_not_count_towards_the_total():
    # Twelve good images and twenty rejects. Someone uploading thirty-two
    # files sees "32" and expects to pass; the count that matters is 12.
    good = [ok(f"g{i}.jpg", 0.22) for i in range(MIN_HALF_BODY)]
    good += [ok(f"h{i}.jpg", 0.5) for i in range(7)]
    bad = [PhotoVerdict(f"b{i}.jpg", Verdict.REJECTED, [Reason.BLURRY]) for i in range(20)]

    result = inspect_set(good + bad)

    assert len(result.usable) == 12
    assert not result.acceptable
    assert any("only 12 usable" in p for p in result.problems)


def test_exactly_the_minimum_is_accepted():
    # The boundary itself must pass, or the message telling customers they
    # need 15 is a lie.
    photos = [ok(f"b{i}.jpg", 0.22) for i in range(MIN_HALF_BODY)]
    photos += [ok(f"h{i}.jpg", 0.5) for i in range(MIN_USABLE - MIN_HALF_BODY)]
    result = inspect_set(photos)
    assert result.acceptable, result.problems
