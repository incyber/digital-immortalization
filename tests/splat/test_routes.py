"""What decides how a person's splat gets built, and what the customer is told.

The route is not a preference. It is a claim about how much of a dead person's
face was measured rather than invented, so these defend two things: that the
better material always wins, and that a family is told which one they got in
words they would use themselves.
"""

import pytest

from avatar.splat.routes import (
    MIN_PHOTOS_FOR_GENERATION,
    MIN_VIDEO_SECONDS,
    MIN_VIDEO_VIEWS,
    Intake,
    Route,
    choose_route,
)
from avatar.storage.keys import photo_key, source_clip_key

TENANT = "tenant-a"
SET = "set-1"


def photos(count: int) -> tuple[str, ...]:
    return tuple(photo_key(TENANT, SET, f"photo-{i:03d}.jpg") for i in range(count))


def an_intake(**overrides) -> Intake:
    base = {"tenant_id": TENANT, "photo_set_id": SET, "photo_keys": photos(20)}
    base.update(overrides)
    return Intake(**base)


def with_video(**overrides) -> Intake:
    clip = {
        "video_key": source_clip_key(TENANT, SET),
        "video_seconds": 30.0,
        "video_frames": 45,
        "video_frames_with_face": 44,
    }
    clip.update(overrides)
    return an_intake(**clip)


def test_a_usable_video_is_reconstructed():
    assert choose_route(with_video()).route is Route.RECONSTRUCT


def test_a_video_wins_even_against_a_large_album():
    """Measured geometry beats invented geometry, and no pile of photographs
    converts one into the other."""
    assert choose_route(with_video(photo_keys=photos(40))).route is Route.RECONSTRUCT


def test_photographs_alone_are_generated():
    assert choose_route(an_intake()).route is Route.GENERATE


def test_a_clip_too_short_to_show_a_second_angle_falls_back_to_photographs():
    short = with_video(video_seconds=MIN_VIDEO_SECONDS - 1)
    assert choose_route(short).route is Route.GENERATE


def test_a_clip_the_face_is_rarely_visible_in_falls_back_to_photographs():
    """Length is not coverage. Thirty seconds of a shoulder is one viewpoint."""
    averted = with_video(video_frames_with_face=MIN_VIDEO_VIEWS - 1)
    assert choose_route(averted).route is Route.GENERATE


def test_nothing_usable_is_refused_rather_than_built_badly():
    assert choose_route(an_intake(photo_keys=photos(1))).route is Route.REFUSE


def test_a_refused_intake_is_not_buildable():
    assert not choose_route(an_intake(photo_keys=())).buildable
    assert choose_route(an_intake()).buildable


def test_the_refusal_names_both_ways_out_of_it():
    """A customer told "no" must be told what would turn it into a yes."""
    decision = choose_route(an_intake(photo_keys=photos(1)))
    text = " ".join(decision.missing)
    assert "video" in text
    assert "photographs" in text
    assert f"{MIN_VIDEO_SECONDS:.0f} seconds" in text
    assert str(MIN_PHOTOS_FOR_GENERATION) in text


def test_the_refusal_counts_what_the_customer_actually_has():
    decision = choose_route(an_intake(photo_keys=photos(2)))
    assert "we have 2" in " ".join(decision.missing)


def test_the_refusal_says_why_the_video_was_no_good():
    decision = choose_route(
        an_intake(
            photo_keys=photos(1),
            video_key=source_clip_key(TENANT, SET),
            video_seconds=3.0,
            video_frames=6,
            video_frames_with_face=6,
        )
    )
    assert "3 seconds long" in " ".join(decision.missing)


def test_the_refusal_is_written_in_words_a_customer_would_use():
    decision = choose_route(an_intake(photo_keys=()))
    text = (decision.reasoning + " " + " ".join(decision.missing)).lower()
    for jargon in ("none", "null", "intake", "route", "gaussian", "reconstruct", "keys"):
        assert jargon not in text, f"{jargon!r} does not belong in front of a customer"


def test_the_reconstruct_reasoning_explains_the_choice_in_plain_words():
    reasoning = choose_route(with_video()).reasoning
    assert "video" in reasoning
    assert "camera" in reasoning
    assert "moved" in reasoning


def test_the_generate_reasoning_says_the_angles_were_filled_in():
    reasoning = choose_route(an_intake()).reasoning
    assert "filled in" in reasoning
    assert "not measured" in reasoning


def test_the_generate_reasoning_names_why_the_video_was_not_used():
    """Support gets asked this. So does the family, and the answer is theirs."""
    reasoning = choose_route(with_video(video_seconds=4.0)).reasoning
    assert "4 seconds long" in reasoning


def test_the_generate_reasoning_says_no_video_was_uploaded_when_none_was():
    assert "no video was uploaded" in choose_route(an_intake()).reasoning


def test_the_decision_records_the_evidence_support_will_ask_for():
    decision = choose_route(with_video(source_short_edge_px=1080))
    trail = " ".join(decision.considered)
    assert "video: 30s" in trail
    assert "frames examined: 45, with a face: 44" in trail
    assert "photographs accepted: 20" in trail
    assert "1080px" in trail


def test_route_values_are_stable():
    """They are written to a row and read back months later."""
    assert Route.RECONSTRUCT.value == "reconstruct"
    assert Route.GENERATE.value == "generate"
    assert Route.REFUSE.value == "refuse"


def test_a_video_must_arrive_with_the_frames_that_were_examined():
    """A clip nobody looked at cannot be called usable on its duration alone."""
    with pytest.raises(ValueError, match="frames examined"):
        Intake(tenant_id=TENANT, photo_set_id=SET, video_key="k", video_seconds=30.0)


def test_more_faces_than_frames_is_impossible_and_refused():
    with pytest.raises(ValueError, match="more frames contained a face"):
        Intake(
            tenant_id=TENANT,
            photo_set_id=SET,
            video_key="k",
            video_seconds=30.0,
            video_frames=4,
            video_frames_with_face=9,
        )


def test_the_views_come_from_the_clip_on_the_video_route():
    intake = with_video()
    assert intake.views == 45
    assert intake.views_with_face == 44


def test_the_views_come_from_the_photographs_when_the_video_is_unusable():
    intake = with_video(video_seconds=2.0, photo_keys=photos(20))
    assert intake.views == 20


def test_every_accepted_photograph_counts_as_a_view_with_a_face():
    """An image with no face in it was never accepted, so the two are equal."""
    intake = an_intake(photo_keys=photos(20))
    assert intake.views_with_face == intake.views == 20
