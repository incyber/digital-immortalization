from avatar.vision.state import MAX_DESCRIPTION_CHARS, STALE_AFTER_S, SceneState


def test_empty_state_contributes_nothing_to_the_prompt():
    assert SceneState().as_prompt_fragment(now=0.0) == ""


def test_populated_state_contributes_one_line():
    s = SceneState()
    s.update("a man in a blue shirt, seated, waving", now=0.0)
    fragment = s.as_prompt_fragment(now=1.0)
    assert fragment.count("\n") == 0
    assert "blue shirt" in fragment


def test_failed_vision_call_leaves_previous_observation_standing():
    s = SceneState()
    s.update("a woman holding a book", now=0.0)
    s.update("", now=1.0)  # this is what describe_frame returns on failure
    assert "book" in s.description


def test_description_is_capped():
    s = SceneState()
    s.update("x" * 500, now=0.0)
    assert len(s.description) == MAX_DESCRIPTION_CHARS


def test_stale_observation_is_dropped_rather_than_asserted():
    s = SceneState()
    s.update("a man in a red coat", now=0.0)
    assert s.as_prompt_fragment(now=STALE_AFTER_S + 1) == ""
