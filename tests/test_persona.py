import json

import pytest

from avatar.persona import InvalidProfile, build_system_prompt, load_profile
from avatar.vision.state import SceneState

PROFILE_PATH = "src/avatar/profiles/colon.json"


def test_shipped_profile_loads():
    p = load_profile(PROFILE_PATH)
    assert p["display_name"] == "Cristóbal Colón"


def test_placeholder_crisis_line_is_refused(tmp_path):
    p = json.loads(open(PROFILE_PATH, encoding="utf-8").read())
    p["crisis_line_number"] = "TBD"
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(InvalidProfile, match="placeholder"):
        load_profile(f)


def test_missing_field_is_refused(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"display_name": "X"}), encoding="utf-8")
    with pytest.raises(InvalidProfile, match="missing required fields"):
        load_profile(f)


def test_prompt_has_no_unfilled_placeholders():
    prompt = build_system_prompt(load_profile(PROFILE_PATH))
    assert "{{" not in prompt and "{" not in prompt


def test_empty_scene_adds_no_observation():
    prompt = build_system_prompt(load_profile(PROFILE_PATH), SceneState(), now=0.0)
    assert "camera" not in prompt.lower()
    assert "camara" not in prompt.lower()


def test_populated_scene_adds_exactly_one_observation():
    # The shipped profile is Spanish, so the observation is too. Asserting on
    # the English wording here would pass only by accident.
    scene = SceneState()
    scene.update("un hombre con camisa azul, saludando", now=0.0)
    prompt = build_system_prompt(load_profile(PROFILE_PATH), scene, now=1.0)
    assert prompt.lower().count("por la camara") == 1
    assert "camisa azul" in prompt


def test_boundaries_come_after_history():
    profile = load_profile(PROFILE_PATH)
    recent = [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "buenas"}]
    prompt = build_system_prompt(profile, SceneState(), recent, now=0.0)
    assert prompt.index("HISTORICAL_CONTEXT") < prompt.index(profile["boundaries"])


def test_history_is_capped():
    profile = load_profile(PROFILE_PATH)
    recent = [{"role": "user", "content": f"turn-{i}"} for i in range(40)]
    prompt = build_system_prompt(profile, SceneState(), recent, now=0.0)
    assert "turn-0:" not in prompt
    assert "turn-39" in prompt
