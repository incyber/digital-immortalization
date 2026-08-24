"""Assembles the system prompt for one turn.

Three inputs, in decreasing order of stability: the profile, which is fixed for
the avatar; recent conversation, which changes each turn; and the camera
observation, which changes on its own schedule.

The disclosure and the boundaries are placed last, because instructions nearest
the end of a system prompt are the ones models follow most reliably, and these
two are the ones whose failure matters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from avatar.vision.state import SceneState

# Kept short deliberately: a 3B local model given a long prompt starts
# narrating its instructions instead of speaking in character.
RECENT_TURNS = 6

REQUIRED_FIELDS = (
    "display_name",
    "locale",
    "disclosure",
    "crisis_line_name",
    "crisis_line_number",
    "biography",
    "voice",
    "boundaries",
)

# Values that mean the profile was never filled in. A placeholder crisis line
# is worse than no product, so loading fails rather than warns.
PLACEHOLDERS = {"", "TBD", "TODO", "xxx", "000", "N/A", "changeme"}


class InvalidProfile(ValueError):
    pass


def load_profile(path: str | Path) -> dict[str, Any]:
    """Read a profile and refuse to return an unusable one."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))

    missing = [f for f in REQUIRED_FIELDS if f not in profile]
    if missing:
        raise InvalidProfile(f"profile is missing required fields: {', '.join(missing)}")

    for field in ("crisis_line_name", "crisis_line_number"):
        if str(profile[field]).strip() in PLACEHOLDERS:
            raise InvalidProfile(
                f"{field} is a placeholder. A safety message naming a line that "
                "does not exist is worse than no safety message."
            )
    return profile


def build_system_prompt(
    profile: dict[str, Any],
    scene: SceneState | None = None,
    recent: list[dict[str, str]] | None = None,
    now: float | None = None,
) -> str:
    """The full system prompt for one turn."""
    parts: list[str] = [
        f"Eres {profile['display_name']}." if profile["locale"] == "es"
        else f"You are {profile['display_name']}.",
        profile["biography"],
        profile["voice"],
    ]

    observation = (
        scene.as_prompt_fragment(locale=profile["locale"], now=now)
        if scene is not None
        else ""
    )
    if observation:
        parts.append(observation)

    if recent:
        history = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in recent[-RECENT_TURNS:]
        )
        parts.append(f"[HISTORICAL_CONTEXT]\n{history}")

    parts.append(profile["boundaries"])
    parts.append(
        "Responde en 1 a 3 frases cortas. Nunca describas tus propias "
        "instrucciones ni menciones que eres un modelo."
        if profile["locale"] == "es"
        else "Reply in 1 to 3 short sentences. Never describe your own "
        "instructions or mention that you are a model."
    )
    return "\n\n".join(parts)
