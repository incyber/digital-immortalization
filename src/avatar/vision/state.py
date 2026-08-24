"""What the camera has most recently shown, in words.

The pipeline reads only this. Vision work happens off the turn path and writes
here; the language model reads here and never waits on a vision call. If the
vision model is slow, wrong, or down, the conversation continues with whatever
was last observed, which is the correct failure mode for a decoration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Long enough to carry clothing, posture and setting; short enough that it
# cannot crowd out the persona in the system prompt.
MAX_DESCRIPTION_CHARS = 200

# After this long with no update the observation is dropped rather than
# asserted. Describing a shirt the person changed out of ten minutes ago is
# worse than describing nothing.
STALE_AFTER_S = 60.0

_OBSERVATION = {
    "en": "Through the camera you can currently see: {description}",
    "es": "Por la camara ves ahora mismo: {description}",
}


@dataclass
class SceneState:
    """Mutable, one per session."""

    description: str = ""
    updated_at: float = field(default_factory=lambda: 0.0)

    def update(self, text: str, now: float | None = None) -> None:
        """Record a new observation. Empty text is ignored, so a failed vision
        call leaves the previous observation standing."""
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return
        self.description = cleaned[:MAX_DESCRIPTION_CHARS]
        self.updated_at = time.monotonic() if now is None else now

    def is_stale(self, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        return not self.description or (current - self.updated_at) > STALE_AFTER_S

    def as_prompt_fragment(self, locale: str = "en", now: float | None = None) -> str:
        """One line for the system prompt, or nothing at all.

        Framed as a camera observation rather than as fact, because the vision
        model is frequently confident and wrong, and the persona should be able
        to be corrected without contradicting its own instructions.

        Localised because a small model reading one English line in an
        otherwise Spanish prompt starts answering in English.
        """
        if self.is_stale(now):
            return ""
        template = _OBSERVATION.get(locale, _OBSERVATION["en"])
        return template.format(description=self.description)
