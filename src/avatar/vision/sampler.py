"""Decides which camera frames are worth paying to describe.

Two independent conditions, both required:

  interval   at least vision_interval_s since the last upload. A hard ceiling
             on cost per session, regardless of how much is happening.
  motion     enough visual change since the last uploaded frame. Suppresses
             redundant spend below that ceiling - a person sitting still costs
             nothing after the first frame.

Rate limit outranks motion. A person waving continuously must not produce a
vision call per frame.
"""

from __future__ import annotations

import cv2
import numpy as np

# Frames are compared at this size in greyscale. Small enough that the diff is
# free, large enough to register a changed posture or a new object in frame.
_DIFF_SIZE = (64, 64)


class MotionGate:
    """Stateful. One per session."""

    def __init__(self, interval_s: float, threshold: float):
        self._interval_s = interval_s
        self._threshold = threshold
        self._last_sent_at: float | None = None
        self._last_signature: np.ndarray | None = None

    @staticmethod
    def _signature(frame_rgb: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(grey, _DIFF_SIZE, interpolation=cv2.INTER_AREA).astype(np.int16)

    def should_send(self, frame_rgb: np.ndarray, now: float) -> bool:
        """True only when both conditions hold. Records the decision."""
        signature = self._signature(frame_rgb)

        # First frame of a call always goes: the model should know what it is
        # looking at before the conversation starts, not four seconds in.
        if self._last_sent_at is None:
            self._accept(signature, now)
            return True

        if (now - self._last_sent_at) < self._interval_s:
            return False

        assert self._last_signature is not None
        if float(np.abs(signature - self._last_signature).mean()) < self._threshold:
            return False

        self._accept(signature, now)
        return True

    def force(self, frame_rgb: np.ndarray, now: float) -> None:
        """Bypass both conditions.

        Used at the first user utterance and on an explicit request such as
        "can you see this?", where the person is deliberately showing something
        and a dropped frame is a visible failure.
        """
        self._accept(self._signature(frame_rgb), now)

    def _accept(self, signature: np.ndarray, now: float) -> None:
        self._last_signature = signature
        self._last_sent_at = now
