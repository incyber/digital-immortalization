"""Crisis check as a pipeline stage.

Placed ahead of the context aggregator, so a match never reaches the model.
This is the structural version of the guardrail: it is not that the model is
instructed to break character, it is that the model is not consulted.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from avatar.safety.crisis import check, safety_reply


class CrisisProcessor(FrameProcessor):
    """Short-circuits a turn when a crisis term is heard."""

    def __init__(self, profile: dict, on_event=None):
        super().__init__()
        self._locale = profile.get("locale", "en")
        self._line_name = profile["crisis_line_name"]
        self._line_number = profile["crisis_line_number"]
        self._on_event = on_event

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            match = check(frame.text, self._locale)
            if match is not None:
                logger.warning(f"crisis guardrail fired on term {match.term!r}")
                if self._on_event is not None:
                    await self._on_event(match, frame.text)

                # Speak the fixed message and drop the transcription, so the
                # aggregator downstream never sees the utterance and the model
                # is never called for this turn.
                await self.push_frame(
                    TTSSpeakFrame(
                        safety_reply(match.locale, self._line_name, self._line_number)
                    ),
                    FrameDirection.DOWNSTREAM,
                )
                return

        await self.push_frame(frame, direction)
