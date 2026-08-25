"""The guardrail must be structural: on a match the transcription is dropped,
so nothing downstream - including the model - ever sees the utterance."""
from dataclasses import dataclass

from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection

from avatar.persona import persona_from_avatar
from avatar.safety.processor import CrisisProcessor


@dataclass
class FakeAvatar:
    id: str = "av-1"
    display_name: str = "Marguerite Chen"
    locale: str = "es"
    country: str = "ES"
    biography: str = "Una violonchelista de Vancouver."
    voice_description: str = ""
    boundaries: str = ""


PROFILE = persona_from_avatar(FakeAvatar(), frozenset({"ES"})).as_dict()


def _wire(collected, on_event=None):
    p = CrisisProcessor(PROFILE, on_event=on_event)

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        collected.append(frame)

    p.push_frame = capture  # type: ignore[method-assign]
    return p


async def test_ordinary_speech_passes_through():
    collected = []
    p = _wire(collected)
    frame = TranscriptionFrame(text="hola, como estas", user_id="u", timestamp="t")
    await p.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert any(isinstance(f, TranscriptionFrame) for f in collected)


async def test_crisis_utterance_never_reaches_downstream():
    collected = []
    p = _wire(collected)
    frame = TranscriptionFrame(text="quiero matarme", user_id="u", timestamp="t")
    await p.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert not any(isinstance(f, TranscriptionFrame) for f in collected)


async def test_crisis_utterance_emits_the_fixed_reply():
    collected = []
    p = _wire(collected)
    await p.process_frame(
        TranscriptionFrame(text="quiero matarme", user_id="u", timestamp="t"),
        FrameDirection.DOWNSTREAM,
    )
    spoken = [f for f in collected if isinstance(f, TTSSpeakFrame)]
    assert len(spoken) == 1
    assert PROFILE["crisis_line_number"] in spoken[0].text


async def test_crisis_is_recorded():
    collected, events = [], []

    async def on_event(match, text):
        events.append((match.term, text))

    p = _wire(collected, on_event=on_event)
    await p.process_frame(
        TranscriptionFrame(text="quiero matarme", user_id="u", timestamp="t"),
        FrameDirection.DOWNSTREAM,
    )
    assert events and events[0][0] == "matarme"
