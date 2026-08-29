"""The wiring, and the four things about it that are easy to get silently wrong.

The motion layers themselves are tested elsewhere, against numbers. What is
left once they are correct is placement and timing, and every failure in that
category is invisible until somebody is on a call:

  a tag the model mangled being read aloud, or written into the history as
  something the person said;

  a speech clock that counts elapsed time instead of audio, which puts every
  nod a little further behind the syllable it was meant to land on;

  analysis that runs on the audio path, which costs speech to buy motion;

  an interruption that reaches the renderer before the director, so the face
  is a frame late being right about a moment somebody is watching closely.

None of those show up in a screenshot, and all four are asserted here.
"""

import asyncio
import time

import numpy as np
import pytest
import pytest_asyncio
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from avatar.motion.director import Attitude
from avatar.motion.processor import (
    BACKCHANNEL_DELAY_S,
    CUE_GAP_JITTER_S,
    CUE_MIN_GAP_S,
    AffectTagStripper,
    ListenerCue,
    MotionDirectorProcessor,
)
from avatar.renderer.base import VideoFrame
from avatar.renderer.processor import RendererProcessor

SR = 24000


class FakeDirector:
    """The director's surface, recording rather than moving.

    Includes apply_prosody and backchannel, which the real MotionDirector does
    not have. Both are offered through an isinstance check on a protocol - see
    processor.py - so this double is also the proof that the offer is made.
    """

    def __init__(self, log=None):
        self.log = log if log is not None else []
        self.affects = []
        self.attitudes = []
        self.interrupts = []
        self.tracks = []
        self.cues = []

    def set_affect(self, affect):
        self.affects.append(affect)

    def set_attitude(self, attitude, t):
        self.attitudes.append((attitude, t))

    def interrupt(self, t):
        self.interrupts.append(t)
        self.log.append("director.interrupt")

    def apply_prosody(self, track):
        self.tracks.append(track)

    def backchannel(self, cue):
        self.cues.append((cue, time.monotonic()))


async def wire(processor, collected):
    """Give a processor the task manager a real pipeline would give it.

    Every processor here runs its analysis on a managed background task, so
    none of them can be exercised as a bare object; without this the first
    submitted chunk raises "TaskManager is not initialized".
    """
    from pipecat.clocks.system_clock import SystemClock
    from pipecat.processors.frame_processor import FrameProcessorSetup
    from pipecat.utils.asyncio.task_manager import TaskManager

    manager = TaskManager(loop=asyncio.get_running_loop())
    await processor.setup(
        FrameProcessorSetup(
            clock=SystemClock(), task_manager=manager, pipeline_worker=None
        )
    )

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        collected.append(frame)

    processor.push_frame = capture  # type: ignore[method-assign]
    return processor


def silence(seconds: float, sample_rate: int = SR) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


def voice(hz: float, seconds: float, sample_rate: int = SR) -> bytes:
    """Something with a pitch, so the analysis has something to find."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    signal = 0.4 * np.sin(2 * np.pi * hz * t)
    return np.clip(signal * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def tts(pcm: bytes, sample_rate: int = SR) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(audio=pcm, sample_rate=sample_rate, num_channels=1)


def heard(pcm: bytes, sample_rate: int = SR) -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=pcm, sample_rate=sample_rate, num_channels=1)


def spoken(collected) -> str:
    """Everything that would reach the voice, and the history, as one string."""
    return "".join(f.text for f in collected if isinstance(f, LLMTextFrame))


async def reply(stripper, collected, *pieces: str) -> None:
    """One model turn, streamed the way a model streams it."""
    await stripper.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    for piece in pieces:
        await stripper.process_frame(LLMTextFrame(text=piece), FrameDirection.DOWNSTREAM)
    await stripper.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)


@pytest.fixture
def collected():
    return []


@pytest.fixture
def director():
    return FakeDirector()


@pytest_asyncio.fixture
async def stripper(collected, director):
    return await wire(AffectTagStripper(director), collected)


# --------------------------------------------------------------------------
# The tag


async def test_a_well_formed_tag_is_read_and_never_spoken(stripper, collected, director):
    await reply(stripper, collected, "[[warm|0.6]]\n", "I remember that day.")

    assert spoken(collected) == "I remember that day."
    assert director.affects[0].label == "warm"
    assert director.affects[0].source == "tag"


async def test_a_tag_split_across_streamed_frames_is_still_removed(stripper, collected):
    """A model emits a tag as tokens, not as a line."""
    await reply(stripper, collected, "[[", "warm", "|0", ".6", "]]\n", "Hello, love.")

    assert spoken(collected) == "Hello, love."


@pytest.mark.parametrize(
    "pieces",
    [
        ("[[warm|0.6\n", "Hello, love."),          # never closed
        ("[[warm 0.6]]\n", "Hello, love."),        # no separator
        ("[[ warm | 0.6 ]]\n", "Hello, love."),    # padded
        ("[[wistful|0.6]]\n", "Hello, love."),     # a label nobody defined
        ("[[warm|0.6 Hello, love.",),              # no newline ever arrives
    ],
    ids=["unclosed", "no-pipe", "padded", "unknown-label", "no-newline"],
)
async def test_a_malformed_tag_never_reaches_the_voice(stripper, collected, pieces):
    """A synthesiser reads "[[warm|0.6" aloud as "warm zero point six".

    Which is worse than any wrong expression, and is the reason removal is
    unconditional while reading the affect is best effort.
    """
    await reply(stripper, collected, *pieces)

    text = spoken(collected)
    assert "[[" not in text and "]]" not in text
    assert "0.6" not in text
    assert "warm" not in text.lower()
    assert "Hello, love." in text or "love" in text


async def test_a_malformed_tag_never_enters_the_conversation_history(
    stripper, collected
):
    """History is written by the assistant aggregator, at the end of the graph.

    It reads the same frames the voice does, so this processor's placement
    immediately after the model is what keeps a tag out of both. The ordering
    that makes that true is asserted in test_the_pipeline_places_it_after_the
    _model below; what is asserted here is that nothing tag-shaped survives
    this processor in the first place.
    """
    await reply(stripper, collected, "[[serious|0.9\n", "That was a long time ago.")

    history = "".join(f.text for f in collected if isinstance(f, LLMTextFrame))
    assert history == "That was a long time ago."


async def test_a_missing_tag_is_silent_and_falls_back_to_the_words(
    stripper, collected, director
):
    """No warning, no retry, no waiting. The word list is always there."""
    await reply(stripper, collected, "I loved that. ", "Thank you, dear.")

    assert spoken(collected) == "I loved that. Thank you, dear."
    assert director.affects
    assert director.affects[-1].source == "lexicon"
    assert director.affects[-1].label == "warm"


async def test_a_reply_without_a_tag_is_not_held_back(stripper, collected):
    """No turn may ever wait on a tag.

    The first token of a reply that cannot be a tag is pushed on the same call
    that delivered it - not at the end of the response, and not after a
    timeout.
    """
    await stripper.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await stripper.process_frame(
        LLMTextFrame(text="I remember."), FrameDirection.DOWNSTREAM
    )

    assert spoken(collected) == "I remember."


async def test_the_crisis_reply_is_left_exactly_as_written(stripper, collected):
    """The fixed safety message comes through this stage on its way to tts."""
    from pipecat.frames.frames import TTSSpeakFrame

    frame = TTSSpeakFrame("Please call 024. They are there right now.")
    await stripper.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert collected[0].text == "Please call 024. They are there right now."


# --------------------------------------------------------------------------
# The speech clock


@pytest_asyncio.fixture
async def motion(collected, director):
    p = await wire(MotionDirectorProcessor(director), collected)
    await p.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    yield p
    await p._stop_watchdog()
    await p._stop_worker()


async def test_the_speech_clock_advances_by_exactly_the_audio_pushed(motion):
    """Elapsed time is not the same quantity and drifts from it immediately."""
    for _ in range(4):
        await motion.process_frame(tts(silence(0.25)), FrameDirection.DOWNSTREAM)

    assert motion.speech_s == pytest.approx(1.0)


async def test_each_chunk_is_stamped_with_the_audio_that_preceded_it(motion, collected):
    """t0 is the timeline, not a sequence number: a pose is placed against it."""
    for _ in range(3):
        await motion.process_frame(tts(silence(0.2)), FrameDirection.DOWNSTREAM)

    stamps = [f.metadata["t0"] for f in collected if isinstance(f, TTSAudioRawFrame)]
    assert stamps == pytest.approx([0.0, 0.2, 0.4])


async def test_the_clock_does_not_count_the_gaps_between_turns(motion):
    """Seconds of synthesised audio, so a pause between turns is not time."""
    await motion.process_frame(tts(silence(0.5)), FrameDirection.DOWNSTREAM)
    await motion.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)
    await motion.process_frame(tts(silence(0.5)), FrameDirection.DOWNSTREAM)

    assert motion.speech_s == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Off the audio path


async def test_audio_is_forwarded_before_analysis_completes(collected, director):
    """The property the whole design rests on, asserted as an ordering.

    Both halves write into one log. When the frame path returns, the audio is
    already out and the analyser has not been entered at all - not entered
    late, not entered quickly, not entered. That is the difference between
    analysis that is downstream of playback and analysis that is in front of
    it, and it is the difference between motion that lags and speech that
    stutters.
    """
    log: list[str] = []

    class SlowBuffer:
        def push(self, pcm, sample_rate, t, sentence_end=False):
            log.append("analysis")

        def flush(self):
            return None

        def reset(self):
            pass

    p = await wire(MotionDirectorProcessor(director, prosody=SlowBuffer()), collected)
    await p.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

    async def note(frame, direction=FrameDirection.DOWNSTREAM):
        log.append("audio")
        collected.append(frame)

    p.push_frame = note  # type: ignore[method-assign]

    await p.process_frame(tts(voice(180.0, 0.5)), FrameDirection.DOWNSTREAM)
    assert log == ["audio"], "analysis must not begin on the audio path at all"

    for _ in range(20):
        await asyncio.sleep(0.005)
        if "analysis" in log:
            break

    assert log == ["audio", "analysis"]
    await p._stop_watchdog()
    await p._stop_worker()


async def test_the_reading_of_the_delivery_reaches_the_director(motion, director):
    """A track per span of speech, produced away from the frame path."""
    for _ in range(4):
        await motion.process_frame(tts(voice(170.0, 0.25)), FrameDirection.DOWNSTREAM)
    for _ in range(20):
        await asyncio.sleep(0.01)
        if director.tracks:
            break

    assert director.tracks, "no prosody track ever reached the director"
    assert director.tracks[0].t0 == pytest.approx(0.0)
    assert motion.last_track is director.tracks[-1]


# --------------------------------------------------------------------------
# Barge-in


async def test_interruption_reaches_the_director_before_the_renderer_cancels(collected):
    """The order, not the fact of both happening.

    The renderer's cancel is bounded at 100ms and frames resume the moment it
    returns. If the director were interrupted after that, the first frames
    back would be drawn from motion belonging to a sentence that has been
    abandoned - the one moment in a call somebody is watching most closely.
    """
    log: list[str] = []

    class LoggingStage:
        fps = 25
        size = (8, 8)

        async def prepare(self, avatar_id):
            pass

        async def cancel(self):
            log.append("renderer.cancel")

        async def render(self, audio):
            yield VideoFrame(data=bytes(8 * 8 * 3), width=8, height=8)

        async def idle(self):
            while True:
                yield VideoFrame(data=bytes(8 * 8 * 3), width=8, height=8)
                await asyncio.sleep(0.01)

    director = FakeDirector(log=log)
    renderer = await wire(RendererProcessor(LoggingStage()), collected)
    motion = await wire(MotionDirectorProcessor(director), [])

    # Wired as the pipeline wires them: whatever motion pushes downstream is
    # what the renderer receives.
    async def forward(frame, direction=FrameDirection.DOWNSTREAM):
        await renderer.process_frame(frame, direction)

    motion.push_frame = forward  # type: ignore[method-assign]

    await motion.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert log == ["director.interrupt", "renderer.cancel"]
    await renderer._stop_idle()
    await motion._stop_watchdog()
    await motion._stop_worker()


async def test_the_director_is_interrupted_at_the_audio_actually_committed(motion, director):
    await motion.process_frame(tts(silence(0.4)), FrameDirection.DOWNSTREAM)
    await motion.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert director.interrupts == [pytest.approx(0.4)]


async def test_barge_in_drops_analysis_of_speech_nobody_heard_the_end_of(
    collected, director
):
    """Audio still waiting to be described belongs to an abandoned sentence.

    Describing it anyway would place stress and pauses against a delivery the
    person talked over, and the buffer would then stitch it to whatever is
    said next. The analyser is deliberately not started here, so the chunks
    are unambiguously still queued when the interruption arrives.
    """

    class CountingBuffer:
        def __init__(self):
            self.spans = 0
            self.resets = 0

        def push(self, pcm, sample_rate, t, sentence_end=False):
            self.spans += 1

        def flush(self):
            return None

        def reset(self):
            self.resets += 1

    buffer = CountingBuffer()
    p = await wire(MotionDirectorProcessor(director, prosody=buffer), collected)

    await p.process_frame(tts(voice(180.0, 0.6)), FrameDirection.DOWNSTREAM)
    await p.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    p._start_worker()
    await asyncio.sleep(0.02)

    assert buffer.spans == 0
    assert buffer.resets == 1
    await p._stop_worker()


# --------------------------------------------------------------------------
# Attitude


async def test_the_attitude_follows_the_conversation(motion, director):
    """One turn each way. Listening and thinking are not the same face.

    A call begins listening, so the first transition asserted here is out of
    that state and back into it; a repeat is not sent as a transition, which
    is what keeps the director from being told four times a second that
    nothing has changed.
    """
    await motion.process_frame(tts(silence(0.2)), FrameDirection.DOWNSTREAM)
    await motion.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await motion.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await motion.process_frame(tts(silence(0.2)), FrameDirection.DOWNSTREAM)

    assert [a for a, _ in director.attitudes] == [
        Attitude.SPEAKING,
        Attitude.LISTENING,
        Attitude.THINKING,
        Attitude.SPEAKING,
    ]


async def test_speaking_is_set_on_the_first_audio_and_not_on_every_chunk(
    motion, director
):
    for _ in range(5):
        await motion.process_frame(tts(silence(0.1)), FrameDirection.DOWNSTREAM)

    assert [a for a, _ in director.attitudes] == [Attitude.SPEAKING]


async def test_a_long_silence_becomes_waiting(collected, director):
    """The event this reacts to is the absence of frames, so it is a timer."""
    p = await wire(MotionDirectorProcessor(director, silence_s=0.05), collected)
    await p.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    await p.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    for _ in range(40):
        await asyncio.sleep(0.01)
        if p.attitude == Attitude.WAITING:
            break

    assert [a for a, _ in director.attitudes][-1] == Attitude.WAITING
    await p._stop_watchdog()
    await p._stop_worker()


# --------------------------------------------------------------------------
# Backchannel


@pytest_asyncio.fixture
async def listener(collected, director):
    p = await wire(ListenerCue(director, seed=11), collected)
    await p.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    yield p
    await p._stop_worker()


async def _speaks(listener, seconds: float = 0.6, hz: float = 165.0) -> None:
    await listener.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await listener.process_frame(heard(voice(hz, seconds)), FrameDirection.DOWNSTREAM)
    await listener.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)


async def test_a_nod_lands_a_beat_after_the_person_stops(listener, director):
    """On the wall clock, and deliberately not on the speech timeline.

    Everything else in this system is timed against synthesised audio, because
    a nod has to be started before its syllable is heard. This one responds to
    speech that has already happened, and the delay is the effect: a nod that
    is instantaneous does not read as attentive, it reads as a machine that
    was waiting for the input to stop.
    """
    stopped = time.monotonic()
    await _speaks(listener)

    for _ in range(60):
        await asyncio.sleep(0.01)
        if director.cues:
            break

    assert director.cues, "nothing acknowledged what the person just said"
    cue, at = director.cues[0]
    assert cue in {"nod_small", "brow_raise"}
    assert BACKCHANNEL_DELAY_S[0] <= at - stopped <= BACKCHANNEL_DELAY_S[1] + 0.08


async def test_backchannel_is_rate_limited(listener, director):
    """A listener who nods at everything reads as inattentive, not attentive."""
    for _ in range(4):
        await _speaks(listener, seconds=0.5)
        await asyncio.sleep(0.3)

    assert len(director.cues) == 1, "one cue per gap, not one per clause"


@pytest.mark.slow
async def test_the_gap_is_long_enough_to_be_the_stated_one(listener, director):
    """The jitter is added to the floor and never subtracted from it.

    Real seconds, because the quantity under test is a number of real seconds
    and a shortened one would be a test of a different rate limit than the one
    that ships. The wait is the full gap plus the whole jitter range, so the
    second cue is allowed however the jitter fell.
    """
    await _speaks(listener, seconds=0.5)
    await asyncio.sleep(0.3)
    assert director.cues

    await asyncio.sleep(CUE_MIN_GAP_S + CUE_GAP_JITTER_S)
    await _speaks(listener, seconds=0.5)
    for _ in range(80):
        await asyncio.sleep(0.01)
        if len(director.cues) > 1:
            break

    assert len(director.cues) == 2
    assert director.cues[1][1] - director.cues[0][1] >= CUE_MIN_GAP_S


async def test_a_rising_ending_raises_the_brow_instead_of_nodding(
    collected, director
):
    """A question is answered with a brow, not with agreement.

    The cue is chosen after the delay rather than before it, so the reading of
    how the clause ended is the one that finished during the wait.
    """
    p = await wire(ListenerCue(director, seed=3), collected)
    await p.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

    # A pitch that slides upward: the shape of a question, and nothing else.
    t = np.arange(int(1.0 * SR)) / SR
    instantaneous = 130.0 * (280.0 / 130.0) ** t
    signal = 0.5 * np.sin(2 * np.pi * np.cumsum(instantaneous) / SR)
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16).tobytes()

    await p.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await p.process_frame(heard(pcm), FrameDirection.DOWNSTREAM)
    await p.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    for _ in range(60):
        await asyncio.sleep(0.01)
        if director.cues:
            break

    assert director.cues and director.cues[0][0] == "brow_raise"
    await p._stop_worker()


async def test_the_person_s_own_audio_is_never_held_up(listener, collected):
    """Their audio is no more ours to delay than the likeness's own."""
    await listener.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await listener.process_frame(heard(voice(170.0, 0.5)), FrameDirection.DOWNSTREAM)

    assert any(isinstance(f, InputAudioRawFrame) for f in collected)


async def test_silence_between_utterances_is_not_analysed(listener, director):
    """It costs the same to describe as speech does and describes nothing."""
    await listener.process_frame(heard(silence(1.0)), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)

    assert director.cues == []


# --------------------------------------------------------------------------
# Placement
#
# Read out of the source rather than by instantiating the graph. The ordering
# is the invariant - each of these three processors is correct only where it
# is - and it must not drift during a refactor.


def _agent_source() -> str:
    from pathlib import Path

    return Path("src/avatar/realtime/agent.py").read_text()


def test_the_pipeline_places_the_stripper_after_the_model_and_before_the_voice():
    src = _agent_source()
    assert src.index("            llm,") < src.index("AffectTagStripper(director)")
    assert src.index("AffectTagStripper(director)") < src.index("            tts,")


def test_the_pipeline_places_the_stripper_before_the_history_is_written():
    """Same placement, second consumer: the aggregator that writes history."""
    src = _agent_source()
    assert src.index("AffectTagStripper(director)") < src.index("aggregators.assistant()")


def test_the_pipeline_places_the_speech_clock_between_the_voice_and_the_renderer():
    src = _agent_source()
    assert src.index("            tts,") < src.index("MotionDirectorProcessor(director)")
    assert src.index("MotionDirectorProcessor(director)") < src.index(
        "RendererProcessor(stage"
    )


def test_the_pipeline_taps_the_listener_at_the_transport():
    """After the transcriber the person's own delivery no longer exists."""
    src = _agent_source()
    assert src.index("transport.input()") < src.index("ListenerCue(director)")
    assert src.index("ListenerCue(director)") < src.index("            stt,")


def test_the_director_is_offered_to_the_renderer():
    """Offered, not required: a renderer that only animates a mouth has no
    attach_motion at all, and every existing backend stays substitutable."""
    src = _agent_source()
    assert 'getattr(stage, "attach_motion", None)' in src
    assert "await attach_motion(director)" in src
