"""The motion system, wired into a live call.

Everything under avatar.motion is pure: numbers in, numbers out, no pipeline
and no clock of its own. That is what makes it testable on a laptop, and it is
also what leaves this file with a job. Three processors put the director where
the information is, and each of them exists because the information it needs
arrives at exactly one point in the graph:

    AffectTagStripper       what the model meant, which only exists between
                            the model and the voice
    MotionDirectorProcessor how the voice said it, and where the speech
                            timeline has got to
    ListenerCue             what the person on the other end is doing

Two rules run through all three.

**Nothing here may ever delay a frame.** Prosody analysis is signal processing
on hundreds of milliseconds of audio; it is fast, but it is not free, and the
one thing worse than motion arriving late is speech arriving late. So every
analysis runs on a task of its own, fed by an unbounded queue, and the audio
has already been pushed downstream by the time anything is queued. A wedged
analyser costs motion detail and never a millisecond of audio.

**Time is speech time, except once.** The director's timeline is seconds of
synthesised audio, not seconds elapsed, because a nod has to be scheduled
before its syllable is audible - see pose.py. ListenerCue is the deliberate
exception and says so where it lives.

One seam is worth explaining before it is read. The director consumes affect
and attitude through methods it has, and prosody and backchannel through
methods it may not: those two are declared here as protocols and offered with
an isinstance check, exactly like RendererStage.attach_motion offers a motion
source to a renderer that may only animate a mouth. A processor that required
a method the director does not have would be a wiring error discovered on a
live call, which is the worst possible place to discover one.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from typing import Protocol, runtime_checkable

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from avatar.motion.affect import LEXICON_SCAN_CHARS, Affect, affect_for
from avatar.motion.director import Attitude
from avatar.motion.prosody import ProsodyBuffer, ProsodyTrack
from avatar.renderer.base import AudioChunk


@runtime_checkable
class ProsodyAware(Protocol):
    """A director that can use a reading of how a sentence was delivered."""

    def apply_prosody(self, track: ProsodyTrack) -> None:
        """Take a span of analysed speech. Called off the audio path."""


@runtime_checkable
class BackchannelAware(Protocol):
    """A director that can be told to acknowledge what it just heard."""

    def backchannel(self, cue: str) -> None:
        """Do this now.

        No timestamp, deliberately: everywhere else in this system a time is a
        point on the speech timeline, and a backchannel is the one thing that
        is not on it. See ListenerCue for why.
        """


class _AudioTap(FrameProcessor):
    """Audio in on the frame path, analysis on a task of its own.

    The single rule this class exists to hold in one place: nothing on the
    frame path waits for analysis. Work is handed over with put_nowait onto an
    unbounded queue, after the frame has already been pushed, so the cost of a
    slow analyser is measured in motion detail rather than in audio.

    Unbounded is a choice and not an oversight. A bounded queue would have to
    either block the producer - the thing this is built to prevent - or drop
    audio, and dropped audio makes the analysis wrong rather than late. What
    bounds memory is ProsodyBuffer, which never holds more than MAX_BUFFER_S.
    """

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    def _start_worker(self) -> None:
        # The processor's own task manager rather than asyncio directly: a raw
        # task is not tracked by the pipeline, so it is not cancelled on
        # shutdown and can be collected before it ever runs.
        if self._worker is None or self._worker.done():
            self._worker = self.create_task(self._run())

    async def _stop_worker(self) -> None:
        if self._worker is not None:
            await self.cancel_task(self._worker)
            self._worker = None

    def _submit(self, item) -> None:
        self._queue.put_nowait(item)

    def _drop_pending(self) -> None:
        """Forget queued work. For barge-in, where it describes dead speech."""
        while not self._queue.empty():
            self._queue.get_nowait()

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                self._consume(item)
            except Exception as exc:  # noqa: BLE001 - motion is never worth a call
                logger.warning(f"motion analysis skipped a span: {exc}")

    def _consume(self, item) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------
# What the model meant


# How much of a reply may be held back while waiting for a tag to close. The
# tag itself is about a dozen characters; this is generous enough to survive a
# model that pads it and short enough that a reply which never had one is
# released within a token or two of the first word.
MAX_TAG_CHARS = 48

# A tag the model opened and never closed, at the head of a stream.
#
# affect.parse_tag already removes a mangled tag, but it recognises one by
# being a line of its own, which is how it looks once the reply is complete.
# Here the reply is still arriving and that newline may never come, so
# "[[warm|0.6 I remember that" would otherwise be read aloud as "warm zero
# point six". Two brackets are required, so somebody's own "[laughs]" is not
# touched.
_UNTERMINATED_TAG = re.compile(r"^\s*\[\[[ \t]*[A-Za-z_]*[ \t]*\|?[ \t]*[0-9.]*[ \t]*")


def _head_is_decided(head: str) -> bool:
    """Can this opening be judged yet, or is more of it still coming?

    Holding text back is the only thing in this file that can delay speech, so
    it is held for as little as possible: the moment the opening cannot be a
    tag, or the tag closes, or it is clear it never will.
    """
    lead = head.lstrip()
    if not lead:
        # Whitespace only. Nothing to read yet, and nothing worth speaking.
        return len(head) >= MAX_TAG_CHARS
    if not lead.startswith("["):
        return True
    if "]]" in lead:
        return True
    return "\n" in head or len(head) >= MAX_TAG_CHARS


def _without_unterminated_tag(head: str) -> str:
    if "]]" in head or not head.lstrip().startswith("[["):
        return head
    return _UNTERMINATED_TAG.sub("", head, count=1)


class AffectTagStripper(FrameProcessor):
    """Reads the model's declared affect, and removes every trace of it.

    Placed immediately after the llm in agent.py, which is what makes the two
    halves of the job one placement rather than two. Text-to-speech is the
    next processor downstream, and the assistant aggregator - the thing that
    writes conversation history - sits at the very end of the same downstream
    path, after the transport. So a tag removed here cannot be spoken and
    cannot be remembered, and neither consumer needs to know a tag exists.

    Reading the affect is best effort and removal is not: nothing shaped like
    a tag is allowed through, whether or not it parsed. A missing or mangled
    tag is silent - no warning, no retry, no waiting - because affect.py's
    lexicon covers it and a turn that waits on an expression is a turn the
    person on the other end hears stall.
    """

    def __init__(self, director=None):
        super().__init__()
        self._director = director
        self._buffering = False
        self._head = ""
        self._held: list[LLMTextFrame] = []
        self._reply = ""
        self._tagged = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._begin()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            await self._on_text(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            # A reply short enough to end while its opening is still held.
            if self._buffering:
                await self._release(direction)
            await self.push_frame(frame, direction)
            self._fall_back_to_the_lexicon()
            return

        if isinstance(frame, InterruptionFrame):
            # Held text belongs to a reply nobody will hear the end of, so it
            # is dropped rather than released into a turn that is over.
            self._begin()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    def _begin(self) -> None:
        self._buffering = True
        self._head = ""
        self._held = []
        self._reply = ""
        self._tagged = False

    async def _on_text(self, frame: LLMTextFrame, direction: FrameDirection) -> None:
        if not self._buffering:
            self._remember(frame.text)
            await self.push_frame(frame, direction)
            return

        self._head += frame.text or ""
        self._held.append(frame)
        if not _head_is_decided(self._head):
            return
        await self._release(direction)

    async def _release(self, direction: FrameDirection) -> None:
        """Push the opening, with anything tag-shaped taken out of it."""
        head, held = self._head, self._held
        self._buffering = False
        self._head, self._held = "", []

        affect, clean = affect_for(_without_unterminated_tag(head))
        self._tagged = affect.source == "tag"

        # affect_for strips, which is right for a whole reply and wrong for
        # the opening of one: the model streams "I loved that. " and then
        # "Thank you", and the space between them is the space between two
        # words. Losing it is how a synthesiser ends up saying "thatthank".
        if clean:
            clean += head[len(head.rstrip()) :]
        # Emitted whichever it is. A lexicon read of the opening is weak, but
        # it is available now, at the start of the reply, which is when the
        # face has to know something; the whole-reply read below refines it.
        self._show(affect)

        self._remember(clean)
        if not clean:
            # The opening was nothing but a tag. An empty text frame would
            # reach the aggregator as an empty assistant turn.
            return

        # The last held frame carries the released text, rather than a fresh
        # one: pipecat frames carry ids and metadata that downstream stages
        # match on, and inventing a frame throws that away.
        frame = held[-1]
        frame.text = clean
        await self.push_frame(frame, direction)

    def _remember(self, text: str) -> None:
        if self._tagged or not text:
            return
        # Capped where the lexicon stops reading anyway.
        room = LEXICON_SCAN_CHARS - len(self._reply)
        if room > 0:
            self._reply += text[:room]

    def _fall_back_to_the_lexicon(self) -> None:
        """A better guess than the opening gave, once the reply is complete.

        Late by design and not late in practice: the model finishes streaming
        well before the voice finishes speaking, so this lands during the
        reply rather than after it. A guess that waited for more text than
        this would be a face that changes expression after the sentence it
        belonged to.
        """
        if self._tagged or not self._reply:
            return
        affect, _ = affect_for(self._reply)
        self._show(affect)

    def _show(self, affect: Affect) -> None:
        if self._director is None:
            return
        try:
            self._director.set_affect(affect)
        except Exception as exc:  # noqa: BLE001 - a face is never worth a call
            logger.warning(f"affect not applied: {exc}")


# --------------------------------------------------------------------------
# How it was said, and where the speech timeline has got to


# How long a call has to be quiet before the likeness stops looking like it is
# about to be spoken to and starts looking like it is waiting. Long enough
# that an ordinary gap between turns never reaches it.
SILENCE_TO_WAITING_S = 8.0

# How often the silence is checked. Bounded above so a long threshold does not
# make the transition arrive up to a quarter late, and below so an ordinary
# call is not woken four times a second for nothing.
_SILENCE_POLL_S = 0.25


class MotionDirectorProcessor(_AudioTap):
    """Owns the speech clock, the attitude, and the prosody analysis.

    The clock is the reason this sits between tts and the renderer. Every
    chunk of synthesised audio is stamped with where it begins on the speech
    timeline before it is pushed, and the timeline is simply the audio that
    came before it - not elapsed time, which drifts from the audio the moment
    anything queues.
    """

    def __init__(
        self,
        director,
        *,
        silence_s: float = SILENCE_TO_WAITING_S,
        prosody: ProsodyBuffer | None = None,
    ):
        super().__init__()
        self._director = director
        self._prosody = prosody or ProsodyBuffer()
        self._silence_s = silence_s
        self._speech_s = 0.0
        self._speaking = False
        self._quiet_since = time.monotonic()
        self._watchdog: asyncio.Task | None = None
        # The last span described. Kept because a director that ignores
        # prosody is the normal case, and without this the analysis would be
        # unobservable from outside - including to a test.
        self.last_track: ProsodyTrack | None = None
        # Mirrored rather than read back from the director, which keeps its
        # attitude private, and so that a repeat is not sent as a transition.
        self.attitude = Attitude.LISTENING

    @property
    def speech_s(self) -> float:
        """Seconds of synthesised audio committed so far."""
        return self._speech_s

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._start_worker()
            self._start_watchdog()
            return

        if isinstance(frame, TTSAudioRawFrame):
            await self._on_audio(frame, direction)
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            self._became(Attitude.LISTENING)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._became(Attitude.THINKING)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (TTSStoppedFrame, BotStoppedSpeakingFrame)):
            # End of an utterance: analyse the tail rather than holding it
            # until the next turn pushes it past the threshold. The attitude
            # is left alone - the face keeps whatever it was doing until the
            # person speaks, or until the silence is long enough to be
            # waiting, and neither of those has happened yet.
            self._speaking = False
            self._quiet_since = time.monotonic()
            self._submit(None)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            await self._on_interruption(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop_watchdog()
            await self._stop_worker()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    async def _on_audio(self, frame: TTSAudioRawFrame, direction: FrameDirection) -> None:
        # Stamped before it is pushed, from the clock as it stands before this
        # chunk is counted: t0 is the audio that preceded it, which is what a
        # renderer needs to place a pose against a sample.
        t0 = self._speech_s
        # AudioChunk is defined on mono 16-bit PCM, which is what this
        # pipeline's TTS emits (audio_out is mono, 24kHz).
        chunk = AudioChunk(pcm=frame.audio, sample_rate=frame.sample_rate, t0=t0)
        if isinstance(frame.metadata, dict):
            # For anything downstream that builds its own AudioChunk and would
            # otherwise default t0 to zero. Metadata survives the copies
            # pipecat makes when a frame is broadcast.
            frame.metadata["t0"] = t0

        # Audio first, before the clock, the attitude or the queue. Nothing
        # about motion is worth a millisecond of speech.
        await self.push_frame(frame, direction)

        self._speech_s = t0 + chunk.duration_s
        self._quiet_since = time.monotonic()
        self._speaking = True
        # Every chunk, and a transition on the first of them: _became sends
        # nothing when the attitude is already speaking. Written this way
        # rather than with a "first chunk" flag so that audio which resumes
        # after the person interrupted announces itself again.
        self._became(Attitude.SPEAKING)
        self._submit(chunk)

    async def _on_interruption(self, frame: Frame, direction: FrameDirection) -> None:
        # Before the frame is pushed, and therefore before RendererProcessor
        # downstream cancels the stage. The order is the point: the renderer's
        # cancel is bounded at 100ms and frames resume immediately after it,
        # so the motion state has to already be correct when they do.
        # Interrupting the director is a state write, not a cancellation, so
        # doing it first costs nothing measurable.
        try:
            self._director.interrupt(self._speech_s)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"director did not take the interruption: {exc}")

        # Queued audio describes speech whose ending nobody heard. Analysing
        # it would place motion against a sentence that was abandoned.
        self._drop_pending()
        self._prosody.reset()
        self._speaking = False
        self._quiet_since = time.monotonic()
        # The director's own interrupt() returns to listening; mirror it, or
        # the next genuine transition to listening would be swallowed.
        self.attitude = Attitude.LISTENING

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    def _became(self, attitude: Attitude) -> None:
        self._quiet_since = time.monotonic()
        if attitude == self.attitude:
            return
        self.attitude = attitude
        try:
            self._director.set_attitude(attitude, self._speech_s)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"attitude not applied: {exc}")

    def _start_watchdog(self) -> None:
        if self._watchdog is None or self._watchdog.done():
            self._watchdog = self.create_task(self._watch_the_silence())

    async def _stop_watchdog(self) -> None:
        if self._watchdog is not None:
            await self.cancel_task(self._watchdog)
            self._watchdog = None

    async def _watch_the_silence(self) -> None:
        """Nothing said, by anybody, for long enough to show.

        A timer rather than a frame, because the event this reacts to is the
        absence of frames. Without it the likeness holds the last thing it was
        doing indefinitely, which reads as a video that has stalled.
        """
        interval = min(_SILENCE_POLL_S, max(0.01, self._silence_s / 4.0))
        while True:
            await asyncio.sleep(interval)
            if self._speaking:
                continue
            if time.monotonic() - self._quiet_since >= self._silence_s:
                # _became sends nothing when the attitude is already this one,
                # so a long silence is one transition rather than four a second.
                self._became(Attitude.WAITING)

    # ------------------------------------------------------------------
    def _consume(self, item) -> None:
        """One span of synthesised audio, analysed off the frame path."""
        if item is None:
            track = self._prosody.flush()
        else:
            track = self._prosody.push(item.pcm, item.sample_rate, item.t0)
        if track is None:
            return
        self.last_track = track
        if isinstance(self._director, ProsodyAware):
            self._director.apply_prosody(track)


# --------------------------------------------------------------------------
# What the person on the other end is doing


# When a backchannel lands, measured from the moment the person stopped.
#
# This is the one thing in the motion system that runs on the wall clock, and
# it is deliberate. Everywhere else, time is seconds of synthesised audio,
# because the likeness writes a sentence before it says it and a nod has to be
# scheduled before its syllable is audible. A backchannel is the opposite
# situation: it responds to speech that has already happened, and there is no
# future audio to schedule it against. Anchored to the speech timeline it
# would arrive whenever the likeness next spoke, which is not what a nod of
# acknowledgement is.
#
# The delay is the whole effect. A person nods a beat after the other one
# finishes, not on the instant; a nod with no delay does not read as attentive,
# it reads as a machine that was waiting for the input to end.
BACKCHANNEL_DELAY_S = (0.150, 0.250)

# At most one of these every few seconds. Any faster and the likeness is
# nodding along to everything, which is the specific failure that makes a
# listener look inattentive rather than attentive.
CUE_MIN_GAP_S = 2.8

# Added to the gap, never subtracted, so the floor above is a floor. Without
# it the cues fall into a rhythm, and a rhythm is the thing every other layer
# of this system is arranged to avoid.
CUE_GAP_JITTER_S = 0.6

# What a rising ending is. "fall-rise" is included because a listener hears it
# as a question even though a single line through the pitch calls it flat.
_QUESTION_CONTOURS = frozenset({"rise", "fall-rise"})


class ListenerCue(_AudioTap):
    """Backchannel: what the likeness does while somebody else is talking.

    Three cues, from the three things the person's own audio can be read for:
    a nod when a clause ends, a raised brow when the clause ended on a
    question, and a lean when they trail off mid-thought. Their audio is
    tapped rather than their transcript because all three are in the delivery,
    and the transcript arrives after the moment has passed anyway.

    Analysis runs only while the person is actually speaking. The gap between
    utterances is silence that would cost the same CPU to describe as speech
    does, and describes nothing.
    """

    def __init__(
        self,
        director,
        *,
        seed: int = 0,
        delay_s: tuple[float, float] = BACKCHANNEL_DELAY_S,
        min_gap_s: float = CUE_MIN_GAP_S,
        prosody: ProsodyBuffer | None = None,
    ):
        super().__init__()
        self._director = director
        self._prosody = prosody or ProsodyBuffer()
        self._delay_s = delay_s
        self._min_gap_s = min_gap_s
        # Seeded, like every other random thing in this system: two calls that
        # went the same way have to be able to be compared afterwards.
        self._rng = random.Random(seed)
        self._listening = False
        self._heard_s = 0.0
        self._contour = "flat"
        self._next_cue_at = 0.0
        # Cues in flight. A backchannel is a couple of hundred milliseconds of
        # waiting, so at shutdown there is usually one, and a task nobody
        # cancels outlives the pipeline that owns it.
        self._pending: set[asyncio.Task] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._start_worker()
            return

        if isinstance(frame, InputAudioRawFrame):
            await self._on_audio(frame, direction)
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            self._listening = True
            self._contour = "flat"
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._listening = False
            # Analyse the tail: the ending is where the question is.
            self._submit(None)
            await self.push_frame(frame, direction)
            self._cue("clause")
            return

        if isinstance(frame, InterruptionFrame):
            self._prosody.reset()
            self._drop_pending()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop_worker()
            for task in list(self._pending):
                await self.cancel_task(task)
            self._pending.clear()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    async def _on_audio(self, frame: InputAudioRawFrame, direction: FrameDirection) -> None:
        # AudioChunk is the renderer's type, reused here for what it is - PCM,
        # a rate and a position on a timeline. This one is the person's own
        # audio and never reaches a renderer.
        chunk = AudioChunk(
            pcm=frame.audio, sample_rate=frame.sample_rate, t0=self._heard_s
        )
        # The person's audio is not ours to hold up either.
        await self.push_frame(frame, direction)

        # The clock counts everything heard, including the silence that is not
        # analysed. That is what makes the gap between two utterances a real
        # discontinuity to ProsodyBuffer, which resets on one rather than
        # reporting the silence between them as a pause inside one.
        self._heard_s = chunk.t0 + chunk.duration_s
        if self._listening:
            self._submit(chunk)

    def _consume(self, item) -> None:
        if item is None:
            track = self._prosody.flush()
        else:
            track = self._prosody.push(item.pcm, item.sample_rate, item.t0)
        if track is None:
            return
        self._contour = track.contour
        if any(pause.kind == "breath" for pause in track.pauses):
            # They trailed off mid-thought. A lean is what a person does with
            # a silence they expect to end.
            self._cue("pause")

    # ------------------------------------------------------------------
    def _cue(self, reason: str) -> None:
        """Schedule a backchannel, or decline to.

        The rate limit is applied here rather than at delivery, so two cues
        arriving together cannot both pass the check and then land on top of
        each other a beat later.
        """
        now = time.monotonic()
        if now < self._next_cue_at:
            return
        delay = self._rng.uniform(*self._delay_s)
        self._next_cue_at = (
            now + delay + self._min_gap_s + self._rng.uniform(0.0, CUE_GAP_JITTER_S)
        )
        task = self.create_task(self._deliver(reason, delay))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _deliver(self, reason: str, delay: float) -> None:
        await asyncio.sleep(delay)
        # Chosen after the wait, not before it: the analysis of the last thing
        # they said finishes during the delay, so by now it is known whether
        # that clause ended as a question.
        if reason == "pause":
            cue = "lean_in"
        elif self._contour in _QUESTION_CONTOURS:
            # Not a gesture from gesture.py on purpose. A brow raise is a
            # sustained expression rather than a shaped movement with an apex,
            # and the gesture scheduler has nothing to anchor it to.
            cue = "brow_raise"
        else:
            cue = "nod_small"

        if isinstance(self._director, BackchannelAware):
            try:
                self._director.backchannel(cue)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"backchannel not applied: {exc}")
