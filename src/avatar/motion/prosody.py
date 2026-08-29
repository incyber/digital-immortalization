"""How something was said, recovered from the audio of it being said.

The director knows what the likeness is saying, because it wrote the words. It
does not know how the voice said them, and that is where the motion lives: a
head nods on the stressed syllable, brows lift on a final rise, a breath is
taken in the gap before a new clause. Text alone cannot place any of those,
because the placement is in the delivery, not in the sentence.

The obvious answer is a forced aligner, and it is refused here for the same
reason viseme.py refuses one: this component is defined to carry no model
weights and no GPU, and it runs on the CPU beside a live call. So everything
below is signal processing over numpy - framewise RMS, an FFT autocorrelation
for pitch, peak picking - and nothing else.

Two consequences shape the design.

The first is that the analysis is downstream of playback. It never sits between
the synthesiser and the caller's ear; the audio has already been sent by the
time a track is produced. If this stage were slow it would still not delay
speech, but it would delay the motion that should accompany speech, so there is
a hard time budget: past it, analysis falls back to energy alone and says so
through `degraded`. A late frame is a worse failure than a track without pitch.

The second is that time here is speech time - seconds of synthesised audio,
supplied by the caller as t0 - and never the wall clock. The whole point of
pose.py's timeline is that a nod can be scheduled before its syllable is heard,
which only works if the timestamps refer to the audio rather than to the moment
it happened to be analysed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Analysis frame: 40ms of audio every 10ms. Long enough to hold two cycles of
# the lowest pitch searched for, short enough that a syllable is several
# frames. The hop sets the timing resolution of everything below.
FRAME_S = 0.040
HOP_S = 0.010

# Pitch search range. Below 60Hz is nothing a human voice does in a call, and
# above 350Hz the FFT autocorrelation starts finding harmonics of formants
# rather than the fundamental.
F0_MIN = 60.0
F0_MAX = 350.0

# Normalised autocorrelation at the chosen lag, above which a frame counts as
# voiced. Speech sits well above this and noise well below; the gap is wide,
# which is why a single threshold is enough.
NACF_VOICED = 0.45

# A lag is preferred over the global peak if it reaches this fraction of it.
# Without it, a pure tone correlates just as well at twice its period and the
# reported pitch drops an octave - the classic failure of every autocorrelation
# pitch tracker, and the one that would make brows lift on the wrong words.
OCTAVE_BIAS = 0.90

# Stress is where loudness and pitch rise together, so the score is a weighted
# sum of the two z-scores. Loudness carries more because it survives the
# unvoiced frames that pitch has nothing to say about.
STRESS_ENERGY_WEIGHT = 0.6
STRESS_PITCH_WEIGHT = 0.4

# How far above the score's own spread a peak must stand, and how close two
# peaks may be. 180ms is about the shortest gap between two syllables a
# listener hears as separately stressed; closer than that and one nod would be
# starting while the last is still finishing.
STRESS_SIGMA = 0.5
STRESS_MIN_GAP_S = 0.180

# A stress peak has to be in audible speech. Without this, the quietest frame
# of a silent passage wins its own z-score competition.
STRESS_MIN_ENERGY = 0.15

# Energy, as a fraction of the adaptive ceiling, below which a frame is silence.
# Same value and same reasoning as viseme.py: enough to reject room tone,
# little enough to keep a trailing consonant.
SILENCE_FLOOR = 0.06

# What counts as a pause, and what kind of pause it is. A beat is a syllable's
# worth of nothing - the gap around an emphasised word. A clause boundary is
# where the head resets and the eyes may move. Longer than that and the body
# takes a breath, which is the one place a visible breath does not look staged.
PAUSE_MIN_S = 0.080
PAUSE_CLAUSE_S = 0.250
PAUSE_BREATH_S = 0.600

# The last stretch of voiced audio decides the contour. Final pitch movement in
# English is carried in roughly the last syllable and a half.
CONTOUR_WINDOW_S = 0.400

# Slope, in semitones per second, that separates a rise or a fall from a level
# delivery. Semitones rather than Hz so a low voice and a high voice are held
# to the same standard.
CONTOUR_SLOPE_ST = 2.0

# A syllable nucleus is a peak in voiced loudness. Two of them cannot be closer
# than this - faster than about eight syllables a second is not speech.
NUCLEUS_MIN_GAP_S = 0.120

# The ceiling that energy is normalised against is a high percentile of the
# frames in hand, not their maximum: one plosive should not flatten a sentence.
# viseme.py does the same job with a per-frame running maximum because it is
# stateful and sees one frame at a time; analysis sees a whole span at once, so
# it can afford the more robust statistic. ProsodyBuffer carries the value
# between spans with the same decay, so a quiet passage does not suddenly
# rescale itself.
CEILING_PERCENTILE = 95.0
CEILING_DECAY = 0.98
CEILING_MIN = 200.0

# Default time budget for one call to analyse(). Deliberately small: this runs
# on the same CPU as the call, and the pitch stage is the only part expensive
# enough to be worth skipping.
BUDGET_S = 0.008

# Cost of the pitch stage per analysis frame, used to decide before starting
# whether it fits in what is left of the budget. Measured at about 1.5e-5 s on
# an unloaded laptop core; carried at twice that, so a machine busy with the
# call it is running beside degrades early rather than overrunning. Overrunning
# is the one failure this constant exists to prevent, so it is meant to be
# pessimistic - being wrong here costs pitch, and being wrong the other way
# costs a video frame.
PITCH_COST_PER_FRAME_S = 3.0e-5

_TINY = 1e-12


@dataclass(frozen=True)
class StressPeak:
    """A syllable delivered harder than its neighbours.

    strength is 0..1 rather than the raw score, because what consumes this is
    a nod amplitude and not a statistic.
    """

    t: float
    strength: float


@dataclass(frozen=True)
class Pause:
    """A gap in speech, and what the body should do with it.

    kind is "beat", "clause" or "breath" - see PAUSE_MIN_S and below for where
    the boundaries are and why they are there.
    """

    t0: float
    t1: float
    kind: str

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass(frozen=True)
class ProsodyTrack:
    """Everything one span of speech offers the director.

    The frame arrays are kept alongside the events because a nod needs the
    event and a lean needs the envelope. They all share the same time base:
    frame i is centred at t0 + i * hop_s + FRAME_S / 2.
    """

    t0: float = 0.0
    duration: float = 0.0
    hop_s: float = HOP_S
    energy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    f0: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    voiced: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    stress: list[StressPeak] = field(default_factory=list)
    pauses: list[Pause] = field(default_factory=list)
    contour: str = "flat"
    speech_rate: float = 0.0
    degraded: bool = False

    @property
    def times(self) -> np.ndarray:
        """Centre time of every frame, on the speech timeline."""
        return self.t0 + np.arange(len(self.energy)) * self.hop_s + FRAME_S / 2.0


def _as_samples(pcm: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Signed 16-bit little-endian mono PCM as float32.

    Chunks arrive from a synthesiser mid-sentence and occasionally mid-sample;
    a trailing odd byte is dropped rather than raising, because the alternative
    is losing a whole utterance to one truncated write.
    """
    if isinstance(pcm, np.ndarray):
        return np.ascontiguousarray(pcm, dtype=np.float32)
    raw = bytes(pcm)
    if len(raw) % 2:
        raw = raw[:-1]
    return np.frombuffer(raw, dtype="<i2").astype(np.float32)


def _frame(samples: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Overlapping analysis frames, as a view where possible.

    Anything shorter than one window is zero-padded into a single frame instead
    of being dropped. A 10ms chunk is a legitimate thing for a synthesiser to
    emit and should produce a small answer, not an empty one.
    """
    if samples.size == 0:
        return np.zeros((0, win), dtype=np.float32)
    if samples.size < win:
        padded = np.zeros(win, dtype=np.float32)
        padded[: samples.size] = samples
        return padded[None, :]
    count = 1 + (samples.size - win) // hop
    return np.lib.stride_tricks.sliding_window_view(samples, win)[::hop][:count]


def _adaptive_ceiling(rms: np.ndarray, previous: float = CEILING_MIN) -> float:
    """Loudest thing recently said, not the loudest thing representable.

    Normalising against full scale would leave a quiet voice permanently at the
    bottom of the range and a loud one permanently clipped, so the reference is
    what this speaker is actually doing. It decays towards the floor so the
    reference follows the voice down as well as up.
    """
    decayed = max(previous * CEILING_DECAY, CEILING_MIN)
    if rms.size == 0:
        return decayed
    loud = float(np.percentile(rms, CEILING_PERCENTILE))
    return max(loud, decayed)


def _zscore(values: np.ndarray, where: np.ndarray | None = None) -> np.ndarray:
    """Standardised values, and zeros when there is nothing to standardise.

    A constant signal has no spread, and dividing by it would turn numerical
    jitter into confident structure - stress peaks invented out of rounding
    error in the pitch interpolator. The floor on the deviation is what stops
    that, and it is deliberately not merely a guard against exact zero.
    """
    mask = np.ones(len(values), dtype=bool) if where is None else where
    if mask.sum() < 4:
        return np.zeros(len(values), dtype=np.float32)
    selected = values[mask]
    spread = float(selected.std())
    if spread < 1e-4:
        return np.zeros(len(values), dtype=np.float32)
    out = np.zeros(len(values), dtype=np.float32)
    out[mask] = (selected - selected.mean()) / spread
    return out


def _pitch(frames: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Fundamental frequency and its confidence, one value per frame.

    Autocorrelation by FFT over every frame at once. Doing it frame by frame in
    Python is what makes pitch tracking feel expensive; done as one batched
    transform it is a few milliseconds a second of audio, which is what lets it
    stay inside the budget at all.

    The correlation is normalised by the energy of the two overlapping halves
    rather than by the frame energy, so the peak height means the same thing at
    every lag and can be compared against one voicing threshold.
    """
    count, win = frames.shape
    lag_min = max(2, int(sample_rate // F0_MAX))
    lag_max = min(win - 2, int(np.ceil(sample_rate / F0_MIN)))
    if count == 0 or lag_max <= lag_min + 2:
        return np.zeros(count, dtype=np.float32), np.zeros(count, dtype=np.float32)

    # Mean removal, so a DC offset - which a synthesiser can and does emit -
    # does not correlate perfectly with itself at every lag and read as voiced.
    centred = frames - frames.mean(axis=1, keepdims=True)

    size = 1 << int(np.ceil(np.log2(2 * win)))
    spectrum = np.fft.rfft(centred, n=size, axis=1)
    acf = np.fft.irfft(spectrum.real**2 + spectrum.imag**2, n=size, axis=1)

    lags = np.arange(lag_min, lag_max + 1)
    power = np.cumsum(centred * centred, axis=1)
    head = power[:, win - lags - 1]
    tail = power[:, -1:] - power[:, lags - 1]
    denominator = np.sqrt(np.maximum(head * tail, 0.0))
    nacf = np.where(denominator > _TINY, acf[:, lags] / np.maximum(denominator, _TINY), 0.0)

    rows = np.arange(count)
    best = nacf.max(axis=1)

    # The lowest-lag peak that gets close to the best one, rather than the best
    # one itself. It has to be a peak: the flank leading up to the true peak
    # also crosses the threshold, and taking the first crossing would report a
    # pitch a few percent sharp on every single frame.
    summit = np.zeros_like(nacf, dtype=bool)
    summit[:, 1:-1] = (nacf[:, 1:-1] >= nacf[:, :-2]) & (nacf[:, 1:-1] > nacf[:, 2:])
    eligible = summit & (nacf >= (best * OCTAVE_BIAS)[:, None])
    chosen = np.where(eligible.any(axis=1), eligible.argmax(axis=1), nacf.argmax(axis=1))
    peak = nacf[rows, chosen]

    # Parabolic interpolation through the three points around the peak. The lag
    # grid is coarse at high pitch - one sample at 24kHz is 8Hz at 300Hz - and
    # without this the reported pitch moves in visible steps.
    left = nacf[rows, np.clip(chosen - 1, 0, len(lags) - 1)]
    right = nacf[rows, np.clip(chosen + 1, 0, len(lags) - 1)]
    curvature = left - 2.0 * peak + right
    shift = np.where(curvature < -_TINY, 0.5 * (left - right) / np.where(
        curvature < -_TINY, curvature, -1.0
    ), 0.0)
    shift = np.clip(shift, -0.5, 0.5)

    refined = lags[chosen] + shift
    f0 = np.where(refined > _TINY, sample_rate / np.maximum(refined, _TINY), 0.0)
    f0 = np.where((f0 >= F0_MIN) & (f0 <= F0_MAX), f0, 0.0)
    return f0.astype(np.float32), peak.astype(np.float32)


def _pauses(energy: np.ndarray, times: np.ndarray, hop_s: float) -> list[Pause]:
    """Runs of silence, sized and named.

    A run is bounded by the centres of its first and last silent frames, which
    each sit half a window inside the true silence - a frame only falls below
    the floor once its whole window is quiet - so the window length is added
    back to recover the gap the listener actually heard.

    Runs touching either end of the span are dropped. Their real length is
    unknown: the audio continues in the next chunk, and calling an unfinished
    silence a breath would have the body inhale in the middle of a word.
    """
    if energy.size == 0:
        return []
    silent = energy < SILENCE_FLOOR
    if not silent.any():
        return []

    edges = np.diff(silent.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1))
    if silent[0]:
        starts.insert(0, 0)
    if silent[-1]:
        ends.append(len(silent) - 1)

    out: list[Pause] = []
    for start, end in zip(starts, ends):
        if start == 0 or end == len(silent) - 1:
            continue
        span = (end - start) * hop_s + FRAME_S
        if span < PAUSE_MIN_S:
            continue
        if span < PAUSE_CLAUSE_S:
            kind = "beat"
        elif span < PAUSE_BREATH_S:
            kind = "clause"
        else:
            kind = "breath"
        centre_start = float(times[start]) - FRAME_S / 2.0
        out.append(Pause(t0=centre_start, t1=centre_start + span, kind=kind))
    return out


def _stress(
    energy: np.ndarray, f0: np.ndarray, voiced: np.ndarray, times: np.ndarray
) -> list[StressPeak]:
    """Where the delivery leans, as a small number of timed events.

    Peaks are taken strongest first and thin out their neighbours, rather than
    every local maximum being kept and filtered afterwards. The difference
    shows when two syllables inside one refractory window are both prominent:
    picked in time order the weaker one wins by arriving first.
    """
    if energy.size == 0:
        return []

    score = STRESS_ENERGY_WEIGHT * _zscore(energy)
    if voiced.any():
        score = score + STRESS_PITCH_WEIGHT * _zscore(np.log(np.maximum(f0, 1.0)), voiced)

    spread = float(score.std())
    if spread < 1e-6:
        return []
    threshold = STRESS_SIGMA * spread

    interior = np.zeros(len(score), dtype=bool)
    if len(score) >= 3:
        interior[1:-1] = (score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:])
    elif len(score) == 1:
        interior[0] = True
    candidates = np.flatnonzero(interior & (score > threshold) & (energy > STRESS_MIN_ENERGY))
    if candidates.size == 0:
        return []

    taken: list[int] = []
    for index in candidates[np.argsort(-score[candidates])]:
        if all(abs(times[index] - times[other]) >= STRESS_MIN_GAP_S for other in taken):
            taken.append(int(index))

    taken.sort()
    return [
        StressPeak(
            t=float(times[i]),
            strength=float(np.clip(score[i] / (3.0 * spread), 0.0, 1.0)),
        )
        for i in taken
    ]


def _contour(f0: np.ndarray, voiced: np.ndarray, times: np.ndarray) -> str:
    """What the pitch did on the way out.

    A straight line through log f0 over the final stretch. Log, so the answer
    is about musical interval and not about how high the voice sits; a line,
    because anything more expressive would be fitting the tracker's noise.

    "fall-rise" is checked separately because it is the one common shape a
    single line describes as flat while a listener hears a question.
    """
    if f0.size == 0 or not voiced.any():
        return "flat"

    window = times >= (times[-1] - CONTOUR_WINDOW_S)
    mask = voiced & window
    if mask.sum() < 4:
        return "flat"

    semitones = 12.0 * np.log2(np.maximum(f0[mask], 1.0))
    when = times[mask]

    def slope(t: np.ndarray, y: np.ndarray) -> float:
        if len(t) < 3 or float(t.std()) < 1e-6:
            return 0.0
        return float(np.polyfit(t - t.mean(), y, 1)[0])

    half = len(when) // 2
    if half >= 3:
        first = slope(when[:half], semitones[:half])
        second = slope(when[half:], semitones[half:])
        if first < -CONTOUR_SLOPE_ST and second > CONTOUR_SLOPE_ST:
            return "fall-rise"

    overall = slope(when, semitones)
    if overall > CONTOUR_SLOPE_ST:
        return "rise"
    if overall < -CONTOUR_SLOPE_ST:
        return "fall"
    return "flat"


def _speech_rate(energy: np.ndarray, voiced: np.ndarray, hop_s: float) -> float:
    """Syllable nuclei per second of voiced audio.

    Per voiced second rather than per elapsed second, so a sentence with a long
    pause in it does not read as slow speech. The director uses this to scale
    how much motion a passage gets; excited speech is fast and moves more.
    """
    speaking = float(voiced.sum()) * hop_s
    if speaking < 0.1 or energy.size < 3:
        return 0.0

    # Smoothed, because the nucleus is the syllable's loud middle and not every
    # ripple of the envelope inside it.
    smooth = np.convolve(energy, np.ones(3) / 3.0, mode="same")
    peaks = (smooth[1:-1] >= smooth[:-2]) & (smooth[1:-1] > smooth[2:])
    indices = np.flatnonzero(peaks) + 1
    indices = indices[voiced[indices] & (smooth[indices] > 2.0 * SILENCE_FLOOR)]

    count = 0
    last = -np.inf
    for index in indices:
        if index * hop_s - last >= NUCLEUS_MIN_GAP_S:
            count += 1
            last = index * hop_s
    return count / speaking


def analyse(
    pcm: bytes | bytearray | memoryview | np.ndarray,
    sample_rate: int,
    t0: float = 0.0,
    budget_s: float = BUDGET_S,
) -> ProsodyTrack:
    """One span of speech, described.

    The budget is a promise, not a target. Energy, silences and their timing
    are always produced because they cost one pass over the samples; pitch is
    produced only if what remains of the budget covers it, and the track says
    `degraded` when it did not. The director already treats pitch as optional -
    an unvoiced frame has none either - so the degraded path loses the brow
    lift and keeps the nods, which is the right thing to lose.
    """
    return _analyse(pcm, sample_rate, t0, budget_s, CEILING_MIN)[0]


def _analyse(
    pcm: bytes | bytearray | memoryview | np.ndarray,
    sample_rate: int,
    t0: float,
    budget_s: float,
    ceiling: float,
) -> tuple[ProsodyTrack, float]:
    """analyse(), plus the loudness reference it settled on.

    ProsodyBuffer needs that reference to carry into the next span, and it is
    kept off ProsodyTrack because it is a detail of the measurement rather than
    something about the speech.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    started = time.perf_counter()

    samples = _as_samples(pcm)
    win = max(2, round(FRAME_S * sample_rate))
    hop = max(1, round(HOP_S * sample_rate))
    hop_s = hop / sample_rate

    frames = _frame(samples, win, hop)
    count = frames.shape[0]
    duration = samples.size / sample_rate

    rms = np.sqrt(np.maximum((frames * frames).mean(axis=1), 0.0))
    level = _adaptive_ceiling(rms, ceiling)
    energy = np.clip(rms / max(level, _TINY), 0.0, 1.0).astype(np.float32)
    times = t0 + np.arange(count) * hop_s + FRAME_S / 2.0

    spent = time.perf_counter() - started
    affordable = spent + count * PITCH_COST_PER_FRAME_S <= budget_s

    if affordable and count:
        f0, confidence = _pitch(frames, sample_rate)
        voiced = (confidence > NACF_VOICED) & (f0 > 0.0) & (energy > SILENCE_FLOOR)
        f0 = np.where(voiced, f0, 0.0).astype(np.float32)
    else:
        f0 = np.zeros(count, dtype=np.float32)
        voiced = np.zeros(count, dtype=bool)

    degraded = bool(count) and not affordable

    # With no pitch there is no voicing, so rate falls back to the frames that
    # are loud enough to be speech at all.
    speaking = voiced if not degraded else energy > SILENCE_FLOOR

    track = ProsodyTrack(
        t0=t0,
        duration=duration,
        hop_s=hop_s,
        energy=energy,
        f0=f0,
        voiced=voiced,
        stress=_stress(energy, f0, voiced, times),
        pauses=_pauses(energy, times, hop_s),
        contour=_contour(f0, voiced, times),
        speech_rate=_speech_rate(energy, speaking, hop_s),
        degraded=degraded,
    )
    return track, level


# How much unanalysed audio makes a span worth analysing. Shorter than this and
# there is not enough of a contour to fit a line through; much longer and the
# motion it describes has already been rendered.
MIN_ANALYSIS_S = 0.400

# How much already-analysed audio is carried into the next span. A syllable
# straddling a chunk boundary needs its beginning to be measurable, and the
# pitch tracker needs a full window before its first new frame.
TAIL_S = 0.200

# A gap larger than this between the end of the buffer and the timestamp of the
# next chunk means the utterance was cut - barge-in, or a new turn - and the
# tail belongs to something the speaker is no longer saying.
CONTINUITY_S = 0.050

# Nothing should ever accumulate this much, since 400ms triggers a span. It is
# here so that a caller who never reaches the threshold leaks bounded memory
# rather than unbounded.
MAX_BUFFER_S = 4.0


class ProsodyBuffer:
    """Chunks in, tracks out, never in anyone's way.

    Synthesised audio arrives in chunks that know nothing about sentences, and
    prosody is a property of spans longer than a chunk. This accumulates until
    there is enough to say something true - a signalled sentence boundary, or
    MIN_ANALYSIS_S of new audio - and returns a track exactly once for each
    span of speech.

    It sits downstream of playback and holds no lock, does no I/O and blocks on
    nothing. `push` is a numpy call on a few hundred milliseconds of samples;
    the audio it was handed has already gone to the caller, and if this were to
    fall behind, the correct consequence is motion that lags, never speech that
    stutters.
    """

    def __init__(self, budget_s: float = BUDGET_S):
        self._budget = budget_s
        self._sample_rate = 0
        self._samples = np.zeros(0, dtype=np.float32)
        self._tail = 0  # leading samples already covered by a previous track
        self._t0 = 0.0
        self._ceiling = CEILING_MIN
        self._emitted_to = -np.inf  # speech time already described by a track
        self._last_pause_end = -np.inf  # end of the last pause reported

    @property
    def pending_s(self) -> float:
        """Seconds of audio not yet described by a track."""
        if self._sample_rate <= 0:
            return 0.0
        return (self._samples.size - self._tail) / self._sample_rate

    def reset(self) -> None:
        """Forget everything buffered. For barge-in and end of turn."""
        self._samples = np.zeros(0, dtype=np.float32)
        self._tail = 0
        self._t0 = 0.0
        # Whatever was being said is over. Events from it cannot be repeated by
        # what comes next, and holding the watermarks would only let a caller
        # who restarts its timeline lose the first events of the new utterance.
        self._emitted_to = -np.inf
        self._last_pause_end = -np.inf

    def push(
        self,
        pcm: bytes | bytearray | memoryview | np.ndarray,
        sample_rate: int,
        t: float,
        sentence_end: bool = False,
    ) -> ProsodyTrack | None:
        """Take a chunk; return a track when there is one worth having.

        `t` is where this chunk starts on the speech timeline. It is trusted
        over the buffer's own count, so a caller that drops or re-clocks audio
        gets timestamps that still line up with what was played.
        """
        chunk = _as_samples(pcm)

        if sample_rate != self._sample_rate:
            self._sample_rate = sample_rate
            self.reset()
        elif self._samples.size:
            expected = self._t0 + self._samples.size / sample_rate
            if abs(t - expected) > CONTINUITY_S:
                # Not the continuation of what is buffered. Whatever was said
                # before is over, and stitching the two would report a pause
                # that is really an edit.
                self.reset()

        if self._samples.size == 0:
            self._t0 = t
            self._tail = 0

        if chunk.size:
            self._samples = np.concatenate([self._samples, chunk])

        limit = int(MAX_BUFFER_S * sample_rate)
        if self._samples.size > limit:
            dropped = self._samples.size - limit
            self._samples = self._samples[dropped:]
            self._tail = max(0, self._tail - dropped)
            self._t0 += dropped / sample_rate

        pending = self._samples.size - self._tail
        if pending <= 0:
            return None
        if not sentence_end and pending / sample_rate < MIN_ANALYSIS_S:
            return None

        return self._emit()

    def flush(self) -> ProsodyTrack | None:
        """Analyse whatever is left. For the end of a turn."""
        if self._sample_rate <= 0 or self._samples.size - self._tail <= 0:
            return None
        return self._emit()

    @staticmethod
    def _trailing_silence_s(track: ProsodyTrack) -> float:
        """How much silence the span ends in, and so how much is still open."""
        if track.energy.size == 0:
            return 0.0
        reversed_silence = (track.energy < SILENCE_FLOOR)[::-1]
        # An entirely silent span is all still open: the pause may have started
        # before it. MAX_BUFFER_S is what stops that growing forever, and a
        # silence long enough to hit it is not a pause in anything.
        count = len(reversed_silence) if reversed_silence.all() else int(np.argmin(reversed_silence))
        return count * track.hop_s + FRAME_S if count else 0.0

    def _emit(self) -> ProsodyTrack:
        """Analyse the buffer, keep a tail, and never repeat an event.

        The tail is analysed again with the next span - that is what it is for
        - so events landing in it would otherwise be reported twice, and the
        head would nod twice on one syllable. The watermark is in speech time
        rather than in samples so it survives the caller re-clocking a chunk.

        Stress and pauses are deduplicated differently, and the difference is
        not cosmetic. A stress peak inside the previous span had its chance
        there, so the watermark settles it. A pause did not: one reaching the
        end of a span is deliberately left unclassified, and a pause ending
        exactly on the boundary would then be dropped by both spans and never
        reported at all. So a pause is recognised by where it starts, and a
        second look at one already reported overlaps it and is discarded.
        """
        track, self._ceiling = _analyse(
            self._samples,
            self._sample_rate,
            self._t0,
            self._budget,
            self._ceiling,
        )

        stress = [peak for peak in track.stress if peak.t > self._emitted_to]
        pauses = [
            pause
            for pause in track.pauses
            if pause.t0 >= self._last_pause_end - track.hop_s
        ]
        if pauses:
            self._last_pause_end = pauses[-1].t1

        end = self._t0 + self._samples.size / self._sample_rate
        self._emitted_to = end

        # A silence still in progress is kept whole rather than tailed off,
        # because _pauses refuses to classify a run that reaches the edge of
        # the span - its real length is in the audio that has not arrived. A
        # breath is longer than a span, so with a fixed tail the longest and
        # most useful pause of the three could never be recognised at all. The
        # speech before it still goes out now; only the silence waits, which is
        # the part that is genuinely not yet knowable.
        keep_s = TAIL_S + self._trailing_silence_s(track)
        keep = min(self._samples.size, int(keep_s * self._sample_rate))
        self._samples = self._samples[self._samples.size - keep :].copy()
        self._tail = keep
        self._t0 = end - keep / self._sample_rate

        return ProsodyTrack(
            t0=track.t0,
            duration=track.duration,
            hop_s=track.hop_s,
            energy=track.energy,
            f0=track.f0,
            voiced=track.voiced,
            stress=stress,
            pauses=pauses,
            contour=track.contour,
            speech_rate=track.speech_rate,
            degraded=track.degraded,
        )
