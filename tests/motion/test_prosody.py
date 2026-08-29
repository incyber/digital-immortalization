"""Proof that the delivery is being read, and not invented.

Prosody drives things a viewer notices immediately when they are wrong: the
head nods on the stressed syllable, the brows lift on a question, the body
breathes in the gap. Each of those is a claim about the audio, and a claim that
is silently wrong is worse than one that is absent - a nod on the wrong word
reads as a tic, where no nod at all reads as composure.

So the signals here are synthesised rather than recorded. A 200Hz sine has a
pitch of 200Hz and nothing else, an amplitude bump is at a time this file
chose, and a silence is exactly as long as it says. That makes every assertion
below a statement about correctness rather than about a recording someone once
listened to and agreed sounded right.

Two of these tests are not about correctness at all. One measures how long the
analysis takes, because this runs on the CPU beside a live call and the whole
budget mechanism is worthless if nobody notices it drifting. The other feeds it
empty, single-sample, clipped and DC-offset input, because a NaN reaching the
renderer does not look like a bug, it looks like a face that stopped.
"""

import time
from itertools import pairwise

import numpy as np
import pytest

from avatar.motion.prosody import (
    CEILING_MIN,
    FRAME_S,
    MIN_ANALYSIS_S,
    STRESS_MIN_GAP_S,
    TAIL_S,
    ProsodyBuffer,
    analyse,
)

SR = 24000


def pcm(signal: np.ndarray) -> bytes:
    """Float in roughly -1..1 as the signed 16-bit PCM the synthesiser emits."""
    return np.clip(signal * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def tone(hz: float, seconds: float, amp: float = 0.5, sample_rate: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return amp * np.sin(2 * np.pi * hz * t)


def voice(hz: float, seconds: float, amp: float = 0.4, sample_rate: int = SR) -> np.ndarray:
    """A tone with harmonics, which is what a voiced sound actually is.

    A pure sine is the easy case for an autocorrelation tracker. Harmonics are
    where octave errors come from, so most of the pitch tests use this.
    """
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    phase = 2 * np.pi * hz * t
    stack = sum(gain * np.sin(n * phase) for n, gain in enumerate([1.0, 0.5, 0.3, 0.15], start=1))
    return amp * stack / 1.95


def sweep(start: float, end: float, seconds: float, sample_rate: int = SR) -> np.ndarray:
    """A tone whose pitch slides from start to end, geometrically."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    instantaneous = start * (end / start) ** (t / seconds)
    return 0.5 * np.sin(2 * np.pi * np.cumsum(instantaneous) / sample_rate)


def bumps(
    centres: tuple[float, ...] = (0.35, 0.95, 1.55),
    seconds: float = 2.0,
    carrier: float = 180.0,
    height: float = 0.8,
) -> np.ndarray:
    """Continuous speech-like tone with louder moments at known times."""
    t = np.arange(int(seconds * SR)) / SR
    envelope = np.full_like(t, 0.2)
    for centre in centres:
        envelope = envelope + height * np.exp(-0.5 * ((t - centre) / 0.06) ** 2)
    return envelope * np.sin(2 * np.pi * carrier * t)


def gap(seconds: float, around: float = 0.5) -> np.ndarray:
    """Speech, a silence of a known length, then speech again."""
    speech = voice(160.0, around)
    return np.concatenate([speech, np.zeros(int(seconds * SR)), speech])


# Generous, so the tests are about the analysis and not about how fast the
# machine running them happens to be.
PATIENT = 1.0


# ---------------------------------------------------------------- pitch


def test_a_steady_tone_is_voiced_at_its_own_pitch():
    track = analyse(pcm(tone(200.0, 1.0)), SR, budget_s=PATIENT)

    assert track.voiced.mean() > 0.95
    assert abs(float(np.median(track.f0[track.voiced])) - 200.0) < 5.0


@pytest.mark.parametrize("hz", [70.0, 90.0, 120.0, 150.0, 200.0, 260.0, 320.0])
def test_pitch_is_recovered_across_the_whole_search_range(hz):
    """Both ends included: a low male voice and a raised female one."""
    track = analyse(pcm(voice(hz, 0.6)), SR, budget_s=PATIENT)

    assert track.voiced.mean() > 0.9
    assert abs(float(np.median(track.f0[track.voiced])) - hz) < hz * 0.03


def test_pitch_is_not_reported_an_octave_out():
    """The failure every autocorrelation tracker has, and the one that matters.

    A periodic signal correlates with itself just as well at twice its period,
    and the flank leading up to the true peak crosses any threshold before the
    peak does. Either mistake puts the reported pitch somewhere the voice never
    went, which moves brows on the wrong words rather than not moving them.
    """
    track = analyse(pcm(voice(200.0, 1.0)), SR, budget_s=PATIENT)
    found = track.f0[track.voiced]

    assert len(found) > 50
    assert found.min() > 150.0, "halved somewhere"
    assert found.max() < 260.0, "doubled or read off the flank somewhere"


@pytest.mark.parametrize("sample_rate", [16000, 22050, 24000, 48000])
def test_pitch_does_not_depend_on_the_sample_rate(sample_rate):
    """TTS backends differ, and a nod cannot depend on which one is loaded."""
    track = analyse(pcm(voice(130.0, 0.6, sample_rate=sample_rate)), sample_rate, budget_s=PATIENT)

    assert abs(float(np.median(track.f0[track.voiced])) - 130.0) < 4.0


def test_white_noise_is_not_voiced():
    """Fricatives and room noise are aperiodic and must be left alone.

    Reporting a pitch for them would put a pitch accent on every 's'.
    """
    noise = np.random.default_rng(0).normal(0.0, 0.3, SR)
    track = analyse(pcm(noise), SR, budget_s=PATIENT)

    assert track.voiced.mean() < 0.05
    assert np.all(track.f0[~track.voiced] == 0.0)


def test_silence_says_nothing():
    track = analyse(pcm(np.zeros(SR)), SR, budget_s=PATIENT)

    assert track.stress == []
    assert not track.voiced.any()
    assert np.all(track.f0 == 0.0)
    assert track.speech_rate == 0.0
    assert np.all(np.isfinite(track.energy))


# ---------------------------------------------------------------- stress


def test_stress_lands_where_the_delivery_leans():
    """Within 50ms, because that is about a frame of video at 25fps.

    A nod that peaks a frame either side of the syllable still reads as being
    on it; two frames out and it reads as a reaction to it.
    """
    track = analyse(pcm(bumps(centres=(0.35, 0.95, 1.55))), SR, budget_s=PATIENT)
    found = [peak.t for peak in track.stress]

    assert len(found) == 3
    for expected, actual in zip((0.35, 0.95, 1.55), found):
        assert abs(actual - expected) < 0.05, f"{actual:.3f} should be near {expected}"


def test_stress_strength_is_a_usable_zero_to_one():
    track = analyse(pcm(bumps()), SR, budget_s=PATIENT)

    assert all(0.0 <= peak.strength <= 1.0 for peak in track.stress)
    assert all(peak.strength > 0.0 for peak in track.stress)


def test_two_stress_peaks_are_never_closer_than_the_refractory_gap():
    """Nods that overlap read as a shiver rather than as emphasis."""
    close = tuple(np.arange(0.3, 1.8, 0.09))
    track = analyse(pcm(bumps(centres=close, height=0.6)), SR, budget_s=PATIENT)
    times = [peak.t for peak in track.stress]

    assert len(times) > 2
    assert min(np.diff(times)) >= STRESS_MIN_GAP_S - 1e-6


def test_the_stronger_of_two_close_prominences_is_the_one_kept():
    """Picked strongest first, not in time order.

    Scanning forward would let the weaker syllable win by arriving first and
    then suppress the one a listener actually hears as stressed.
    """
    t = np.arange(int(1.5 * SR)) / SR
    envelope = np.full_like(t, 0.2)
    envelope += 0.5 * np.exp(-0.5 * ((t - 0.60) / 0.04) ** 2)
    envelope += 0.9 * np.exp(-0.5 * ((t - 0.68) / 0.04) ** 2)
    track = analyse(pcm(envelope * np.sin(2 * np.pi * 180 * t)), SR, budget_s=PATIENT)

    assert len(track.stress) == 1
    assert abs(track.stress[0].t - 0.68) < 0.05


def test_an_unvarying_tone_has_nothing_to_be_stressed_about():
    """The guard against inventing structure out of the tracker's own jitter.

    Standardising a constant signal divides by a spread that is only rounding
    error, and every ripple becomes a confident peak.
    """
    track = analyse(pcm(voice(150.0, 2.0)), SR, budget_s=PATIENT)

    assert track.stress == []


# ---------------------------------------------------------------- pauses


@pytest.mark.parametrize(
    ("length", "kind"),
    [(0.12, "beat"), (0.20, "beat"), (0.40, "clause"), (0.50, "clause"), (0.90, "breath")],
)
def test_a_silence_is_classified_by_how_long_it_is(length, kind):
    track = analyse(pcm(gap(length)), SR, budget_s=PATIENT)

    assert [pause.kind for pause in track.pauses] == [kind]


def test_a_pause_is_measured_where_the_silence_actually_is():
    """A frame only falls below the floor once its whole window is quiet, so
    the measured run is a window shorter than the gap unless that is added back
    - and a breath timed half a window late arrives during the next word."""
    track = analyse(pcm(gap(0.5, around=0.6)), SR, budget_s=PATIENT)
    pause = track.pauses[0]

    assert abs(pause.t0 - 0.6) < 0.05
    assert abs(pause.duration - 0.5) < 0.05


def test_a_gap_too_short_to_be_a_pause_is_not_one():
    """The closure of a stop consonant is silent and is not a pause."""
    track = analyse(pcm(gap(0.05)), SR, budget_s=PATIENT)

    assert track.pauses == []


def test_a_silence_running_off_the_end_is_not_called_a_breath():
    """Its real length is in the next chunk. Guessing it inhales mid-word."""
    trailing = np.concatenate([voice(160.0, 0.5), np.zeros(int(1.0 * SR))])
    track = analyse(pcm(trailing), SR, budget_s=PATIENT)

    assert track.pauses == []


def test_several_pauses_are_found_in_one_span():
    speech = voice(160.0, 0.4)
    signal = np.concatenate(
        [speech, np.zeros(int(0.15 * SR)), speech, np.zeros(int(0.7 * SR)), speech]
    )
    track = analyse(pcm(signal), SR, budget_s=PATIENT)

    assert [pause.kind for pause in track.pauses] == ["beat", "breath"]
    assert track.pauses[0].t1 < track.pauses[1].t0


# ---------------------------------------------------------------- contour


def test_a_rising_final_pitch_reads_as_a_question():
    assert analyse(pcm(sweep(120.0, 260.0, 1.0)), SR, budget_s=PATIENT).contour == "rise"


def test_a_falling_final_pitch_reads_as_a_statement():
    assert analyse(pcm(sweep(260.0, 120.0, 1.0)), SR, budget_s=PATIENT).contour == "fall"


def test_a_level_delivery_reads_flat():
    assert analyse(pcm(voice(150.0, 1.0)), SR, budget_s=PATIENT).contour == "flat"


def test_a_late_fall_and_rise_is_seen_as_both():
    """The shape a single straight line calls flat and a listener calls a
    question. It is checked separately for exactly that reason."""
    t = np.arange(int(1.2 * SR)) / SR
    instantaneous = 200 * np.exp(-1.0 * np.minimum(t, 1.0)) * np.exp(2.5 * np.maximum(t - 1.0, 0))
    signal = 0.5 * np.sin(2 * np.pi * np.cumsum(instantaneous) / SR)

    assert analyse(pcm(signal), SR, budget_s=PATIENT).contour == "fall-rise"


def test_a_contour_is_only_ever_one_of_the_four_names():
    for signal in (np.zeros(SR), voice(150.0, 1.0), sweep(120.0, 260.0, 1.0), bumps()):
        track = analyse(pcm(signal), SR, budget_s=PATIENT)
        assert track.contour in {"fall", "rise", "flat", "fall-rise"}


def test_only_the_end_of_the_span_decides_the_contour():
    """A sentence that fell for a second and rose at the last moment is a
    question. Fitting the whole span would call it a statement."""
    signal = np.concatenate([sweep(260.0, 150.0, 1.2), sweep(150.0, 280.0, 0.4)])

    assert analyse(pcm(signal), SR, budget_s=PATIENT).contour == "rise"


# ---------------------------------------------------------------- energy, rate


def test_energy_is_normalised_into_zero_to_one():
    for signal in (tone(200.0, 0.5, amp=0.02), tone(200.0, 0.5, amp=1.0), bumps()):
        energy = analyse(pcm(signal), SR, budget_s=PATIENT).energy
        assert energy.min() >= 0.0
        assert energy.max() <= 1.0


def test_a_quiet_voice_still_reaches_the_top_of_the_range():
    """The ceiling adapts to the speaker, as viseme.py's does.

    Normalising against full scale would leave a quiet voice permanently
    expressionless, which is the wrong way round: quiet speech is not
    unemphatic speech.
    """
    loud = analyse(pcm(voice(150.0, 0.8, amp=0.9)), SR, budget_s=PATIENT)
    quiet = analyse(pcm(voice(150.0, 0.8, amp=0.03)), SR, budget_s=PATIENT)

    assert quiet.energy.max() > 0.9
    assert abs(float(quiet.energy.mean()) - float(loud.energy.mean())) < 0.1


def test_room_tone_is_not_promoted_into_speech():
    """The ceiling adapts, but only down to a floor. Without it, an empty room
    is normalised until its own hiss has stress peaks in it."""
    hiss = np.random.default_rng(1).normal(0.0, CEILING_MIN / 32767.0 / 8.0, SR)
    track = analyse(pcm(hiss), SR, budget_s=PATIENT)

    assert track.energy.max() < 0.5
    assert track.stress == []


@pytest.mark.parametrize(("modulation", "low", "high"), [(4.0, 2.5, 5.5), (8.0, 6.0, 10.0)])
def test_speech_rate_follows_the_syllable_rate(modulation, low, high):
    t = np.arange(int(2.0 * SR)) / SR
    envelope = 0.5 * (1 + 0.8 * np.sin(2 * np.pi * modulation * t - np.pi / 2))
    track = analyse(pcm(0.6 * envelope * np.sin(2 * np.pi * 200 * t)), SR, budget_s=PATIENT)

    assert low < track.speech_rate < high


def test_a_pause_does_not_make_the_speech_around_it_read_as_slow():
    """Nuclei per voiced second, not per elapsed second. The director scales
    motion by this, and a thoughtful pause is not a slower talker."""
    t = np.arange(int(1.0 * SR)) / SR
    envelope = 0.5 * (1 + 0.8 * np.sin(2 * np.pi * 5.0 * t - np.pi / 2))
    speech = 0.6 * envelope * np.sin(2 * np.pi * 200 * t)

    dense = analyse(pcm(speech), SR, budget_s=PATIENT)
    interrupted = analyse(
        pcm(np.concatenate([speech, np.zeros(int(0.8 * SR)), speech])), SR, budget_s=PATIENT
    )

    assert abs(dense.speech_rate - interrupted.speech_rate) < 1.0


# ---------------------------------------------------------------- robustness


# Everything a synthesiser, a codec or a truncated write has ever handed this
# kind of code. Built by name so a failure reads as the case that failed rather
# than as a screenful of PCM.
AWKWARD = {
    "empty": lambda: b"",
    "one sample": lambda: np.array([1234], dtype=np.int16).tobytes(),
    "odd byte count": lambda: pcm(tone(200.0, 0.05))[:-1],
    "a single 10ms chunk": lambda: pcm(tone(200.0, 0.010)),
    "all zeros": lambda: pcm(np.zeros(SR)),
    "full-scale clipping": lambda: np.full(SR, 32767, dtype=np.int16).tobytes(),
    "alternating full scale": lambda: np.tile(np.array([32767, -32768], dtype=np.int16), SR // 2),
    "dc offset": lambda: np.full(SR, 8000, dtype=np.int16).tobytes(),
    "dc offset under speech": lambda: pcm(voice(150.0, 0.5) + 0.4),
    "one loud click": lambda: pcm(np.eye(1, SR, SR // 2)[0]),
    "a clipped square wave": lambda: pcm(np.sign(tone(200.0, 0.5)) * 4.0),
}


@pytest.mark.parametrize("name", list(AWKWARD))
def test_nothing_ever_produces_a_nan_or_an_infinity(name):
    """A NaN reaching the renderer does not look like a bug. It looks like a
    face that stopped, halfway through a sentence, and stayed stopped."""
    track = analyse(AWKWARD[name](), SR, budget_s=PATIENT)

    assert np.all(np.isfinite(track.energy))
    assert np.all(np.isfinite(track.f0))
    assert np.isfinite(track.speech_rate)
    assert np.isfinite(track.duration)
    assert np.all(np.isfinite([peak.t for peak in track.stress]))
    assert np.all(np.isfinite([peak.strength for peak in track.stress]))
    assert np.all(np.isfinite([pause.t0 for pause in track.pauses]))
    assert np.all(np.isfinite([pause.t1 for pause in track.pauses]))
    assert track.energy.min(initial=0.0) >= 0.0
    assert track.energy.max(initial=0.0) <= 1.0


@pytest.mark.parametrize("samples", [0, 1, 240, 960, SR])
def test_the_frame_arrays_always_agree_in_length(samples):
    track = analyse(pcm(tone(200.0, samples / SR)), SR, budget_s=PATIENT)

    assert len(track.energy) == len(track.f0) == len(track.voiced)
    assert len(track.times) == len(track.energy)


def test_timestamps_are_on_the_speech_timeline_not_the_wall_clock():
    """The reason pose.py is timed against speech: a nod that must peak on a
    syllable has to be scheduled before that syllable is heard, which is only
    possible if the timestamps refer to the audio."""
    here = analyse(pcm(bumps()), SR, t0=0.0, budget_s=PATIENT)
    later = analyse(pcm(bumps()), SR, t0=42.0, budget_s=PATIENT)

    assert later.t0 == 42.0
    assert abs(float(later.times[0]) - (42.0 + FRAME_S / 2)) < 1e-6
    assert len(here.stress) == len(later.stress)
    for a, b in zip(here.stress, later.stress):
        assert abs((b.t - a.t) - 42.0) < 1e-6


def test_a_sample_rate_of_zero_is_a_programming_error():
    """Not clamped quietly. A wrong rate makes every timestamp wrong, and a
    track that is silently on the wrong timeline is worse than a traceback."""
    with pytest.raises(ValueError):
        analyse(pcm(tone(200.0, 0.5)), 0)


# ---------------------------------------------------------------- the budget


def test_the_budget_is_paid_for_out_of_pitch_and_not_out_of_frames():
    """Past the budget it stops tracking pitch, and says so. What it must not
    do is return late, or return less of the audio than it was given."""
    full = analyse(pcm(bumps()), SR, budget_s=PATIENT)
    rushed = analyse(pcm(bumps()), SR, budget_s=0.0)

    assert rushed.degraded and not full.degraded
    assert not rushed.voiced.any()
    assert np.all(rushed.f0 == 0.0)
    assert len(rushed.energy) == len(full.energy)
    assert np.allclose(rushed.energy, full.energy)
    assert [round(p.t, 2) for p in rushed.stress] == [round(p.t, 2) for p in full.stress]
    assert [p.kind for p in rushed.pauses] == [p.kind for p in full.pauses]


def test_a_degraded_track_is_still_a_usable_one():
    track = analyse(pcm(bumps()), SR, budget_s=0.0)

    assert track.contour == "flat"
    assert track.speech_rate > 0.0
    assert np.all(np.isfinite(track.energy))


@pytest.mark.parametrize("seconds", [MIN_ANALYSIS_S, 1.0])
def test_an_ordinary_span_fits_in_the_default_budget(seconds):
    """The budget exists for the machine having a bad moment, not for the
    normal case. If the normal case degrades, the pitch stage has regressed."""
    assert not analyse(pcm(voice(150.0, seconds)), SR).degraded


def test_one_second_of_speech_is_analysed_in_a_fraction_of_a_frame():
    """The regression test for the thing the budget protects against.

    25ms is under a frame at 25fps, and this measures the full path with pitch
    tracking on - the degraded path is fast by construction and proves nothing.
    Best of several runs, because a scheduler hiccup on a shared machine is not
    a regression in this module.
    """
    data = pcm(bumps(centres=tuple(np.arange(0.1, 1.0, 0.2)), seconds=1.0))
    analyse(data, SR, budget_s=PATIENT)  # warm numpy's FFT plan cache

    elapsed = []
    for _ in range(7):
        started = time.perf_counter()
        track = analyse(data, SR, budget_s=PATIENT)
        elapsed.append(time.perf_counter() - started)

    assert not track.degraded
    assert min(elapsed) < 0.025, f"{min(elapsed) * 1000:.1f}ms for one second of audio"


# ---------------------------------------------------------------- the buffer


def chunks(signal: np.ndarray, ms: int = 100):
    """Audio as a synthesiser delivers it: fixed-size, sentence-unaware."""
    step = ms * SR // 1000
    for index in range(0, len(signal), step):
        yield signal[index : index + step], index / SR


def test_the_buffer_says_nothing_until_it_has_enough_to_say():
    buffer = ProsodyBuffer(budget_s=PATIENT)
    speech = voice(160.0, MIN_ANALYSIS_S - 0.05)

    for chunk, t in chunks(speech):
        assert buffer.push(pcm(chunk), SR, t) is None


def test_the_buffer_emits_exactly_once_when_it_has_enough():
    buffer = ProsodyBuffer(budget_s=PATIENT)
    emitted = [buffer.push(pcm(chunk), SR, t) for chunk, t in chunks(voice(160.0, 0.5))]

    assert [track is not None for track in emitted] == [False, False, False, True, False]
    track = emitted[3]
    assert track.t0 == 0.0
    assert track.duration >= MIN_ANALYSIS_S


def test_a_sentence_boundary_emits_before_the_threshold():
    """The end of a sentence is the most useful moment there is - it is where
    the contour lives - and waiting 400ms for it would report a question after
    the answer has started."""
    buffer = ProsodyBuffer(budget_s=PATIENT)

    assert buffer.push(pcm(voice(160.0, 0.1)), SR, 0.0) is None
    track = buffer.push(pcm(voice(160.0, 0.1)), SR, 0.1, sentence_end=True)

    assert track is not None
    assert buffer.pending_s == 0.0


def test_the_buffer_reports_each_stress_peak_exactly_once():
    """The tail is analysed twice on purpose. Reporting its events twice would
    nod twice on one syllable."""
    buffer = ProsodyBuffer(budget_s=PATIENT)
    found = []
    for chunk, t in chunks(bumps(centres=(0.3, 0.9, 1.5))):
        track = buffer.push(pcm(chunk), SR, t)
        if track is not None:
            found.extend(peak.t for peak in track.stress)

    assert len(found) == 3
    for expected, actual in zip((0.3, 0.9, 1.5), sorted(found)):
        assert abs(actual - expected) < 0.05


@pytest.mark.parametrize(
    ("length", "kind"), [(0.12, "beat"), (0.40, "clause"), (0.90, "breath")]
)
def test_the_buffer_classifies_a_pause_longer_than_its_own_span(length, kind):
    """The one that a fixed tail would have made impossible.

    A span is 400ms of new audio and 200ms of tail, and a silence is only
    classified once it has ended, so with a fixed tail a breath - which is
    longer than that by definition - could never be recognised. The buffer
    holds an unfinished silence instead of tailing it off, which is why the
    longest and most useful of the three kinds arrives at all.
    """
    buffer = ProsodyBuffer(budget_s=PATIENT)
    found = []
    for chunk, t in chunks(gap(length, around=0.8)):
        track = buffer.push(pcm(chunk), SR, t)
        if track is not None:
            found.extend(track.pauses)

    assert [pause.kind for pause in found] == [kind]
    assert abs(found[0].duration - length) < 0.05
    assert abs(found[0].t0 - 0.8) < 0.05


def test_an_endless_silence_is_not_a_pause_and_is_not_a_leak():
    """Held silence is bounded. Nothing that runs for the length of a call may
    grow with it, and a silence that never ends is not a pause in anything."""
    buffer = ProsodyBuffer(budget_s=PATIENT)
    for index in range(60):
        track = buffer.push(pcm(np.zeros(int(0.5 * SR))), SR, index * 0.5)
        assert track is None or track.pauses == []

    assert buffer.pending_s <= 4.0


def test_consecutive_spans_overlap_by_the_tail():
    """A syllable straddling a chunk boundary needs its beginning to still be
    measurable, and the pitch tracker needs a full window before its first new
    frame. Both come out of keeping the tail."""
    buffer = ProsodyBuffer(budget_s=PATIENT)
    tracks = [
        track
        for track in (buffer.push(pcm(chunk), SR, t) for chunk, t in chunks(voice(160.0, 1.2)))
        if track is not None
    ]

    assert len(tracks) >= 2
    for earlier, later in pairwise(tracks):
        assert abs((earlier.t0 + earlier.duration) - (later.t0 + TAIL_S)) < 0.02


def test_the_buffer_follows_the_speech_timeline_it_is_given():
    """Not its own sample count, and certainly not the clock: the caller knows
    where in the utterance this audio sits, and this does not."""
    buffer = ProsodyBuffer(budget_s=PATIENT)
    track = None
    for chunk, t in chunks(bumps(centres=(0.3,), seconds=0.5)):
        track = buffer.push(pcm(chunk), SR, 100.0 + t) or track

    assert track is not None
    assert track.t0 == 100.0
    assert all(100.0 <= peak.t <= 100.5 for peak in track.stress)


def test_a_jump_in_the_timeline_starts_a_new_span():
    """Barge-in and turn changes leave a stale tail behind. Stitching it onto
    the next utterance would report a pause that is really an edit."""
    buffer = ProsodyBuffer(budget_s=PATIENT)
    buffer.push(pcm(voice(160.0, 0.2)), SR, 0.0)
    track = buffer.push(pcm(voice(160.0, 0.5)), SR, 9.0)

    assert track is not None
    assert track.t0 == 9.0
    assert track.pauses == []


def test_a_reset_forgets_what_was_buffered():
    buffer = ProsodyBuffer(budget_s=PATIENT)
    buffer.push(pcm(voice(160.0, 0.3)), SR, 0.0)
    buffer.reset()

    assert buffer.pending_s == 0.0
    assert buffer.push(pcm(voice(160.0, 0.1)), SR, 0.3) is None


def test_flush_has_nothing_to_say_about_an_empty_buffer():
    buffer = ProsodyBuffer(budget_s=PATIENT)

    assert buffer.flush() is None
    buffer.push(pcm(voice(160.0, 0.2)), SR, 0.0)
    assert buffer.flush() is not None
    assert buffer.flush() is None


def test_the_buffer_does_not_grow_without_bound():
    """It cannot happen through the normal path, since 400ms triggers a span.
    It is bounded anyway, because a live call runs for hours and the one thing
    a downstream stage must not do is consume the process."""
    buffer = ProsodyBuffer(budget_s=PATIENT)
    track = buffer.push(pcm(voice(160.0, 30.0)), SR, 0.0)

    assert track is not None
    assert track.duration <= 4.0
    assert buffer.pending_s <= TAIL_S + 1e-6


def test_a_change_of_sample_rate_starts_over():
    buffer = ProsodyBuffer(budget_s=PATIENT)
    buffer.push(pcm(voice(160.0, 0.3)), SR, 0.0)

    assert buffer.push(pcm(voice(160.0, 0.3, sample_rate=16000)), 16000, 0.3) is None
    assert buffer.pending_s == pytest.approx(0.3, abs=0.01)
