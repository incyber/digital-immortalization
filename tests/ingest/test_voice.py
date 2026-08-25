"""Voice reference checks.

A recreated parent speaking in a stock voice is not a quality gap, it is the
wrong person. These pin the checks that decide whether a clone will work,
applied at upload while the customer can still pick a different file.
"""

import subprocess

import pytest

from avatar.ingest.voice import (
    IDEAL_SECONDS,
    MIN_SECONDS,
    VoiceProblem,
    inspect_voice,
    normalise,
)


def a_recording(seconds=15.0, rate=24000, volume="0.5", freq=180):
    """Synthetic audio at a known duration, rate and level."""
    return subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={seconds}:sample_rate={rate}",
            "-af", f"volume={volume}", "-f", "wav", "-",
        ],
        capture_output=True, check=False,
    ).stdout


def test_a_clean_recording_is_usable():
    v = inspect_voice(a_recording())
    assert v.usable, [p.value for p in v.problems]
    assert v.quality == "good"
    assert v.duration_s == pytest.approx(15.0, abs=0.2)
    assert v.sample_rate == 24000


def test_measurements_are_real_not_header_guesses():
    """Piping to ffprobe reports zero duration and silence on a valid file,
    which looks identical to corruption. The probe writes to disk instead."""
    v = inspect_voice(a_recording(seconds=8.0))
    assert v.duration_s > 1.0
    assert v.loudness_dbfs > -90.0, "a real level must be measured, not defaulted"


def test_too_short_is_refused():
    v = inspect_voice(a_recording(seconds=MIN_SECONDS - 2))
    assert VoiceProblem.TOO_SHORT in v.problems


def test_too_long_is_refused():
    # Nothing is gained past a minute or two, and it costs storage and upload.
    v = inspect_voice(a_recording(seconds=130))
    assert VoiceProblem.TOO_LONG in v.problems


def test_a_near_silent_recording_is_refused():
    v = inspect_voice(a_recording(volume="0.0005"))
    assert VoiceProblem.TOO_QUIET in v.problems or VoiceProblem.SILENT in v.problems


def test_a_low_rate_recording_is_flagged():
    """Phone voicemails are often 8kHz, which is why they disappoint."""
    v = inspect_voice(a_recording(rate=8000))
    assert VoiceProblem.LOW_SAMPLE_RATE in v.problems


def test_something_that_is_not_audio_is_refused():
    assert VoiceProblem.UNREADABLE in inspect_voice(b"this is not audio").problems


def test_empty_input_is_refused():
    assert VoiceProblem.UNREADABLE in inspect_voice(b"").problems


def test_a_short_but_otherwise_fine_recording_is_only_usable():
    """The word shown to somebody with one surviving voicemail matters.

    'usable' tells them it will work; a percentage would send them hunting for
    a better file that does not exist.
    """
    v = inspect_voice(a_recording(seconds=IDEAL_SECONDS - 6))
    assert v.usable
    assert v.quality == "usable"


def test_normalisation_raises_a_quiet_reference():
    """A quiet reference clones to a voice that whispers: the model
    reproduces level along with timbre."""
    quiet = inspect_voice(a_recording(volume="0.02"))
    fixed = inspect_voice(normalise(a_recording(volume="0.02")))
    assert fixed.loudness_dbfs > quiet.loudness_dbfs + 5


def test_normalisation_produces_mono_at_the_expected_rate():
    result = inspect_voice(normalise(a_recording(rate=44100), target_rate=24000))
    assert result.sample_rate == 24000
    assert result.usable


def test_normalisation_refuses_something_unreadable():
    with pytest.raises(ValueError, match="could not convert"):
        normalise(b"not audio at all")


def test_nothing_here_claims_to_verify_who_is_speaking():
    """Deliberate. No technical check can establish that a voice belongs to
    the person being recreated - that is what the consent record is for, and
    conflating the two would make the gate look stronger than it is."""
    import avatar.ingest.voice as module

    source = module.__doc__ or ""
    assert "consent record" in source
    assert not hasattr(module, "verify_speaker")
