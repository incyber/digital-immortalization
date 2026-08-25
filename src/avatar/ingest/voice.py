"""The voice reference: a recording of the person, used to clone their voice.

Until now a recreated parent has spoken in a stock voice. For this product
that is not a quality gap, it is the wrong person - you have rebuilt their
face and given them a stranger's voice, which is worse than an obviously
synthetic one because it is confidently incorrect.

Cloning needs a short, clean sample. Families usually have one: a voicemail, a
video, an old recording. The checks here are the ones that decide whether a
clone will work, applied at upload while the customer can still choose a
different file.

Deliberately not checked: who is speaking. Nothing here can verify that the
voice belongs to the person being recreated. That is what the consent record
is for, and conflating a technical check with a rights check would make the
gate look stronger than it is.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum

# Zero-shot cloning needs a few seconds. Below this there is not enough of the
# voice to characterise it; above about a minute adds nothing and costs
# storage and upload time.
MIN_SECONDS = 4.0
IDEAL_SECONDS = 15.0
MAX_SECONDS = 120.0

# Below this the recording is too quiet to be a usable reference. Measured as
# RMS in dBFS.
MIN_LOUDNESS_DBFS = -40.0

# A reference that clips is distorted, and the clone inherits the distortion.
MAX_PEAK_DBFS = -0.5

# Cloning quality falls off below this. Phone recordings are usually 8k, which
# is why a voicemail often disappoints and a video usually does not.
MIN_SAMPLE_RATE = 16000

ACCEPTED_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4",
    "audio/m4a", "audio/x-m4a", "audio/ogg", "audio/flac", "video/mp4",
}


class VoiceProblem(str, Enum):
    TOO_SHORT = "shorter than four seconds"
    TOO_LONG = "longer than two minutes"
    TOO_QUIET = "too quiet to characterise the voice"
    CLIPPED = "distorted by clipping"
    LOW_SAMPLE_RATE = "recorded below 16kHz, which limits how well it clones"
    UNREADABLE = "could not be read as audio"
    SILENT = "no speech detected"


@dataclass
class VoiceVerdict:
    """Whether this recording can produce a usable clone."""

    duration_s: float = 0.0
    sample_rate: int = 0
    loudness_dbfs: float = -120.0
    peak_dbfs: float = -120.0
    problems: list[VoiceProblem] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.problems

    @property
    def quality(self) -> str:
        """A word for the customer, not a score.

        'Good enough' is what somebody with one surviving voicemail needs to
        hear; a percentage would invite them to hunt for a better file that
        does not exist.
        """
        if self.problems:
            return "unusable"
        if self.duration_s < IDEAL_SECONDS or self.sample_rate < 22050:
            return "usable"
        return "good"


def probe(audio_bytes: bytes) -> dict:
    """Duration, sample rate and levels, via ffprobe and ffmpeg.

    Runs the real decoder rather than parsing headers, because the files
    families actually have are voicemails and phone videos in containers that
    header parsers get wrong.

    Written to a temporary file rather than piped. Both tools need to seek to
    read duration and container metadata, and a pipe cannot: piping a valid
    recording reports zero duration and silence, which looks exactly like a
    corrupt file.
    """
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as handle:
        handle.write(audio_bytes)
        path = handle.name

    try:
        return _probe_path(path)
    finally:
        os.unlink(path)


def _probe_path(path: str) -> dict:
    # check=False throughout: a non-zero exit on a broken file is an expected
    # outcome the verdict reports, not an exception to propagate.
    meta = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,duration",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1", path,
        ],
        capture_output=True, timeout=60, check=False,
    )

    values: dict[str, str] = {}
    for line in meta.stdout.decode(errors="ignore").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values.setdefault(key.strip(), value.strip())

    levels = subprocess.run(
        # -v info, not error: volumedetect reports its measurements at info
        # level, so quietening ffmpeg makes every recording look silent.
        ["ffmpeg", "-v", "info", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, timeout=120, check=False,
    )
    stderr = levels.stderr.decode(errors="ignore")

    def _level(marker: str) -> float:
        for line in stderr.splitlines():
            if marker in line:
                try:
                    return float(line.split(marker)[1].split("dB")[0].strip(": "))
                except (ValueError, IndexError):
                    return -120.0
        return -120.0

    def _float(key: str) -> float:
        try:
            return float(values.get(key, "") or 0.0)
        except ValueError:
            return 0.0

    return {
        "duration_s": _float("duration"),
        "sample_rate": int(values.get("sample_rate") or 0),
        "loudness_dbfs": _level("mean_volume"),
        "peak_dbfs": _level("max_volume"),
    }


def inspect_voice(audio_bytes: bytes) -> VoiceVerdict:
    """Judge one recording."""
    if not audio_bytes:
        return VoiceVerdict(problems=[VoiceProblem.UNREADABLE])

    try:
        measured = probe(audio_bytes)
    except Exception:  # noqa: BLE001 - any decode failure is the same answer
        return VoiceVerdict(problems=[VoiceProblem.UNREADABLE])

    verdict = VoiceVerdict(
        duration_s=measured["duration_s"],
        sample_rate=measured["sample_rate"],
        loudness_dbfs=measured["loudness_dbfs"],
        peak_dbfs=measured["peak_dbfs"],
    )

    if verdict.duration_s <= 0 or verdict.sample_rate <= 0:
        verdict.problems.append(VoiceProblem.UNREADABLE)
        return verdict

    if verdict.duration_s < MIN_SECONDS:
        verdict.problems.append(VoiceProblem.TOO_SHORT)
    if verdict.duration_s > MAX_SECONDS:
        verdict.problems.append(VoiceProblem.TOO_LONG)
    if verdict.loudness_dbfs <= -90.0:
        verdict.problems.append(VoiceProblem.SILENT)
    elif verdict.loudness_dbfs < MIN_LOUDNESS_DBFS:
        verdict.problems.append(VoiceProblem.TOO_QUIET)
    if verdict.peak_dbfs > MAX_PEAK_DBFS:
        verdict.problems.append(VoiceProblem.CLIPPED)
    if verdict.sample_rate < MIN_SAMPLE_RATE:
        verdict.problems.append(VoiceProblem.LOW_SAMPLE_RATE)

    return verdict


def normalise(audio_bytes: bytes, target_rate: int = 24000) -> bytes:
    """Mono WAV at the rate the cloner expects, loudness-normalised.

    Done once at upload rather than on every synthesis. Loudness normalisation
    matters more than it sounds: a quiet reference produces a clone that
    whispers, because the model reproduces level along with timbre.
    """
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as src:
        src.write(audio_bytes)
        src_path = src.name
    out_path = src_path + ".wav"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-i", src_path,
                "-ac", "1", "-ar", str(target_rate),
                "-af", "loudnorm=I=-18:TP=-2:LRA=11",
                out_path,
            ],
            capture_output=True, timeout=300, check=False,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            raise ValueError("could not convert that recording")
        return pathlib.Path(out_path).read_bytes()
    finally:
        os.unlink(src_path)
        if os.path.exists(out_path):
            os.unlink(out_path)
