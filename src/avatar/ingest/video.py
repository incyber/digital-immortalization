"""Frames out of an uploaded clip.

A short video is a better source than a handful of photographs and much easier
to supply: twenty seconds of someone talking contains more usable angles,
expressions and mouth positions than most families have in stills, and nobody
has to hunt through an album for it.

Frames are spread across the whole clip rather than taken consecutively.
Adjacent frames of a video are nearly identical, so a run of them adds
duplicates and no coverage, which is exactly what the photo checks then reject
one by one.

Nothing here judges quality. Every frame goes through the same sharpness, face
and framing checks as an uploaded photograph, so a clip and an album are held
to one standard.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
from loguru import logger

# Ceiling on frames taken from one clip. Beyond this the checks are doing the
# same work on the same face repeatedly.
MAX_FRAMES = 60

# The shortest gap between two frames. Half a second is roughly where a head
# has moved enough for the next frame to be worth having.
MIN_GAP_S = 0.5

# Longest clip accepted. A minute at half-second spacing is already more than
# MAX_FRAMES, so anything longer only costs upload time.
MAX_DURATION_S = 120.0

VIDEO_CONTENT_TYPES = (
    "video/mp4", "video/quicktime", "video/x-m4v", "video/webm", "video/x-matroska",
)


class VideoError(ValueError):
    pass


def is_video(content_type: str, filename: str) -> bool:
    if content_type.split(";")[0].strip().lower() in VIDEO_CONTENT_TYPES:
        return True
    return Path(filename).suffix.lower() in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}


@dataclass(frozen=True)
class ClipPlan:
    """How many frames a clip will yield, and from where, before any decoding.

    Separated from the decoding so that a progress bar has a denominator from
    the first moment. Counting afterwards means the customer watches a bar
    with no end on it for as long as the whole job takes, which is the state
    they read as "nothing is happening".

    The counts come from container metadata alone, which is a seek and a
    header read rather than a decode, so knowing the total is close to free.
    """

    duration_s: float
    offsets: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.offsets)


def plan_frames(
    path: Path | str,
    *,
    max_frames: int = MAX_FRAMES,
    min_gap_s: float = MIN_GAP_S,
) -> ClipPlan:
    """Where in the clip to take frames from, and how many there will be.

    Blocking, and deliberately not async: every caller runs it off the event
    loop. See ingest/video_service.py.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoError("that file could not be read as a video")

        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total <= 0:
            raise VideoError("that video has no readable duration")

        duration = total / fps
        if duration > MAX_DURATION_S:
            raise VideoError(
                f"that clip is {duration:.0f}s; {MAX_DURATION_S:.0f}s is the most that helps"
            )
    finally:
        capture.release()

    # Spacing is whichever is coarser: the gap that fills max_frames across
    # the clip, or the minimum useful gap. A four-second clip therefore
    # yields eight frames rather than sixty near-identical ones.
    gap = max(min_gap_s, duration / max_frames)
    wanted = [i * gap for i in range(int(duration / gap) + 1)]
    return ClipPlan(duration_s=duration, offsets=tuple(wanted[:max_frames]))


class FrameReader:
    """One open decoder, read one frame at a time.

    Exists so that sixty 1080p frames are never all in memory at once. The
    machine this runs on holds the gateway and a speech model in 4GB, and the
    version of this that built a list of every frame first was a plausible
    out-of-memory kill on an entirely ordinary upload - which presents as the
    machine vanishing rather than as an error anybody can read.

    Every method blocks and every one of them is meant to be called through
    asyncio.to_thread. That is safe across threads because the calls are
    sequential: one frame is finished before the next is asked for. It is not
    safe to read from two threads at once, and nothing here does.
    """

    def __init__(self, path: Path | str):
        self._path = str(path)
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self._path)
        if not capture.isOpened():
            capture.release()
            raise VideoError("that file could not be read as a video")
        self._capture = capture

    def read_at(self, seconds: float) -> bytes | None:
        """One JPEG, or None where the decoder could not produce a frame.

        None is ordinary rather than exceptional: a seek past a damaged
        keyframe fails for that offset and the rest of the clip is still good.
        """
        if self._capture is None:
            raise VideoError("the clip was not opened")
        self._capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not encoded:
            return None
        return buffer.tobytes()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def iter_frames(path: Path | str, plan: ClipPlan) -> Iterator[bytes]:
    """Every decodable frame of a plan, one at a time."""
    with FrameReader(path) as reader:
        for seconds in plan.offsets:
            frame = reader.read_at(seconds)
            if frame is not None:
                yield frame


def extract_frames(
    data: bytes,
    *,
    max_frames: int = MAX_FRAMES,
    min_gap_s: float = MIN_GAP_S,
) -> list[bytes]:
    """Evenly spaced JPEG frames from a clip, all of them, in memory.

    Written through a temporary file rather than decoded from memory: the
    decoder needs to seek, and a container that cannot be seeked reports one
    frame and no duration.

    Kept for callers that already hold the whole clip and want the whole
    result. Anything serving a customer should use plan_frames and FrameReader
    instead, so that progress can be reported and one frame is held at a time.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)

    try:
        plan = plan_frames(path, max_frames=max_frames, min_gap_s=min_gap_s)
        frames = list(iter_frames(path, plan))
        if not frames:
            raise VideoError("no frame of that video could be decoded")

        logger.info(f"took {len(frames)} frames from {plan.duration_s:.1f}s of video")
        return frames
    finally:
        path.unlink(missing_ok=True)


def probe_duration(data: bytes) -> float:
    """Seconds, or 0.0 when the container will not say.

    Used to reject an over-long upload before spending time decoding it.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return float(result.stdout.strip() or 0.0)
    except (ValueError, OSError, subprocess.SubprocessError):
        return 0.0
    finally:
        path.unlink(missing_ok=True)
