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
from pathlib import Path

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


def extract_frames(
    data: bytes,
    *,
    max_frames: int = MAX_FRAMES,
    min_gap_s: float = MIN_GAP_S,
) -> list[bytes]:
    """Evenly spaced JPEG frames from a clip.

    Written through a temporary file rather than decoded from memory: the
    decoder needs to seek, and a container that cannot be seeked reports one
    frame and no duration.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)

    try:
        capture = cv2.VideoCapture(str(path))
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

        # Spacing is whichever is coarser: the gap that fills max_frames across
        # the clip, or the minimum useful gap. A four-second clip therefore
        # yields eight frames rather than sixty near-identical ones.
        gap = max(min_gap_s, duration / max_frames)
        wanted = [i * gap for i in range(int(duration / gap) + 1)]

        frames: list[bytes] = []
        for seconds in wanted[:max_frames]:
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if encoded:
                frames.append(buffer.tobytes())

        capture.release()
        if not frames:
            raise VideoError("no frame of that video could be decoded")

        logger.info(f"took {len(frames)} frames from {duration:.1f}s of video")
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
