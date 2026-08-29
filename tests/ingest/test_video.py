"""Frames out of a clip.

The behaviour worth defending is the spacing. Adjacent frames of a video are
nearly identical, so taking a run of them adds duplicates and no coverage -
which the photo checks then reject one at a time, making a good clip look like
a bad one.
"""


import cv2
import numpy as np
import pytest

from avatar.ingest.video import (
    MAX_FRAMES,
    VideoError,
    extract_frames,
    is_video,
)


def _clip(tmp_path, seconds: float, fps: int = 25, size=(160, 120)):
    """A real encoded clip, because the decoder is what is being tested."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        pytest.skip("no mp4 encoder available")
    for i in range(int(seconds * fps)):
        frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


@pytest.mark.parametrize(
    "content_type,filename,expected",
    [
        ("video/mp4", "a.mp4", True),
        ("video/quicktime", "a.mov", True),
        ("", "phone-clip.MOV", True),
        ("image/jpeg", "a.jpg", False),
        ("application/octet-stream", "a.txt", False),
    ],
)
def test_video_is_recognised_by_type_or_extension(content_type, filename, expected):
    # Phones send an empty or odd content type often enough that the extension
    # has to be a fallback rather than a nicety.
    assert is_video(content_type, filename) is expected


def test_a_short_clip_yields_frames_spread_across_it(tmp_path):
    frames = extract_frames(_clip(tmp_path, seconds=4.0))

    # Four seconds at the half-second floor, not one per decoded frame.
    assert 6 <= len(frames) <= 9
    assert all(f.startswith(b"\xff\xd8") for f in frames), "frames must be JPEG"


def test_a_long_clip_is_capped(tmp_path):
    frames = extract_frames(_clip(tmp_path, seconds=60.0), min_gap_s=0.1)

    assert len(frames) <= MAX_FRAMES


def test_something_that_is_not_a_video_is_rejected():
    with pytest.raises(VideoError):
        extract_frames(b"this is not a video")


def test_frames_differ_from_each_other(tmp_path):
    """Spacing exists to produce different frames, so check they are.

    Compared after decoding. The first bytes of a JPEG are the same header on
    every frame, so comparing raw bytes at the front proves nothing.
    """
    frames = extract_frames(_clip(tmp_path, seconds=4.0))
    means = {
        round(float(cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR).mean()))
        for f in frames
    }
    assert len(means) > 1


@pytest.mark.asyncio
async def test_the_uploaded_clip_becomes_the_base_the_renderer_drives(tmp_path):
    """The head moves in the result because it moved in the recording.

    This is the whole difference between a talking head and a photograph with
    a mouth on it, and it is what every product doing this well actually does:
    animate the person's own footage rather than synthesise motion onto a still.
    """
    from avatar.ingest.finalise import _copy_source_clip
    from avatar.storage.keys import source_clip_key

    class Store:
        def __init__(self, contents):
            self.contents = contents

        async def get(self, tenant, key):
            if key not in self.contents:
                raise FileNotFoundError(key)
            return self.contents[key]

    key = source_clip_key("tenant-1", "set-1")
    store = Store({key: b"the-original-footage"})

    clip = await _copy_source_clip(store, "tenant-1", "set-1", tmp_path)

    assert clip == tmp_path / "base.mp4"
    assert clip.read_bytes() == b"the-original-footage"


@pytest.mark.asyncio
async def test_no_clip_is_the_ordinary_case_not_a_failure(tmp_path):
    """Somebody who uploaded photographs still gets an avatar."""
    from avatar.ingest.finalise import _copy_source_clip

    class Empty:
        async def get(self, tenant, key):
            raise FileNotFoundError(key)

    assert await _copy_source_clip(Empty(), "tenant-1", "set-1", tmp_path) is None
    assert not (tmp_path / "base.mp4").exists()
