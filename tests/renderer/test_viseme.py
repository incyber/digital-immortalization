import numpy as np

from avatar.renderer.base import AudioChunk
from avatar.renderer.viseme import VisemeRenderer

SR = 16000


def tone(secs=1.0, amp=20000, freq=220.0):
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.int16).tobytes()


async def test_loud_audio_opens_the_mouth(tmp_assets):
    r = VisemeRenderer(tmp_assets)
    await r.prepare("t")
    [f async for f in r.render(AudioChunk(tone(), SR))]
    assert max(r.last_plate_indices()) > 0


async def test_silence_keeps_the_mouth_closed(tmp_assets):
    r = VisemeRenderer(tmp_assets)
    await r.prepare("t")
    [f async for f in r.render(AudioChunk(b"\x00\x00" * SR, SR))]
    assert set(r.last_plate_indices()) == {0}


async def test_idle_frames_are_closed_mouth(tmp_assets):
    r = VisemeRenderer(tmp_assets)
    await r.prepare("t")
    it = r.idle()
    first = await anext(it)
    await it.aclose()

    x, y, w, h = tmp_assets.mouth_box
    rendered = np.frombuffer(first.data, dtype=np.uint8).reshape(first.height, first.width, 3)
    assert np.array_equal(rendered[y : y + h, x : x + w], tmp_assets.plates[0])


async def test_idle_loop_advances_through_the_clip(tmp_assets):
    # A frozen idle frame reads as a dropped connection, so the loop must move.
    r = VisemeRenderer(tmp_assets)
    await r.prepare("t")
    it = r.idle()
    a = await anext(it)
    b = await anext(it)
    await it.aclose()
    assert a.data != b.data
