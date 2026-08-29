"""One suite, run against every RendererStage implementation.

A backend that passes this is substitutable in the pipeline. A backend that
does not is not, regardless of how good its output looks.
"""
import time

from avatar.renderer.base import AudioChunk

SR = 16000
ONE_SECOND = AudioChunk(pcm=b"\x00\x00" * SR, sample_rate=SR)
TEN_SECONDS = AudioChunk(pcm=b"\x00\x00" * SR * 10, sample_rate=SR)


async def test_render_emits_frames_at_declared_size(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets)
    await r.prepare("test")
    frames = [f async for f in r.render(ONE_SECOND)]
    assert frames, "one second of audio must produce frames"
    assert all((f.width, f.height) == r.size for f in frames)
    assert all(len(f.data) == f.width * f.height * 3 for f in frames)


async def test_frame_count_tracks_audio_duration(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets)
    await r.prepare("test")
    n = len([f async for f in r.render(ONE_SECOND)])
    assert abs(n - r.fps) <= 1


async def test_idle_is_endless(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets)
    await r.prepare("test")
    it = r.idle()
    seen = [await anext(it) for _ in range(50)]
    assert len(seen) == 50
    await it.aclose()


async def test_cancel_returns_within_100ms(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets)
    await r.prepare("test")
    agen = r.render(TEN_SECONDS)
    await anext(agen)
    start = time.perf_counter()
    await r.cancel()
    elapsed = time.perf_counter() - start
    await agen.aclose()
    assert elapsed < 0.100, f"cancel took {elapsed * 1000:.1f} ms"


async def test_cancel_actually_stops_the_stream(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets)
    await r.prepare("test")
    agen = r.render(TEN_SECONDS)
    await anext(agen)
    await r.cancel()
    remaining = [f async for f in agen]
    assert len(remaining) < r.fps, "cancel must stop delivery, not merely flag it"


async def test_render_after_cancel_still_works(renderer_factory, tmp_assets):
    r = renderer_factory(tmp_assets)
    await r.prepare("test")
    await r.cancel()
    frames = [f async for f in r.render(ONE_SECOND)]
    assert frames, "a cancelled stage must be reusable for the next turn"


async def test_a_renderer_may_ignore_motion_without_breaking(renderer_factory, tmp_assets):
    """Motion is optional, and that is what keeps backends substitutable.

    A mouth-only renderer has nothing to do with a head pose. Making motion a
    method rather than an argument to render() means those backends stay valid
    and this suite stays unchanged as 3D arrives.
    """
    r = renderer_factory(tmp_assets)
    await r.prepare("test")

    if hasattr(r, "attach_motion"):
        await r.attach_motion(None)

    frames = [f async for f in r.render(ONE_SECOND)]
    assert frames


async def test_audio_carries_its_place_on_the_speech_timeline():
    """Without this, motion can only react; with it, motion can anticipate."""
    assert ONE_SECOND.t0 == 0.0

    later = AudioChunk(pcm=ONE_SECOND.pcm, sample_rate=SR, t0=12.5)
    assert later.t0 == 12.5
    assert later.duration_s == ONE_SECOND.duration_s
