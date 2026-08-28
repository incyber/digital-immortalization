"""The GPU step is optional, and its failure is not the build's failure.

Both properties matter more than the rendering itself. Every developer machine
and every test runs with no endpoint configured, so an unconfigured build must
be the old build exactly. And a GPU outage at signup must cost an avatar its
head motion, not its existence - the plate assets are already written and
already callable by the time this runs.
"""

import numpy as np
import pytest

from avatar.config import Settings
from avatar.ingest import gpu_assets
from avatar.ingest.gpu_assets import (
    GpuUnavailable,
    attach_base_clip,
    is_configured,
    render_base_clip,
)


@pytest.fixture
def frame():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


def test_an_unconfigured_build_does_not_call_the_gpu(tmp_path, frame):
    cfg = Settings(runpod_api_key="", runpod_endpoint_id="")

    assert is_configured(cfg) is False
    assert attach_base_clip(cfg, tmp_path, frame) is None
    assert not (tmp_path / "base.mp4").exists()


def test_rendering_without_an_endpoint_is_an_error_not_a_silence(frame):
    """Distinct from a GPU failure, because the log should say which it was."""
    with pytest.raises(GpuUnavailable, match="no RunPod endpoint"):
        render_base_clip(Settings(runpod_api_key="k", runpod_endpoint_id=""), frame)


def test_a_gpu_failure_leaves_the_avatar_callable(tmp_path, frame, monkeypatch):
    cfg = Settings(runpod_api_key="k", runpod_endpoint_id="e")

    def explode(*_args, **_kwargs):
        raise GpuUnavailable("the endpoint is on fire")

    monkeypatch.setattr(gpu_assets, "render_base_clip", explode)

    assert attach_base_clip(cfg, tmp_path, frame) is None
    assert not (tmp_path / "base.mp4").exists()


def test_a_rendered_clip_is_written_beside_the_assets(tmp_path, frame, monkeypatch):
    monkeypatch.setattr(gpu_assets, "render_base_clip", lambda *a, **k: b"mp4-bytes")

    clip = attach_base_clip(
        Settings(runpod_api_key="k", runpod_endpoint_id="e"), tmp_path, frame
    )

    assert clip == tmp_path / "base.mp4"
    assert clip.read_bytes() == b"mp4-bytes"
    # Nothing half-written left behind by the staged rename.
    assert [p.name for p in tmp_path.iterdir()] == ["base.mp4"]


def test_the_job_payload_carries_an_image_and_a_motion_template(frame, monkeypatch):
    sent = {}

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, payload, **_kwargs):
            sent.update(payload)
            return type(
                "R", (), {"output": {"video": "AAA="}, "execution_ms": 1, "cost": 0.0}
            )()

    from avatar.gpu import serverless

    monkeypatch.setattr(serverless, "ServerlessClient", FakeClient)

    render_base_clip(Settings(runpod_api_key="k", runpod_endpoint_id="e"), frame)

    assert sent["task"] == "animate"
    assert sent["image"]
    # Per-avatar, not baked into the worker image: a template extracted from a
    # real person would make every avatar move like that same stranger.
    assert sent["motion_template"]
