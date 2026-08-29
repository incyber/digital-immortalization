"""The GPU half of an avatar build: a base clip with real head motion.

What this produces and what it does not. LivePortrait warps the customer's own
photograph along a motion template, so the result is that person moving rather
than a plausible stranger. That clip is the base a lip-sync renderer animates.
It is not, on its own, lip-sync: the plate renderer that ships today pastes a
mouth into a fixed box, and a fixed box is wrong the moment the head moves. So
the clip is written alongside the plate assets as `base.mp4` and left for the
renderer that can use it, rather than substituted into one that cannot.

Configuration decides whether this runs at all. With no RunPod endpoint set the
build is exactly what it was, which is the state every developer machine and
every test is in. Nothing here is on the turn path; this is signup-time work.
"""

from __future__ import annotations

import asyncio
import base64
import pickle
import tempfile
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from avatar.config import Settings
from avatar.ingest.idle_motion import IdleStyle, build_idle_template

# How long a base clip runs. Long enough that a loop is not obvious, short
# enough that a build is seconds of GPU rather than minutes: LivePortrait costs
# roughly real time per frame, and this is billed by the second.
CLIP_SECONDS = 6.0

# The longest an avatar build will wait for the GPU. Deliberately far below the
# platform's own execution timeout: a build is a person watching a progress bar,
# and fifteen minutes of that is indistinguishable from a hang. If the clip is
# not ready by then the avatar ships without head motion.
BUILD_WAIT_S = 240

# What the source photograph is encoded at before upload. The worker downsizes
# to 1024 on the long edge anyway; sending less than this loses detail the crop
# actually uses.
JPEG_QUALITY = 95


class GpuUnavailable(RuntimeError):
    """No endpoint is configured, or the job did not produce a clip."""


def is_configured(cfg: Settings) -> bool:
    return bool(cfg.runpod_api_key and cfg.runpod_endpoint_id)


def render_base_clip(
    cfg: Settings,
    base_rgb: np.ndarray,
    *,
    style: IdleStyle | None = None,
    seed: int = 0,
) -> bytes:
    """One photograph in, an mp4 of that person with head motion out.

    Raises rather than returning None on failure. A caller that wants the build
    to continue without a clip should catch it: the distinction between "not
    configured" and "the GPU failed" matters to the log, and a None return
    flattens both into the same silence.
    """
    if not is_configured(cfg):
        raise GpuUnavailable("no RunPod endpoint is configured")

    # Imported here so the module can be read, and the flow tested, on a
    # machine with no GPU client configured at all.
    from avatar.gpu.serverless import ServerlessClient

    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not ok:
        raise GpuUnavailable("the base frame could not be encoded")

    # Generated here rather than shipped in the worker image. Which style an
    # avatar moves with is a per-avatar decision, and a template extracted from
    # a real person would make every avatar move like that same stranger.
    template = pickle.dumps(
        build_idle_template(seconds=CLIP_SECONDS, style=style, seed=seed)
    )

    client = ServerlessClient(cfg.runpod_api_key, cfg.runpod_endpoint_id)
    result = client.run(
        {
            "task": "animate",
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "motion_template": base64.b64encode(template).decode("ascii"),
        },
        wait_s=BUILD_WAIT_S,
    )

    output = result.output or {}
    if output.get("error"):
        raise GpuUnavailable(f"animation failed: {output['error']}")

    video = output.get("video")
    if not video:
        raise GpuUnavailable(f"job finished {result.state.value} with no video")

    logger.info(
        f"base clip rendered in {result.execution_ms}ms (~${result.cost:.4f})"
    )
    return base64.b64decode(video)


async def attach_base_clip(
    cfg: Settings,
    destination: Path,
    base_rgb: np.ndarray,
    *,
    style: IdleStyle | None = None,
) -> Path | None:
    """Render the clip and write it beside the assets. None if it did not run.

    Failure here does not fail the build. The plate assets are already written
    and already callable by the time this is reached, so a GPU outage costs the
    avatar its head motion rather than its existence.

    Async, and the blocking client runs on a thread. The first version called
    it directly from the gateway's event loop, where its polling sleep stopped
    every other request in the process - including the one asking how the build
    was going.
    """
    if not is_configured(cfg):
        return None

    try:
        video = await asyncio.to_thread(render_base_clip, cfg, base_rgb, style=style)
    except Exception as exc:  # noqa: BLE001 - any GPU failure is non-fatal here
        logger.warning(f"no base clip for {destination.name}: {exc}")
        return None

    # Written through a temporary file in the same directory and renamed, so a
    # crash mid-write cannot leave a truncated mp4 that a renderer later treats
    # as valid.
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination, suffix=".mp4", delete=False
    ) as handle:
        handle.write(video)
        staged = Path(handle.name)

    clip = destination / "base.mp4"
    staged.replace(clip)
    logger.info(f"base clip written to {clip} ({len(video) // 1024}KB)")
    return clip
