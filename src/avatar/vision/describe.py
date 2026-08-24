"""Turns one camera frame into one sentence.

Called off the turn path. Every failure returns an empty string rather than
raising, because the caller's correct response to any vision problem is to
keep the previous observation and carry on talking.
"""

from __future__ import annotations

import base64

import cv2
import httpx
import numpy as np
from loguru import logger

# Kept tight on purpose. A long description crowds the persona out of the
# system prompt and invites the model to narrate the room instead of talking.
_PROMPTS = {
    "en": (
        "Describe what you see in one short sentence: the person's clothing, "
        "posture, any gesture they are making, and the setting. "
        "No preamble, no speculation about who they are."
    ),
    "es": (
        "Describe lo que ves en una sola frase corta: la ropa de la persona, "
        "su postura, cualquier gesto que haga, y el entorno. "
        "Sin preambulo y sin especular sobre quien es."
    ),
}


def prompt_for(locale: str) -> str:
    return _PROMPTS.get(locale, _PROMPTS["en"])

# Frames are sent at this long-edge size. Larger costs tokens and time for
# detail that does not change the sentence.
MAX_EDGE = 512
JPEG_QUALITY = 70


def encode_frame(frame_rgb: np.ndarray) -> bytes:
    """Downscale and JPEG-encode a frame for transport to the vision model."""
    h, w = frame_rgb.shape[:2]
    scale = MAX_EDGE / max(h, w)
    if scale < 1.0:
        frame_rgb = cv2.resize(
            frame_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        return b""
    return buf.tobytes()


async def describe_frame(
    jpeg: bytes, model: str, base_url: str, timeout_s: float = 8.0, locale: str = "en"
) -> str:
    """One sentence describing the frame, or "" on any failure.

    Ollama's native /api/chat is used rather than the OpenAI-compatible shim
    because the shim's image handling has varied between versions and this is
    the one call in the system that carries an image.
    """
    if not jpeg:
        return ""

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt_for(locale),
                "images": [base64.b64encode(jpeg).decode("ascii")],
            }
        ],
        "options": {"num_predict": 60, "temperature": 0.2},
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
            response.raise_for_status()
            return (response.json().get("message") or {}).get("content", "").strip()
    except Exception as exc:  # noqa: BLE001 - every failure has the same handling
        logger.warning(f"vision call failed, keeping previous observation: {exc}")
        return ""
