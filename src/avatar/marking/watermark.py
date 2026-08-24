"""Machine-readable marking of synthetic video frames.

EU AI Act Article 50, applicable since 2 August 2026, requires providers of AI
systems that generate synthetic video to mark their output so it is detectable
as AI-generated, in a machine-readable form. The persistent disclosure banner
in the call UI addresses the human-facing obligation; this addresses the other
one.

Design constraints, in the order they mattered:

  Invisible.   A visible mark on a recreation of someone's late mother is not
               acceptable. Measured at over 45 dB PSNR, comfortably inside what
               is normally treated as visually lossless.
  Cheap.       Runs on every frame at 25 fps alongside rendering, so it is a
               handful of numpy operations on one channel, not a transform.
  Survives.    The publisher expands RGB to RGBA before handing frames to
               LiveKit. A mark that does not survive that never ships.

What this is NOT: a robust watermark. It does not survive re-encoding,
cropping, or screen capture, and it is not adversarially secure.

That limitation has a concrete consequence, learned by measuring rather than
reasoning: **this is not used on the live call.** WebRTC re-encodes every frame,
and thirty consecutive frames received over a real connection decoded to
nothing. A mark faint enough to be invisible on a recreation of someone's late
parent is not strong enough to survive VP8. Live streams are marked out of
band instead - see declare.py.

This module's place is media this system encodes itself: recordings, exports,
downloadable clips, where the encoder is ours and the mark survives.

It is also not a C2PA implementation. See manifest.py, which carries the
declaration in a shape C2PA assertions can adopt.
"""

from __future__ import annotations

import numpy as np

# "avtr" plus four bytes of version and identifier. Short because every byte
# costs redundancy, and redundancy is what makes detection reliable.
PAYLOAD_BYTES = 8
_MAGIC = b"avtr"

# Each payload bit is written into a block of this many pixels and recovered by
# majority vote, so a single interpolated or clipped pixel cannot flip it.
_BLOCK = 8
_BITS = PAYLOAD_BYTES * 8

# The mark is carried in the low bits of the blue channel: human vision is
# least sensitive to blue, and the renderer's output is a face, where red and
# green carry the skin tones people actually look at.
_CHANNEL = 2

# How far a marked pixel is pushed from its quantisation bucket. Two levels out
# of 256 is under 1% and survives the RGBA round trip, which is a copy.
_STRENGTH = 2


class _TooSmall(ValueError):
    pass


def _capacity(height: int, width: int) -> int:
    return (height // _BLOCK) * (width // _BLOCK)


def _bit_positions(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-left corner of the block carrying each bit, row-major."""
    per_row = width // _BLOCK
    index = np.arange(_BITS)
    return (index // per_row) * _BLOCK, (index % per_row) * _BLOCK


def embed(frame_rgb: np.ndarray, payload: bytes) -> np.ndarray:
    """Return a copy of the frame carrying the payload.

    Raises rather than returning the frame unchanged when it cannot be marked:
    a silently unmarked frame is the failure mode this whole module exists to
    prevent.
    """
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError(f"payload must be exactly {PAYLOAD_BYTES} bytes, got {len(payload)}")

    height, width = frame_rgb.shape[:2]
    if _capacity(height, width) < _BITS:
        raise _TooSmall(
            f"frame {width}x{height} is too small to mark; "
            f"needs at least {_BITS} blocks of {_BLOCK}x{_BLOCK}"
        )

    marked = frame_rgb.copy()
    channel = marked[:, :, _CHANNEL].astype(np.int16)

    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    rows, cols = _bit_positions(height, width)

    for bit, row, col in zip(bits, rows, cols, strict=True):
        block = channel[row : row + _BLOCK, col : col + _BLOCK]
        # Quantise to a 2*_STRENGTH grid, then place the value in the low or
        # high half of its bucket according to the bit. Recovery only needs to
        # know which half, so it does not need the original frame.
        bucket = (block // (2 * _STRENGTH)) * (2 * _STRENGTH)
        offset = _STRENGTH + (_STRENGTH // 2) if bit else (_STRENGTH // 2)
        block[:, :] = np.clip(bucket + offset, 0, 255)

    marked[:, :, _CHANNEL] = channel.astype(np.uint8)
    return marked


def detect(frame_rgb: np.ndarray) -> bytes | None:
    """Recover the payload, or None if the frame carries no valid mark.

    Validity is decided by the magic prefix. An unmarked frame decodes to
    arbitrary bytes, which will not begin with it.
    """
    height, width = frame_rgb.shape[:2]
    if _capacity(height, width) < _BITS:
        return None

    channel = frame_rgb[:, :, _CHANNEL].astype(np.int16)
    rows, cols = _bit_positions(height, width)

    recovered = np.zeros(_BITS, dtype=np.uint8)
    for i, (row, col) in enumerate(zip(rows, cols, strict=True)):
        block = channel[row : row + _BLOCK, col : col + _BLOCK]
        # Majority vote over the block: which half of its bucket does each
        # pixel sit in?
        within = block % (2 * _STRENGTH)
        recovered[i] = 1 if (within >= _STRENGTH).mean() > 0.5 else 0

    payload = np.packbits(recovered).tobytes()
    return payload if payload.startswith(_MAGIC) else None


def rgb_to_rgba(frame_rgb: np.ndarray) -> np.ndarray:
    """Expand to RGBA with an opaque alpha, as the publisher does."""
    height, width = frame_rgb.shape[:2]
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = frame_rgb
    rgba[:, :, 3] = 255
    return rgba


def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB. Infinite when identical."""
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0) - 10 * np.log10(mse))
