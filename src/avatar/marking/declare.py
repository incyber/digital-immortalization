"""Out-of-band declaration that the published media is synthetic.

Why this exists rather than only a pixel watermark: WebRTC re-encodes every
frame. Measured against this project's own pipeline, a spatial watermark strong
enough to be invisible does not survive VP8 - thirty consecutive frames
received over a real connection decoded to nothing. In-band marking of a live
stream is not a thing that works, and shipping it while claiming Article 50
coverage would have been worse than shipping nothing, because it would have
looked done.

So the live stream is marked where a machine can actually read it: participant
attributes and participant metadata, both carried by the signalling channel,
both delivered to every subscriber before the first frame, and neither touched
by the video codec.

The pixel watermark keeps its place for media this system encodes itself -
recordings and exports - where the encoder is ours and the mark survives. See
watermark.py.
"""

from __future__ import annotations

import json

from livekit import rtc
from loguru import logger

from avatar.marking.manifest import SessionManifest

# Read by clients and by anything inspecting the room server-side. Flat string
# values because participant attributes are string-to-string.
ATTR_SYNTHETIC = "synthetic"
ATTR_SOURCE_TYPE = "digital_source_type"
ATTR_GENERATOR = "claim_generator"
ATTR_WATERMARK = "watermark_payload"


async def declare(participant: rtc.LocalParticipant, manifest: SessionManifest) -> None:
    """Attach the synthetic-content declaration to this participant.

    Attributes carry the flags a client can branch on cheaply. Metadata carries
    the full manifest for anything that wants the detail.
    """
    from avatar.marking.manifest import DIGITAL_SOURCE_TYPE

    try:
        # Metadata first, then the attribute flag. Both propagate to
        # subscribers independently, so ordering them this way means a client
        # that sees the flag can rely on the manifest already being there.
        await participant.set_metadata(json.dumps(manifest.to_dict()))
        await participant.set_attributes(
            {
                ATTR_SYNTHETIC: "true",
                ATTR_SOURCE_TYPE: DIGITAL_SOURCE_TYPE,
                ATTR_GENERATOR: "avatar/0.1.0",
                ATTR_WATERMARK: manifest.watermark_payload,
            }
        )
        logger.info("published synthetic-content declaration")
    except Exception as exc:
        # Unlike a dropped frame, this cannot be retried per-frame, and a call
        # that proceeds undeclared is the outcome the obligation forbids. The
        # caller decides; this reports.
        logger.error(f"could not publish synthetic-content declaration: {exc}")
        raise
