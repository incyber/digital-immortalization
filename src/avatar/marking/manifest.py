"""Per-session declaration of what produced the synthetic media.

The frame watermark says "this is synthetic." This says what made it, for whom,
and under what authority - the record that is actually useful when a family, a
regulator, or a court asks.

Field names follow C2PA assertion vocabulary where one exists, so that emitting
a signed C2PA manifest later is a serialisation change rather than a redesign.
This is not itself C2PA: there is no signature, no certificate chain, no
hard binding to the media. Calling it C2PA would be a false claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# C2PA's controlled vocabulary for the action that produced the asset.
DIGITAL_SOURCE_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


@dataclass
class ModelRef:
    """One model that contributed to the output."""

    name: str
    role: str  # "speech-to-text" | "language" | "text-to-speech" | "renderer" | "vision"


@dataclass
class SessionManifest:
    """What a single call produced, and on whose authority."""

    session_id: str
    avatar_id: str
    avatar_display_name: str
    consent_record_id: str
    rights_holder: str
    started_at: str
    models: list[ModelRef] = field(default_factory=list)
    watermark_payload: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_generator": "avatar/0.1.0",
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "digitalSourceType": DIGITAL_SOURCE_TYPE,
                                "softwareAgent": "avatar/0.1.0",
                            }
                        ]
                    },
                },
                {
                    "label": "avatar.session",
                    "data": {
                        "session_id": self.session_id,
                        "avatar_id": self.avatar_id,
                        "avatar_display_name": self.avatar_display_name,
                        "started_at": self.started_at,
                        "watermark_payload": self.watermark_payload,
                    },
                },
                {
                    "label": "avatar.consent",
                    "data": {
                        "consent_record_id": self.consent_record_id,
                        "rights_holder": self.rights_holder,
                    },
                },
                {
                    "label": "avatar.models",
                    "data": {"models": [asdict(m) for m in self.models]},
                },
            ],
        }


def watermark_payload_for(session_id: str) -> bytes:
    """The 8 bytes carried in every frame of this session.

    Four bytes of magic, one of format version, three derived from the session
    id - enough to tie a recovered frame back to a specific call without
    putting anything identifying in the pixels themselves.
    """
    from hashlib import blake2s

    digest = blake2s(session_id.encode(), digest_size=3).digest()
    return b"avtr" + bytes([1]) + digest
