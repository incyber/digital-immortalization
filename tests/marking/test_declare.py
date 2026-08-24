"""The declaration must reach subscribers by a path the video codec cannot
destroy, and must fail loudly rather than leaving a call undeclared."""
import json

import pytest

from avatar.marking.declare import (
    ATTR_GENERATOR,
    ATTR_SOURCE_TYPE,
    ATTR_SYNTHETIC,
    ATTR_WATERMARK,
    declare,
)
from avatar.marking.manifest import DIGITAL_SOURCE_TYPE, ModelRef, SessionManifest


class FakeParticipant:
    def __init__(self, fail=False):
        self.attributes = None
        self.metadata = None
        self._fail = fail

    async def set_attributes(self, attributes):
        if self._fail:
            raise RuntimeError("not connected")
        self.attributes = attributes

    async def set_metadata(self, metadata):
        self.metadata = metadata


def a_manifest():
    return SessionManifest(
        session_id="s-1",
        avatar_id="colon",
        avatar_display_name="Cristóbal Colón",
        consent_record_id="c-1",
        rights_holder="Public domain",
        started_at="2026-08-24T12:00:00Z",
        models=[ModelRef(name="llama3.2:3b", role="language")],
        watermark_payload="6176747201aabbcc",
    )


async def test_attributes_flag_the_stream_as_synthetic():
    p = FakeParticipant()
    await declare(p, a_manifest())
    assert p.attributes[ATTR_SYNTHETIC] == "true"
    assert p.attributes[ATTR_SOURCE_TYPE] == DIGITAL_SOURCE_TYPE
    assert p.attributes[ATTR_GENERATOR].startswith("avatar/")
    assert p.attributes[ATTR_WATERMARK] == "6176747201aabbcc"


async def test_metadata_carries_the_full_manifest():
    p = FakeParticipant()
    await declare(p, a_manifest())
    payload = json.loads(p.metadata)
    labels = {a["label"] for a in payload["assertions"]}
    assert "c2pa.actions" in labels
    assert "avatar.consent" in labels


async def test_metadata_is_set_before_the_flag():
    # A client seeing synthetic=true must be able to read the manifest, so the
    # manifest has to be published first.
    order = []

    class Ordered(FakeParticipant):
        async def set_metadata(self, metadata):
            order.append("metadata")
            await super().set_metadata(metadata)

        async def set_attributes(self, attributes):
            order.append("attributes")
            await super().set_attributes(attributes)

    await declare(Ordered(), a_manifest())
    assert order == ["metadata", "attributes"]


async def test_a_failed_declaration_is_raised_not_swallowed():
    # A dropped frame degrades video. An undeclared call is the thing Article
    # 50 forbids, so the caller has to be able to see it and stop.
    with pytest.raises(RuntimeError):
        await declare(FakeParticipant(fail=True), a_manifest())
