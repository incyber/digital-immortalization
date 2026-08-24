from avatar.marking.manifest import (
    DIGITAL_SOURCE_TYPE,
    ModelRef,
    SessionManifest,
    watermark_payload_for,
)
from avatar.marking.watermark import PAYLOAD_BYTES


def a_manifest(**over):
    base = {
        "session_id": "s-1",
        "avatar_id": "colon",
        "avatar_display_name": "Cristóbal Colón",
        "consent_record_id": "c-1",
        "rights_holder": "Public domain",
        "started_at": "2026-08-24T12:00:00Z",
        "models": [ModelRef(name="whisper-small", role="speech-to-text")],
        "watermark_payload": "6176747201aabbcc",
    }
    base.update(over)
    return SessionManifest(**base)


def test_declares_the_output_as_algorithmic_media():
    actions = a_manifest().to_dict()["assertions"][0]["data"]["actions"]
    assert actions[0]["digitalSourceType"] == DIGITAL_SOURCE_TYPE


def test_records_the_consent_it_was_produced_under():
    data = {a["label"]: a["data"] for a in a_manifest().to_dict()["assertions"]}
    assert data["avatar.consent"]["consent_record_id"] == "c-1"
    assert data["avatar.consent"]["rights_holder"] == "Public domain"


def test_records_every_contributing_model():
    manifest = a_manifest(
        models=[
            ModelRef(name="whisper-small", role="speech-to-text"),
            ModelRef(name="llama3.2:3b", role="language"),
            ModelRef(name="piper", role="text-to-speech"),
        ]
    )
    data = {a["label"]: a["data"] for a in manifest.to_dict()["assertions"]}
    assert len(data["avatar.models"]["models"]) == 3
    assert {m["role"] for m in data["avatar.models"]["models"]} == {
        "speech-to-text",
        "language",
        "text-to-speech",
    }


def test_payload_is_the_right_length_for_the_watermark():
    assert len(watermark_payload_for("session-abc")) == PAYLOAD_BYTES


def test_payload_is_stable_for_a_session():
    assert watermark_payload_for("s-1") == watermark_payload_for("s-1")


def test_payload_differs_between_sessions():
    assert watermark_payload_for("s-1") != watermark_payload_for("s-2")


def test_payload_does_not_leak_the_session_id():
    # Frames travel further than manifests do. The identifier in the pixels is
    # a hash, so a recovered frame links to a session record without carrying
    # anything readable on its own.
    payload = watermark_payload_for("session-abc")
    assert b"session" not in payload
