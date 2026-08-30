"""Provisioning refuses to leave an unsafe endpoint behind.

The one behaviour worth defending here: if the platform stores settings that
differ from what was sent, the endpoint is deleted rather than returned. A bug
in provisioning must cost nothing, never a GPU that bills while idle.
"""

import pytest

from avatar.gpu.provision import Provisioner
from avatar.gpu.serverless import ServerlessError

SAFE = {
    "id": "ep-1",
    "workersMin": 0,
    "workersMax": 1,
    "idleTimeout": 5,
    "executionTimeoutMs": 900_000,
}


# Every provision starts by listing templates, because names are unique per
# account and an identical one is reused rather than recreated.
NO_EXISTING_TEMPLATES: list = []


class FakeCalls(Provisioner):
    """Records requests and replays canned responses in order."""

    def __init__(self, responses):
        super().__init__("key")
        self.responses = [NO_EXISTING_TEMPLATES, *responses]
        self.calls = []

    def _request(self, method, path, **kwargs):
        self.calls.append((method, path))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_an_api_key_is_required():
    with pytest.raises(ServerlessError, match="API key"):
        Provisioner("")


def test_a_verified_endpoint_is_returned():
    provisioner = FakeCalls([{"id": "tpl-1"}, {"id": "ep-1"}, SAFE])

    result = provisioner.provision("worker", "ghcr.io/example/gpu-worker:latest")

    assert result.endpoint_id == "ep-1"
    assert result.template_id == "tpl-1"
    assert result.verified["workersMin"] == 0
    assert ("DELETE", "/endpoints/ep-1") not in provisioner.calls


@pytest.mark.parametrize(
    "stored",
    [
        {**SAFE, "workersMin": 1},
        {**SAFE, "workersMax": 5},
        {**SAFE, "idleTimeout": 600},
        {**SAFE, "executionTimeoutMs": 0},
    ],
    ids=["workersMin", "workersMax", "idleTimeout", "executionTimeout"],
)
def test_an_endpoint_stored_unsafe_is_deleted(stored):
    provisioner = FakeCalls([{"id": "tpl-1"}, {"id": "ep-1"}, stored, {}])

    with pytest.raises(ServerlessError, match="deleted"):
        provisioner.provision("worker", "image")

    assert ("DELETE", "/endpoints/ep-1") in provisioner.calls


def test_an_endpoint_that_cannot_be_read_is_deleted():
    """Unverifiable is treated as unsafe, not as probably fine."""
    provisioner = FakeCalls(
        [{"id": "tpl-1"}, {"id": "ep-1"}, ServerlessError("503"), {}]
    )

    with pytest.raises(ServerlessError):
        provisioner.provision("worker", "image")

    assert ("DELETE", "/endpoints/ep-1") in provisioner.calls


def test_the_endpoint_request_carries_the_safe_settings():
    sent = {}

    class Capture(FakeCalls):
        def _request(self, method, path, **kwargs):
            if path == "/endpoints" and method == "POST":
                sent.update(kwargs.get("json", {}))
            return super()._request(method, path, **kwargs)

    Capture([{"id": "tpl-1"}, {"id": "ep-1"}, SAFE]).provision("worker", "image")

    assert sent["workersMin"] == 0
    assert sent["workersMax"] == 1
    # Long enough for a worker to finish downloading its own code and claim a
    # job. At five seconds the platform stopped it first, every time, and the
    # queue never drained.
    assert sent["idleTimeout"] == 30
    # Milliseconds, which is what the API takes. Sending seconds here would
    # give a 900ms timeout that kills every real job.
    assert sent["executionTimeoutMs"] == 900_000


def test_a_serverless_template_exposes_no_ports():
    """A worker that listens is a Pod habit; nothing should reach it directly."""
    sent = {}

    class Capture(FakeCalls):
        def _request(self, method, path, **kwargs):
            if path == "/templates":
                sent.update(kwargs.get("json", {}))
            return super()._request(method, path, **kwargs)

    Capture([{"id": "tpl-1"}, {"id": "ep-1"}, SAFE]).provision("worker", "image")

    assert sent["ports"] == []
    assert sent["isServerless"] is True


def test_an_identical_template_is_reused_rather_than_recreated():
    """Re-provisioning after a failure is normal, and names are unique."""
    existing = [
        {
            "id": "tpl-existing",
            "name": "worker-template",
            "imageName": "image",
            "containerDiskInGb": 90,
        }
    ]
    provisioner = FakeCalls([{"id": "ep-1"}, SAFE])
    provisioner.responses[0] = existing

    result = provisioner.provision("worker", "image")

    assert result.template_id == "tpl-existing"
    assert ("POST", "/templates") not in provisioner.calls


def test_a_template_with_different_settings_is_a_conflict():
    """Silently running on someone else's disk size is how the first one failed."""
    existing = [
        {
            "id": "tpl-existing",
            "name": "worker-template",
            "imageName": "a-different-image",
            "containerDiskInGb": 40,
        }
    ]
    provisioner = FakeCalls([])
    provisioner.responses[0] = existing

    with pytest.raises(ServerlessError, match="different settings"):
        provisioner.provision("worker", "image")
