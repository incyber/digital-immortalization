"""Renting a warm GPU without ever leaving one unattended.

The ordering is the design. A pod created before its lease exists has a window
in which nothing is enforcing anything, and that window is the one the review
could not close. So the tests here are mostly about *when* things happen, not
whether they succeed.
"""

import pytest

from avatar.gpu.lease import LEASE_TTL_S
from avatar.gpu.pod import PodError, PodRenter, RentedPod


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.expiries = {}
        self.order = []

    async def set(self, key, value, ex=None):
        self.order.append(("set", key))
        self.store[key] = value
        self.expiries[key] = ex

    async def delete(self, key):
        self.order.append(("delete", key))
        self.store.pop(key, None)


class FakeRenter(PodRenter):
    def __init__(self, redis, responses):
        super().__init__("key", redis)
        self.responses = list(responses)
        self.calls = []

    async def _request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs.get("json")))
        self.redis_at_call = list(self._redis.order)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_the_lease_exists_before_the_pod_does():
    """The window this closes is the one nothing else could."""
    redis = FakeRedis()
    renter = FakeRenter(redis, [{"id": "pod-1"}])

    await renter.rent(image="image", name="call-1")

    # The set happened before the create call was made, not after it returned.
    assert renter.redis_at_call == [("set", "lease:pod:call-1")]
    assert redis.expiries["lease:pod:call-1"] == LEASE_TTL_S


@pytest.mark.asyncio
async def test_the_pod_is_told_which_lease_to_read():
    redis = FakeRedis()
    renter = FakeRenter(redis, [{"id": "pod-1"}])

    pod = await renter.rent(image="image", name="call-1")

    _, _, body = renter.calls[0]
    assert body["env"]["LEASE_KEY"] == "lease:pod:call-1"
    assert pod.lease == "lease:pod:call-1"


@pytest.mark.asyncio
async def test_a_failed_creation_drops_the_lease():
    """The pod may exist anyway. Dropping the lease stops it either way."""
    redis = FakeRedis()
    renter = FakeRenter(redis, [PodError("500")])

    with pytest.raises(PodError):
        await renter.rent(image="image", name="call-1")

    assert "lease:pod:call-1" not in redis.store


@pytest.mark.asyncio
async def test_a_creation_with_no_id_drops_the_lease():
    redis = FakeRedis()
    renter = FakeRenter(redis, [{"nothing": "useful"}])

    with pytest.raises(PodError, match="no id"):
        await renter.rent(image="image", name="call-1")

    assert "lease:pod:call-1" not in redis.store


@pytest.mark.asyncio
async def test_no_persistent_volume_is_attached():
    """A volume keeps charging after the pod stops."""
    redis = FakeRedis()
    renter = FakeRenter(redis, [{"id": "pod-1"}])

    await renter.rent(image="image", name="call-1")

    assert renter.calls[0][2]["volumeInGb"] == 0


@pytest.mark.asyncio
async def test_terminate_drops_the_lease_before_calling_the_api():
    """If the API call fails, the pod still stops at its next check."""
    redis = FakeRedis()
    renter = FakeRenter(redis, [{}])
    pod = RentedPod(id="pod-1", url="u", cost_per_hour=0.2, lease="lease:pod:call-1")
    redis.store[pod.lease] = "held"

    await renter.terminate(pod)

    assert renter.redis_at_call == [("delete", "lease:pod:call-1")]


@pytest.mark.asyncio
async def test_terminate_survives_an_api_failure():
    redis = FakeRedis()
    renter = FakeRenter(redis, [PodError("gone")])
    pod = RentedPod(id="pod-1", url="u", cost_per_hour=0.2, lease="lease:pod:call-1")
    redis.store[pod.lease] = "held"

    await renter.terminate(pod)  # must not raise

    assert pod.lease not in redis.store
