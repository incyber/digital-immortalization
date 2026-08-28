"""The lease must fail towards a dead pod, never towards a live one.

Every test here is about a failure. That is the point: the design is only worth
anything if the broken paths end with the pod stopping, and the broken paths
are the ones that actually happen.
"""

import asyncio

import pytest

from avatar.gpu.lease import LEASE_TTL_S, Lease, lease_key


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.store: dict[str, str] = {}
        self.expiries: dict[str, int] = {}
        self.fail = fail
        self.sets = 0

    async def set(self, key, value, ex=None):
        if self.fail:
            raise ConnectionError("store unreachable")
        self.sets += 1
        self.store[key] = value
        self.expiries[key] = ex

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.mark.asyncio
async def test_the_lease_is_written_with_an_expiry():
    """A key with no TTL is a permanent permission, which is the old bug."""
    redis = FakeRedis()
    await Lease(redis, "pod-1").renew()

    assert redis.store[lease_key("pod-1")] == "held"
    assert redis.expiries[lease_key("pod-1")] == LEASE_TTL_S


@pytest.mark.asyncio
async def test_an_unreachable_store_fails_before_the_pod_is_used():
    """Better to never start than to start something nothing can stop."""
    with pytest.raises(ConnectionError):
        async with Lease(FakeRedis(fail=True), "pod-1").held():
            pass


@pytest.mark.asyncio
async def test_the_lease_is_revoked_when_the_block_ends():
    redis = FakeRedis()
    async with Lease(redis, "pod-1").held():
        assert lease_key("pod-1") in redis.store

    assert lease_key("pod-1") not in redis.store


@pytest.mark.asyncio
async def test_an_exception_inside_the_block_still_revokes():
    redis = FakeRedis()

    with pytest.raises(RuntimeError):
        async with Lease(redis, "pod-1").held():
            raise RuntimeError("the call fell over")

    assert lease_key("pod-1") not in redis.store


@pytest.mark.asyncio
async def test_a_renewal_failure_does_not_raise():
    """A failed renewal is already correct behaviour: the lease ages out."""
    redis = FakeRedis()
    lease = Lease(redis, "pod-1")
    async with lease.held():
        redis.fail = True
        await asyncio.sleep(0)  # let the loop run once

    # No exception escaped, and the pod's fate is now expiry rather than us.
    assert lease_key("pod-1") not in redis.store
