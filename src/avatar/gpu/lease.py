"""The orchestrator half of the pod lease.

Renewal is written from here, outside the pod, and expires on its own if this
stops. That is the inversion: a crashed orchestrator, a lost network or a
closed laptop all end with the pod dead rather than alive, because none of them
can renew.

Nothing in here has to succeed for the guarantee to hold. Its only power is to
*extend* the pod's life; every failure shortens it.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from loguru import logger

# The store enforces this, not us. Long enough that a slow round trip is not a
# denial, short enough that an abandoned pod is gone in under a minute.
LEASE_TTL_S = 60

# Renewed at a third of the TTL, so two consecutive failures still leave the
# lease valid and only a sustained outage ends the pod.
RENEW_EVERY_S = 20


def lease_key(pod_id: str) -> str:
    return f"lease:pod:{pod_id}"


class Lease:
    """A renewable permission for one pod to keep running."""

    def __init__(self, redis, pod_id: str, ttl_s: int = LEASE_TTL_S):
        self._redis = redis
        self._pod_id = pod_id
        self._ttl = ttl_s
        self._task: asyncio.Task | None = None

    async def renew(self) -> None:
        await self._redis.set(lease_key(self._pod_id), "held", ex=self._ttl)

    async def revoke(self) -> None:
        """Drop the lease so the pod stops at its next check.

        Not the primary way a pod ends - expiry is - but it turns a normal
        shutdown from a 45-second wait into a prompt one.
        """
        try:
            await self._redis.delete(lease_key(self._pod_id))
            logger.info(f"revoked lease for {self._pod_id}")
        except Exception as exc:  # noqa: BLE001 - expiry covers this
            logger.warning(f"could not revoke lease for {self._pod_id}: {exc}")

    async def _loop(self) -> None:
        while True:
            try:
                await self.renew()
            except Exception as exc:  # noqa: BLE001
                # Logged, not raised. A failed renewal is already the correct
                # behaviour: the lease ages towards expiry and the pod stops.
                logger.warning(f"lease renewal failed for {self._pod_id}: {exc}")
            await asyncio.sleep(RENEW_EVERY_S)

    @asynccontextmanager
    async def held(self):
        """Hold the lease for the duration of the block.

        The first renewal is awaited before yielding, so a caller that cannot
        reach the store fails before the pod is used rather than after.
        """
        await self.renew()
        self._task = asyncio.create_task(self._loop())
        try:
            yield self
        finally:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            await self.revoke()
