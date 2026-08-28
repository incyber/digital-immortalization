"""Renting the warm GPU a live call needs, and never renting it unattended.

Every method here is written on the assumption that this process will die at
the worst moment. The pod is created with the lease already written, so a crash
one line after creation still ends with the pod stopping on its own; the lease
is written *before* the pod exists, so there is no window where a pod is
running and nothing is set to expire.

That ordering is the whole design. The previous version created the pod first
and armed the guard afterwards, which left a gap in which nothing was enforcing
anything - the one window the review could not close.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from loguru import logger

from avatar.gpu.lease import LEASE_TTL_S, lease_key
from avatar.gpu.serverless import ServerlessError

REST = "https://rest.runpod.io/v1"

# 24GB is enough for MuseTalk's UNet, the VAE and whisper-tiny with room for a
# batch. Listed cheapest first; the platform takes the first one it can place.
DEFAULT_GPU_TYPES = (
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4500",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 4090",
)

# The image carries its own weights, so this is image size plus working room.
DEFAULT_CONTAINER_DISK_GB = 90

READY_TIMEOUT_S = 900
POLL_S = 10


@dataclass(frozen=True)
class RentedPod:
    id: str
    url: str
    cost_per_hour: float
    # Carried rather than derived from the id. The key is written before the
    # pod exists and so cannot be named after it, and the pod reads this exact
    # string from its environment - deriving it a second time somewhere else is
    # how the two halves drift apart.
    lease: str


class PodError(RuntimeError):
    pass


class PodRenter:
    def __init__(self, api_key: str, redis, timeout_s: float = 60.0):
        if not api_key:
            raise ServerlessError("a RunPod API key is required")
        self._key = api_key
        self._redis = redis
        self._timeout_s = timeout_s

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.request(
                method,
                f"{REST}{path}",
                headers={"Authorization": f"Bearer {self._key}"},
                **kwargs,
            )
            if response.status_code >= 400:
                raise PodError(
                    f"runpod {method} {path} -> {response.status_code}: {response.text[:300]}"
                )
            return response.json() if response.content else {}

    # ------------------------------------------------------------------
    async def rent(
        self,
        *,
        image: str,
        name: str,
        registry_auth_id: str | None = None,
        env: dict[str, str] | None = None,
        gpu_types: tuple[str, ...] = DEFAULT_GPU_TYPES,
    ) -> RentedPod:
        """Create a pod that is already under a lease when it starts.

        The lease is written first, against an id the pod will be told to use.
        A pod that comes up finds a live lease; a pod that never comes up
        leaves a key that expires by itself.
        """
        # Named by us rather than by the platform, so the key exists before the
        # pod does. The pod reads this exact name from its environment.
        key = lease_key(name)
        await self._redis.set(key, "held", ex=LEASE_TTL_S)
        logger.info(f"lease {key} written before any pod exists")

        body = {
            "name": name,
            "imageName": image,
            "computeType": "GPU",
            "gpuTypeIds": list(gpu_types),
            "gpuCount": 1,
            "containerDiskInGb": DEFAULT_CONTAINER_DISK_GB,
            # No persistent volume. Nothing on this pod needs to outlive it,
            # and a volume is a charge that continues after the pod stops.
            "volumeInGb": 0,
            "ports": ["7100/http"],
            "env": {"LEASE_KEY": key, **(env or {})},
        }
        if registry_auth_id:
            body["containerRegistryAuthId"] = registry_auth_id

        try:
            created = await self._request("POST", "/pods", json=body)
        except Exception:
            # The pod may or may not exist. Dropping the lease means that if it
            # does, it stops at its first check rather than running unwatched.
            await self._redis.delete(key)
            raise

        pod_id = created.get("id")
        if not pod_id:
            await self._redis.delete(key)
            raise PodError(f"pod creation returned no id: {created}")

        logger.info(f"pod {pod_id} created under lease {key}")
        return RentedPod(
            id=pod_id,
            url=f"https://{pod_id}-7100.proxy.runpod.net",
            cost_per_hour=float(created.get("costPerHr") or 0.0),
            lease=key,
        )

    async def wait_until_ready(self, pod: RentedPod, timeout_s: int = READY_TIMEOUT_S) -> None:
        """Block until the service answers, or give the pod back.

        A pod that never becomes ready is terminated here rather than left to
        the lease. The lease would get there eventually; this gets there in
        seconds, and the difference is money.
        """
        deadline = asyncio.get_running_loop().time() + timeout_s
        async with httpx.AsyncClient(timeout=10.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    response = await client.get(f"{pod.url}/health")
                    if response.status_code == 200 and response.json().get("ok"):
                        logger.info(f"pod {pod.id} ready")
                        return
                except Exception:  # noqa: BLE001, S110 - not ready yet is normal
                    pass
                await asyncio.sleep(POLL_S)

        await self.terminate(pod)
        raise PodError(f"pod {pod.id} did not become ready within {timeout_s}s")

    async def terminate(self, pod: RentedPod) -> None:
        """Stop paying for a pod, twice over.

        The lease is dropped first. If the API call fails, times out, or this
        process dies immediately after, the pod still stops within its next few
        checks because nothing is renewing it. Termination is the fast path,
        not the load-bearing one.
        """
        try:
            await self._redis.delete(pod.lease)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"could not drop lease for {pod.id}: {exc}")

        try:
            await self._request("DELETE", f"/pods/{pod.id}")
            logger.info(f"terminated pod {pod.id}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"terminate of {pod.id} failed; the lease will expire: {exc}")

    async def list_pods(self) -> list[dict]:
        """What is actually running, asked of the platform.

        Used by the audit tool. It believes nothing this application recorded,
        because a leaked pod is precisely one the application has forgotten.
        """
        data = await self._request("GET", "/pods")
        return data if isinstance(data, list) else data.get("pods", [])
