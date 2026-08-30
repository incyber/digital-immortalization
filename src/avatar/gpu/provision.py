"""Create the serverless endpoint, and refuse to keep an unsafe one.

The safety of this whole design rests on four values held by RunPod rather
than by us:

    workersMin = 0          nothing stays allocated when idle
    workersMax = 1          at most one worker can ever bill at once
    idleTimeout = 5s        a worker stops 5 seconds after finishing
    executionTimeoutMs      "when exceeded, the job fails and the worker stops"

Sending those at creation is not the same as having them. The previous design's
wall-clock ceiling was also configured, was also described to the customer as
protection, and never ran once. So this reads the endpoint back from the
platform after creating it and checks what the platform actually stored. If any
of the four is wrong, the endpoint is deleted before this function returns and
its id is never written anywhere.

That inverts the failure. A bug here means no endpoint, which costs nothing. It
cannot mean an endpoint that quietly bills.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from loguru import logger

from avatar.gpu.serverless import (
    DEFAULT_EXECUTION_TIMEOUT_S,
    DEFAULT_IDLE_TIMEOUT_S,
    DEFAULT_MAX_WORKERS,
    ServerlessError,
    assert_endpoint_is_safe,
)

REST = "https://rest.runpod.io/v1"

# L4 is the cheapest card that fits both models with room to spare, at $0.44/hr
# measured from RunPod's own listing rather than from a summary of it. A5000 is
# named as an alternative so a region with no L4 free does not mean no job.
DEFAULT_GPU_TYPES = ("NVIDIA L4", "NVIDIA RTX A5000")

# Weights are baked into the image, so the container needs room for the image
# rather than for downloads. No volume is attached: a network volume is a
# separate ongoing charge and nothing here needs to persist between jobs.
#
# 90GB, from measurement rather than estimate. The image is 11.5GB compressed
# across 28 layers and roughly 28GB unpacked, and a pull holds both at once -
# about 40GB at peak. The first endpoint was given exactly 40GB and its worker
# sat in `initializing` for eighteen minutes without ever becoming ready.
DEFAULT_CONTAINER_DISK_GB = 90


@dataclass(frozen=True)
class Provisioned:
    endpoint_id: str
    template_id: str
    verified: dict


class Provisioner:
    def __init__(self, api_key: str, timeout_s: float = 60.0):
        if not api_key:
            raise ServerlessError("a RunPod API key is required")
        self._key = api_key
        self._timeout_s = timeout_s

    def _request(self, method: str, path: str, **kwargs) -> dict:
        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.request(
                method,
                f"{REST}{path}",
                headers={"Authorization": f"Bearer {self._key}"},
                **kwargs,
            )
            if response.status_code >= 400:
                raise ServerlessError(
                    f"runpod {method} {path} -> {response.status_code}: {response.text[:300]}"
                )
            return response.json() if response.content else {}

    # ------------------------------------------------------------------
    def create_registry_auth(self, name: str, username: str, password: str) -> str:
        """Store a read-only credential so RunPod can pull a private image.

        The token behind this has read:packages and nothing else, which is why
        the image can stay private. A public image would need no credential at
        all, and would also publish the customer-facing half of the product to
        anyone who guessed the name.
        """
        # Reused by name if it already exists. Names are unique per account and
        # the stored password is never readable back, so a second create with
        # the same name returns a 500 rather than replacing it - which turned a
        # routine re-provision into a failure.
        for existing in self._request("GET", "/containerregistryauth"):
            if existing.get("name") == name:
                logger.info(f"reusing registry auth {existing['id']}")
                return existing["id"]

        body = {"name": name, "username": username, "password": password}
        auth_id = self._request("POST", "/containerregistryauth", json=body).get("id")
        if not auth_id:
            raise ServerlessError("registry auth creation returned no id")
        logger.info(f"registry auth {auth_id} for {username}")
        return auth_id

    def create_template(
        self,
        name: str,
        image: str,
        *,
        registry_auth_id: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        # Template names are unique per account, and re-provisioning after a
        # failure is normal rather than exceptional. An identical template is
        # reused instead of erroring; one that differs is a real conflict and
        # is reported as such, because silently running on someone else's disk
        # size is how the last failure happened.
        for existing in self._request("GET", "/templates"):
            if existing.get("name") != name:
                continue
            if (
                existing.get("imageName") == image
                and int(existing.get("containerDiskInGb") or 0) >= DEFAULT_CONTAINER_DISK_GB
            ):
                logger.info(f"reusing template {existing['id']}")
                return existing["id"]
            raise ServerlessError(
                f"a template named {name!r} already exists with different settings "
                f"({existing.get('imageName')}, {existing.get('containerDiskInGb')}GB); "
                "delete it or choose another name"
            )

        body: dict = {
            "name": name,
            "imageName": image,
            "isServerless": True,
            "containerDiskInGb": DEFAULT_CONTAINER_DISK_GB,
            # No exposed ports. A serverless worker pulls its work from the
            # queue; anything listening would be a Pod habit carried over.
            "ports": [],
            "env": env or {},
        }
        if registry_auth_id:
            body["containerRegistryAuthId"] = registry_auth_id

        template_id = self._request("POST", "/templates", json=body).get("id")
        if not template_id:
            raise ServerlessError("template creation returned no id")
        logger.info(f"template {template_id} -> {image}")
        return template_id

    def create_endpoint(
        self,
        name: str,
        template_id: str,
        *,
        gpu_types: tuple[str, ...] = DEFAULT_GPU_TYPES,
        execution_timeout_s: int = DEFAULT_EXECUTION_TIMEOUT_S,
    ) -> str:
        body = {
            "name": name,
            "templateId": template_id,
            "computeType": "GPU",
            "gpuTypeIds": list(gpu_types),
            "gpuCount": 1,
            "workersMin": 0,
            "workersMax": DEFAULT_MAX_WORKERS,
            "idleTimeout": DEFAULT_IDLE_TIMEOUT_S,
            "executionTimeoutMs": execution_timeout_s * 1000,
            # Flashboot keeps a snapshot warm to cut cold starts. It is not
            # enabled: it is an optimisation for endpoints under steady load,
            # and this one is idle almost all of the time.
            "flashboot": False,
        }
        endpoint_id = self._request("POST", "/endpoints", json=body).get("id")
        if not endpoint_id:
            raise ServerlessError("endpoint creation returned no id")
        return endpoint_id

    def retarget_template(self, template_id: str, image: str, env: dict[str, str]) -> None:
        """Point an existing template at a new image, resending its whole env.

        The platform does not return `env` when a template is read, so an
        update that omits it silently empties it - the worker then starts,
        finds no bucket to fetch from, and does nothing anybody can see. So
        the caller passes the complete environment every time and this refuses
        an obviously incomplete one rather than shipping a broken worker.
        """
        required = ("BUNDLE_BUCKET", "BUNDLE_KEY", "BUNDLE_SHA256")
        missing = [name for name in required if not env.get(name)]
        if missing:
            raise ServerlessError(
                "refusing to update the template without " + ", ".join(missing)
            )

        self._request(
            "PATCH",
            f"/templates/{template_id}",
            json={"imageName": image, "env": env},
        )
        logger.info(f"template {template_id} -> {image}")

    def read_endpoint(self, endpoint_id: str) -> dict:
        return self._request("GET", f"/endpoints/{endpoint_id}")

    def delete_endpoint(self, endpoint_id: str) -> None:
        self._request("DELETE", f"/endpoints/{endpoint_id}")
        logger.info(f"deleted endpoint {endpoint_id}")

    # ------------------------------------------------------------------
    def provision(
        self,
        name: str,
        image: str,
        *,
        registry_auth_id: str | None = None,
        env: dict[str, str] | None = None,
        execution_timeout_s: int = DEFAULT_EXECUTION_TIMEOUT_S,
    ) -> Provisioned:
        """Create the endpoint, verify it, and delete it if it is not safe."""
        template_id = self.create_template(
            f"{name}-template", image, registry_auth_id=registry_auth_id, env=env
        )
        endpoint_id = self.create_endpoint(
            name, template_id, execution_timeout_s=execution_timeout_s
        )

        try:
            stored = self.read_endpoint(endpoint_id)
        except ServerlessError:
            # Unverifiable is treated the same as unsafe. An endpoint whose
            # settings cannot be read is one whose settings are not known.
            self.delete_endpoint(endpoint_id)
            raise

        problems = assert_endpoint_is_safe(stored)
        if problems:
            self.delete_endpoint(endpoint_id)
            raise ServerlessError(
                "endpoint was created with unsafe settings and has been deleted: "
                + "; ".join(problems)
            )

        logger.info(
            f"endpoint {endpoint_id} verified: workersMin={stored.get('workersMin')} "
            f"workersMax={stored.get('workersMax')} idleTimeout={stored.get('idleTimeout')}s "
            f"executionTimeout={int(stored.get('executionTimeoutMs', 0)) // 1000}s"
        )
        return Provisioned(
            endpoint_id=endpoint_id,
            template_id=template_id,
            verified={
                key: stored.get(key)
                for key in ("workersMin", "workersMax", "idleTimeout", "executionTimeoutMs")
            },
        )
