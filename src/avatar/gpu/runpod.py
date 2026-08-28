"""Renting a GPU, with teardown that cannot be forgotten.

RunPod bills per second, which is cheap, but it does not stop idle pods. A pod
left running overnight bills the full hourly rate for doing nothing, and the
usual way that happens is a process crashing between "create" and "terminate".

So nothing here creates a pod outside a context manager, the teardown runs in
a finally block, and a wall-clock ceiling terminates the pod even if the work
never returns. There is also a sweep that kills anything this project has ever
started, for the case where the machine itself went away mid-run.

The outer protection is not code: RunPod is prepaid, so the account balance is
a hard ceiling. Keep auto-pay off and the worst case is the balance, not the
bill.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import httpx
from loguru import logger

API = "https://rest.runpod.io/v1"

# Everything this project starts is tagged, so the sweep can find pods even if
# the process that made them is gone.
POD_TAG = "avatar-worker"

# Nothing here should ever run this long. A job that does has hung, and the
# ceiling is what stops a hang becoming a monthly bill.
DEFAULT_MAX_MINUTES = 30

# Checked before creating anything. Below this a run cannot finish, and
# failing early is cheaper than failing halfway.
MIN_BALANCE_USD = 1.0


class GpuError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pod:
    id: str
    name: str
    gpu: str
    cost_per_hour: float
    started_at: float

    @property
    def minutes_running(self) -> float:
        return (time.monotonic() - self.started_at) / 60.0

    @property
    def cost_so_far(self) -> float:
        return self.cost_per_hour * (self.minutes_running / 60.0)


class RunPodClient:
    def __init__(self, api_key: str, timeout_s: float = 60.0):
        if not api_key:
            raise GpuError("a RunPod API key is required")
        self._key = api_key
        self._timeout_s = timeout_s

    def _request(self, method: str, path: str, **kwargs) -> dict:
        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.request(
                method,
                f"{API}{path}",
                headers={"Authorization": f"Bearer {self._key}"},
                **kwargs,
            )
            if response.status_code >= 400:
                raise GpuError(f"runpod {method} {path} -> {response.status_code}: {response.text[:200]}")
            return response.json() if response.content else {}

    def balance(self) -> float:
        """Remaining prepaid credit, in dollars."""
        try:
            data = self._request("GET", "/billing/balance")
            return float(data.get("currentBalance", data.get("balance", 0.0)))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"could not read balance: {exc}")
            return -1.0

    def list_pods(self) -> list[dict]:
        data = self._request("GET", "/pods")
        return data if isinstance(data, list) else data.get("pods", data.get("data", []))

    def create(self, *, image: str, gpu_type: str, name: str, ports: str, cost_per_hour: float) -> Pod:
        payload = {
            "name": name,
            "imageName": image,
            "gpuTypeIds": [gpu_type],
            "gpuCount": 1,
            "containerDiskInGb": 40,
            "ports": ports.split(","),
            "env": {"AVATAR_TAG": POD_TAG},
        }
        data = self._request("POST", "/pods", json=payload)
        pod_id = data.get("id") or data.get("podId")
        if not pod_id:
            raise GpuError(f"runpod did not return a pod id: {data}")

        logger.info(f"started pod {pod_id} ({gpu_type}) at ${cost_per_hour:.2f}/hr")
        return Pod(
            id=pod_id, name=name, gpu=gpu_type,
            cost_per_hour=cost_per_hour, started_at=time.monotonic(),
        )

    def terminate(self, pod_id: str) -> None:
        """Destroy a pod. Safe to call on one that is already gone."""
        try:
            self._request("DELETE", f"/pods/{pod_id}")
            logger.info(f"terminated pod {pod_id}")
        except GpuError as exc:
            if "404" in str(exc):
                return
            raise

    def sweep(self) -> list[str]:
        """Terminate every pod this project started.

        For the case the controlling process died between create and finally.
        Run it on a schedule and on start-up; a leaked pod costs money for as
        long as nobody looks.
        """
        killed: list[str] = []
        for pod in self.list_pods():
            env = pod.get("env") or {}
            name = pod.get("name", "")
            if env.get("AVATAR_TAG") == POD_TAG or name.startswith(POD_TAG):
                pod_id = pod.get("id")
                if pod_id:
                    self.terminate(pod_id)
                    killed.append(pod_id)
        return killed


@contextmanager
def rented_gpu(
    client: RunPodClient,
    *,
    image: str,
    gpu_type: str = "NVIDIA L4",
    cost_per_hour: float = 0.39,
    ports: str = "7002/http,7003/http",
    max_minutes: int = DEFAULT_MAX_MINUTES,
    name: str | None = None,
):
    """Rent a GPU for the duration of the block, and always give it back.

    Teardown is in a finally, so it runs on success, on exception, and on
    KeyboardInterrupt. The wall-clock ceiling is checked by the caller through
    `pod.minutes_running`; it exists because a hung job would otherwise hold
    the pod until somebody notices.
    """
    balance = client.balance()
    if 0 <= balance < MIN_BALANCE_USD:
        raise GpuError(
            f"balance is ${balance:.2f}, below the ${MIN_BALANCE_USD:.2f} floor; "
            "top up before starting a run"
        )

    pod = client.create(
        image=image,
        gpu_type=gpu_type,
        name=name or f"{POD_TAG}-{int(time.time())}",
        ports=ports,
        cost_per_hour=cost_per_hour,
    )

    try:
        yield pod
    finally:
        # Runs on success, on exception, and on interrupt. This is the whole
        # point of the module.
        try:
            client.terminate(pod.id)
        finally:
            logger.info(
                f"pod {pod.id} ran {pod.minutes_running:.1f} min, "
                f"about ${pod.cost_so_far:.3f}"
            )
            if pod.minutes_running > max_minutes:
                logger.warning(
                    f"pod exceeded its {max_minutes} minute ceiling - "
                    "the job hung rather than finished"
                )
