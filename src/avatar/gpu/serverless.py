"""GPU work on RunPod Serverless: nothing is allocated between jobs.

This replaces a Pod-based design that could not be made safe. The problem with
Pods is structural: a Pod runs until something acts to stop it, so every
failure - crashed orchestrator, slept laptop, dropped network, killed process -
leaves a GPU billing. Guarding that requires leases, deadman switches, sweeps
and an external watcher, each with its own failure mode, and one window that
cannot be closed at all: between creating a Pod and its supervisor arming,
nothing is enforcing anything.

Serverless removes the entire class rather than defending against it. There is
no pod to forget, because between jobs nothing exists. Three provider-enforced
guarantees do the work, all verified in RunPod's documentation:

    workersMin = 0        nothing stays allocated when idle
    idleTimeout = 30s     a worker shuts down 30 seconds after finishing
    executionTimeout      "when exceeded, the job fails and the worker stops"

None of those depend on this code being correct, running, or even alive. The
safest code is the code that does not need to exist.

Pods remain the right answer for one case this does not cover: a live call
that must hold warm models across a multi-minute conversation. That decision
should be made when that renderer exists, deliberately, with the lease design
the review specified - not inherited by default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import httpx
from loguru import logger

API = "https://api.runpod.ai/v2"

# The platform kills a job that exceeds this and stops the worker. It is the
# ceiling that matters, and it is enforced by RunPod rather than here.
DEFAULT_EXECUTION_TIMEOUT_S = 900

# How long a worker lingers after finishing. Was 5s, on the reasoning that our
# work is bursty and an idle worker costs money. That reasoning was right about
# the money and wrong about the clock: this worker downloads and verifies its
# code before it can poll for anything, which takes about fifteen seconds, and
# the platform was sending SIGTERM at eight. Jobs sat in the queue while a
# healthy worker was started and stopped underneath them, over and over.
#
# Thirty seconds is still far below the sixty-second ceiling the safety check
# enforces, and the idle time it can cost - about a third of a cent per job -
# buys a worker that survives long enough to claim one.
DEFAULT_IDLE_TIMEOUT_S = 30

# A concurrency cap that doubles as a spend cap: this many workers is the most
# that can ever bill at once.
DEFAULT_MAX_WORKERS = 1


class JobState(str, Enum):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"

    @property
    def terminal(self) -> bool:
        return self in (
            JobState.COMPLETED, JobState.FAILED,
            JobState.CANCELLED, JobState.TIMED_OUT,
        )

    @property
    def billing(self) -> bool:
        """Whether a worker is running, and therefore costing money."""
        return self is JobState.IN_PROGRESS


class ServerlessError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobResult:
    id: str
    state: JobState
    output: dict | None = None
    error: str | None = None
    execution_ms: int = 0

    @property
    def cost(self) -> float:
        """What the run cost, from measured execution time.

        Read back from the platform rather than assumed from a constant, so a
        wrong or pricier GPU shows up in the number instead of hiding in it.
        """
        return (self.execution_ms / 3_600_000.0) * 0.44


class ServerlessClient:
    """Submit work to an endpoint. There is no lifecycle to manage."""

    def __init__(self, api_key: str, endpoint_id: str, timeout_s: float = 60.0):
        if not api_key:
            raise ServerlessError("a RunPod API key is required")
        if not endpoint_id:
            raise ServerlessError("a serverless endpoint id is required")
        self._key = api_key
        self._endpoint = endpoint_id
        self._timeout_s = timeout_s

    def _request(self, method: str, path: str, **kwargs) -> dict:
        with httpx.Client(timeout=self._timeout_s) as client:
            response = client.request(
                method,
                f"{API}/{self._endpoint}{path}",
                headers={"Authorization": f"Bearer {self._key}"},
                **kwargs,
            )
            if response.status_code >= 400:
                raise ServerlessError(
                    f"runpod {method} {path} -> {response.status_code}: {response.text[:200]}"
                )
            return response.json() if response.content else {}

    def submit(self, payload: dict) -> str:
        """Queue a job. Returns immediately with its id."""
        data = self._request("POST", "/run", json={"input": payload})
        job_id = data.get("id")
        if not job_id:
            raise ServerlessError(f"no job id in response: {data}")
        logger.info(f"queued job {job_id}")
        return job_id

    def status(self, job_id: str) -> JobResult:
        data = self._request("GET", f"/status/{job_id}")
        raw = data.get("status", "")
        try:
            state = JobState(raw)
        except ValueError:
            # An unrecognised state is treated as still running rather than
            # finished: assuming completion would stop us watching something
            # that is still billing.
            logger.warning(f"unknown job state {raw!r}, treating as in progress")
            state = JobState.IN_PROGRESS

        return JobResult(
            id=job_id,
            state=state,
            output=data.get("output"),
            error=data.get("error"),
            execution_ms=int(data.get("executionTime") or 0),
        )

    def cancel(self, job_id: str) -> None:
        """Stop a job. Safe on one that has already finished."""
        try:
            self._request("POST", f"/cancel/{job_id}")
            logger.info(f"cancelled job {job_id}")
        except ServerlessError as exc:
            logger.warning(f"cancel of {job_id} did not take: {exc}")

    def run(self, payload: dict, *, poll_s: float = 2.0, wait_s: float = 1800) -> JobResult:
        """Submit and wait.

        Cancels on timeout or interrupt, but that is a courtesy rather than a
        safety mechanism: the platform's own execution timeout stops the worker
        whether or not this call survives to cancel anything.
        """
        job_id = self.submit(payload)
        deadline = time.monotonic() + wait_s

        try:
            while time.monotonic() < deadline:
                result = self.status(job_id)
                if result.state.terminal:
                    logger.info(
                        f"job {job_id} {result.state.value} "
                        f"in {result.execution_ms}ms (~${result.cost:.4f})"
                    )
                    return result
                time.sleep(poll_s)

            self.cancel(job_id)
            raise ServerlessError(f"job {job_id} did not finish within {wait_s}s")
        except KeyboardInterrupt:
            self.cancel(job_id)
            raise


def assert_endpoint_is_safe(config: dict) -> list[str]:
    """Check an endpoint cannot bill while idle.

    Read back from the platform rather than trusted from whatever was sent at
    creation. The whole safety argument rests on these three values, so they
    are verified rather than assumed - the previous design's ceiling was also
    'configured' and turned out never to run.
    """
    problems: list[str] = []

    workers_min = config.get("workersMin", config.get("workersStandby", 0))
    if workers_min and int(workers_min) > 0:
        problems.append(
            f"workersMin is {workers_min}: active workers bill continuously, "
            "including when idle. It must be 0."
        )

    idle = config.get("idleTimeout")
    if idle is not None and int(idle) > 60:
        problems.append(f"idleTimeout is {idle}s: a worker bills for that long after each job")

    execution = config.get("executionTimeout", config.get("executionTimeoutMs"))
    if not execution:
        problems.append("no executionTimeout: nothing stops a hung job")

    workers_max = config.get("workersMax")
    if workers_max and int(workers_max) > DEFAULT_MAX_WORKERS:
        problems.append(
            f"workersMax is {workers_max}: that many workers can bill at once"
        )

    return problems
