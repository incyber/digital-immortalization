"""A training runner that does not train.

It moves a job through queued, running and succeeded on a timer, and writes a
placeholder artefact. That is enough to build and test everything around
training - the queue, the status page, the failure paths, the handoff to the
renderer - without a GPU account or a forty-minute wait per iteration.

It is not a stub in the sense of being unfinished: it is the development
backend, in the same way the viseme renderer is, and it implements the full
protocol so the hosted runner is a swap rather than a rewrite.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from avatar.storage.base import BlobStore
from avatar.storage.keys import asset_key
from avatar.training.base import JobState, TrainingRequest, TrainingResult

# Long enough that a status page visibly progresses, short enough not to slow
# the test suite.
DEFAULT_DURATION_S = 6.0


@dataclass
class _Run:
    request: TrainingRequest
    started_at: float
    duration_s: float
    cancelled: bool = False


class LocalTrainingRunner:
    """In-process, in-memory. Development only - a restart loses every run."""

    def __init__(self, store: BlobStore, duration_s: float = DEFAULT_DURATION_S):
        self._store = store
        self._duration_s = duration_s
        self._runs: dict[str, _Run] = {}
        self._counter = 0

    @property
    def name(self) -> str:
        return "local"

    async def start(self, request: TrainingRequest) -> TrainingResult:
        if not request.image_keys:
            return TrainingResult(state=JobState.FAILED, error="no images to train on")

        self._counter += 1
        external_id = f"local-{self._counter:06d}"
        self._runs[external_id] = _Run(
            request=request, started_at=time.monotonic(), duration_s=self._duration_s
        )
        return TrainingResult(state=JobState.RUNNING, external_id=external_id)

    async def poll(self, external_id: str) -> TrainingResult:
        run = self._runs.get(external_id)
        if run is None:
            return TrainingResult(state=JobState.FAILED, error="unknown run")

        if run.cancelled:
            return TrainingResult(state=JobState.CANCELLED, external_id=external_id)

        elapsed = time.monotonic() - run.started_at
        if elapsed < run.duration_s:
            return TrainingResult(
                state=JobState.RUNNING,
                external_id=external_id,
                progress=min(0.99, elapsed / run.duration_s),
            )

        key = asset_key(run.request.tenant_id, run.request.photo_set_id, "identity.safetensors")
        await self._store.put(
            run.request.tenant_id,
            key,
            b"placeholder identity weights produced by the development runner",
            "application/octet-stream",
        )
        return TrainingResult(
            state=JobState.SUCCEEDED, external_id=external_id, output_key=key, progress=1.0
        )

    async def cancel(self, external_id: str) -> None:
        run = self._runs.get(external_id)
        if run is not None:
            run.cancelled = True

    async def wait(self, external_id: str, timeout_s: float = 30.0) -> TrainingResult:
        """Block until terminal. Used by tests, never by a request handler."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = await self.poll(external_id)
            if result.state.terminal:
                return result
            await asyncio.sleep(0.05)
        return TrainingResult(state=JobState.FAILED, error="timed out waiting")
