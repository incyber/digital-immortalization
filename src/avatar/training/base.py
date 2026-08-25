"""The training boundary.

Turning a photo set into a likeness is a long, expensive, failure-prone job
that outlives the request that started it. So it is modelled as a job with a
row, not a function call: a customer who closes the tab can be told what
happened, and a runner that crashes leaves evidence rather than an avatar
stuck forever in "training".

Two backends behind one protocol, for the same reason the renderer has two:
the local one needs no account and no GPU so the whole flow can be built and
tested, and the hosted one is what actually trains.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


@dataclass(frozen=True)
class TrainingRequest:
    """Everything a run needs, resolved before it starts.

    image_keys are storage keys inside one tenant's prefix. Passing keys rather
    than bytes keeps a 25-image set out of process memory and lets a hosted
    provider fetch them from presigned URLs.
    """

    tenant_id: str
    photo_set_id: str
    image_keys: list[str]
    subject_name: str
    # Sizes the pipeline downstream expects. See the renderer's asset builder.
    resolution: int = 1024


@dataclass(frozen=True)
class TrainingResult:
    state: JobState
    external_id: str | None = None
    output_key: str | None = None
    error: str | None = None
    progress: float = 0.0


@runtime_checkable
class TrainingRunner(Protocol):
    """Starts and observes identity-training runs."""

    @property
    def name(self) -> str:
        """Recorded on the job row, so a stuck job can be traced to a provider."""

    async def start(self, request: TrainingRequest) -> TrainingResult:
        """Begin a run. Returns immediately with an external id.

        Never blocks until completion: a training run takes tens of minutes and
        the caller is an HTTP request.
        """

    async def poll(self, external_id: str) -> TrainingResult:
        """Current state of a run."""

    async def cancel(self, external_id: str) -> None:
        """Abandon a run. Must be safe to call on an already-finished job."""
