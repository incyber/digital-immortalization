"""Hosted training on Replicate.

Chosen as the first hosted backend because it is pay-per-run: an early product
with a handful of avatars a week cannot justify a reserved GPU, and a reserved
GPU idle 95% of the time is the fastest way to make the unit economics fail.

The trainer is addressed by version rather than by name so that an upstream
change cannot silently alter what a customer's likeness was trained with.
Images are handed over as presigned URLs, which expire, rather than being
uploaded a second time.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx
from loguru import logger

from avatar.storage.base import BlobStore
from avatar.training.base import JobState, TrainingRequest, TrainingResult

API = "https://api.replicate.com/v1"

# Long enough for a provider fetching a 25-image set to finish.
IMAGE_URL_TTL = timedelta(hours=2)

_STATES = {
    "starting": JobState.QUEUED,
    "processing": JobState.RUNNING,
    "succeeded": JobState.SUCCEEDED,
    "failed": JobState.FAILED,
    "canceled": JobState.CANCELLED,
}


class ReplicateTrainingRunner:
    def __init__(self, api_token: str, model_version: str, store: BlobStore, timeout_s: float = 60):
        if not api_token:
            raise ValueError("a Replicate API token is required")
        self._token = api_token
        self._version = model_version
        self._store = store
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return "replicate"

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._token}",
            "content-type": "application/json",
        }

    async def start(self, request: TrainingRequest) -> TrainingResult:
        if not request.image_keys:
            return TrainingResult(state=JobState.FAILED, error="no images to train on")

        urls = await asyncio.gather(
            *(
                self._store.download_url(request.tenant_id, key, ttl=IMAGE_URL_TTL)
                for key in request.image_keys
            )
        )

        payload = {
            "version": self._version,
            "input": {
                "input_images": urls,
                "trigger_word": _trigger_word(request.subject_name),
                "resolution": request.resolution,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(
                    f"{API}/predictions", headers=self._headers(), json=payload
                )
                response.raise_for_status()
                body = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"could not start training run: {exc}")
            return TrainingResult(state=JobState.FAILED, error=str(exc))

        return TrainingResult(
            state=_STATES.get(body.get("status", ""), JobState.QUEUED),
            external_id=body.get("id"),
        )

    async def poll(self, external_id: str) -> TrainingResult:
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.get(
                    f"{API}/predictions/{external_id}", headers=self._headers()
                )
                response.raise_for_status()
                body = response.json()
        except Exception as exc:  # noqa: BLE001
            # A failed poll is not a failed run. Reporting RUNNING keeps a
            # transient network error from marking a customer's forty-minute
            # training job as dead.
            logger.warning(f"could not poll {external_id}, treating as still running: {exc}")
            return TrainingResult(state=JobState.RUNNING, external_id=external_id)

        state = _STATES.get(body.get("status", ""), JobState.RUNNING)
        output = body.get("output")
        return TrainingResult(
            state=state,
            external_id=external_id,
            output_key=output if isinstance(output, str) else None,
            error=body.get("error"),
        )

    async def cancel(self, external_id: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                await client.post(
                    f"{API}/predictions/{external_id}/cancel", headers=self._headers()
                )
        except Exception as exc:  # noqa: BLE001 - cancelling a finished run is fine
            logger.warning(f"cancel of {external_id} did not take: {exc}")


def _trigger_word(subject_name: str) -> str:
    """The token the trained identity responds to.

    Deliberately not the person's name: a real name collides with what the
    base model already knows about anyone famous enough to be in its training
    data, and the likeness drifts towards that instead of the photographs.
    """
    letters = "".join(c for c in subject_name.lower() if c.isalpha())[:8] or "subject"
    return f"tok{letters}"
