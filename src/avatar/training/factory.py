"""Runner selection."""

from __future__ import annotations

from avatar.config import Settings
from avatar.storage.base import BlobStore
from avatar.training.base import TrainingRunner


def build_runner(cfg: Settings, store: BlobStore) -> TrainingRunner:
    if cfg.training_backend == "local":
        from avatar.training.local import LocalTrainingRunner

        return LocalTrainingRunner(store)

    if cfg.training_backend == "replicate":
        from avatar.training.replicate import ReplicateTrainingRunner

        return ReplicateTrainingRunner(
            api_token=cfg.replicate_api_token,
            model_version=cfg.replicate_trainer_version,
            store=store,
        )

    raise ValueError(
        f"unknown training_backend {cfg.training_backend!r}; expected 'local' or 'replicate'"
    )
