"""Backend selection."""

from __future__ import annotations

from pathlib import Path

from avatar.config import Settings
from avatar.storage.base import BlobStore


def build_store(cfg: Settings) -> BlobStore:
    if cfg.storage_backend == "local":
        from avatar.storage.local import LocalBlobStore

        return LocalBlobStore(Path(cfg.storage_root))

    if cfg.storage_backend == "s3":
        from avatar.storage.s3 import S3BlobStore

        return S3BlobStore(
            bucket=cfg.s3_bucket,
            region=cfg.s3_region,
            endpoint_url=cfg.s3_endpoint_url or None,
            access_key=cfg.s3_access_key or None,
            secret_key=cfg.s3_secret_key or None,
        )

    raise ValueError(f"unknown storage_backend {cfg.storage_backend!r}; expected 'local' or 's3'")
