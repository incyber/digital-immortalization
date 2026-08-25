"""Filesystem BlobStore, for development and tests.

Not for deployment: it has no durability story, no replication, and its signed
URLs are not signed. It exists so the whole upload and training flow can be
built and tested without an account anywhere, in the same way the viseme
renderer lets the call loop be built without a GPU.

The isolation checks are real, though, and identical to the ones the S3
backend will carry. That is the point of the shared contract suite: if the
local backend can be made to cross tenants, so can the real one.
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import timedelta
from pathlib import Path

from avatar.storage.base import (
    DEFAULT_DOWNLOAD_TTL,
    DEFAULT_UPLOAD_TTL,
    StorageError,
    StoredObject,
)
from avatar.storage.keys import belongs_to, tenant_prefix


class LocalBlobStore:
    def __init__(self, root: Path | str):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, tenant_id: str, key: str) -> Path:
        """Map a key to a path, refusing anything outside the tenant's prefix.

        Two independent checks. The prefix check catches a key belonging to
        another tenant; the resolved-path check catches anything that survives
        key validation and still escapes, such as a symlink planted in the
        tree. Either alone would be enough today; both together stay correct
        if one is later weakened.
        """
        if not belongs_to(key, tenant_id):
            raise StorageError("key does not belong to this tenant")

        path = (self._root / key).resolve()
        tenant_root = (self._root / tenant_prefix(tenant_id)).resolve()
        if not str(path).startswith(str(tenant_root)):
            raise StorageError("resolved path escapes the tenant prefix")
        return path

    async def put(
        self, tenant_id: str, key: str, data: bytes, content_type: str
    ) -> StoredObject:
        path = self._resolve(tenant_id, key)

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.with_suffix(path.suffix + ".type").write_text(content_type)

        await asyncio.to_thread(write)
        return StoredObject(key=key, size=len(data), content_type=content_type)

    async def get(self, tenant_id: str, key: str) -> bytes:
        path = self._resolve(tenant_id, key)
        if not path.is_file():
            raise StorageError(f"no object at {key}")
        return await asyncio.to_thread(path.read_bytes)

    async def list(self, tenant_id: str, prefix: str) -> list[StoredObject]:
        if not belongs_to(prefix, tenant_id):
            raise StorageError("prefix does not belong to this tenant")

        base = (self._root / prefix).resolve()
        if not base.is_dir():
            return []

        objects: list[StoredObject] = []
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix == ".type":
                continue
            type_file = path.with_suffix(path.suffix + ".type")
            objects.append(
                StoredObject(
                    key=str(path.relative_to(self._root)),
                    size=path.stat().st_size,
                    content_type=(
                        type_file.read_text() if type_file.is_file() else "application/octet-stream"
                    ),
                )
            )
        return objects

    async def delete(self, tenant_id: str, key: str) -> None:
        path = self._resolve(tenant_id, key)
        if path.is_file():
            path.unlink()
            type_file = path.with_suffix(path.suffix + ".type")
            if type_file.is_file():
                type_file.unlink()

    async def delete_tenant(self, tenant_id: str) -> int:
        base = (self._root / tenant_prefix(tenant_id)).resolve()
        if not base.is_dir():
            return 0
        count = sum(1 for p in base.rglob("*") if p.is_file() and p.suffix != ".type")
        await asyncio.to_thread(shutil.rmtree, base)
        return count

    async def upload_url(
        self, tenant_id: str, key: str, content_type: str, ttl: timedelta = DEFAULT_UPLOAD_TTL
    ) -> str:
        # Not a signed URL. The gateway accepts the bytes itself in local mode;
        # the S3 backend returns a real presigned PUT.
        self._resolve(tenant_id, key)
        return f"/api/uploads/local/{key}"

    async def download_url(
        self, tenant_id: str, key: str, ttl: timedelta = DEFAULT_DOWNLOAD_TTL
    ) -> str:
        self._resolve(tenant_id, key)
        return f"/api/uploads/local/{key}"
