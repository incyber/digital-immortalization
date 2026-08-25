"""The blob store boundary.

One protocol, several backends. Local filesystem for development and tests,
S3-compatible for deployment - which covers AWS S3, Cloudflare R2, Backblaze,
MinIO, and anything else speaking the same API, so the hosting decision stays
open rather than being baked into every call site.

Every method takes a tenant id and every key is built by keys.py. A backend
that trusted a caller-supplied key would make the whole tenancy story
decorative, so the signatures do not offer that option.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, runtime_checkable

# Uploads are direct from the browser to the store, using a URL the gateway
# signs. Fifteen minutes is long enough for a slow connection to finish a
# 25-image set and short enough that a leaked URL expires before it is useful.
DEFAULT_UPLOAD_TTL = timedelta(minutes=15)

# Reads are signed too. Nothing in this store is ever public: it holds
# photographs of dead people.
DEFAULT_DOWNLOAD_TTL = timedelta(minutes=10)


@dataclass(frozen=True)
class StoredObject:
    key: str
    size: int
    content_type: str


class StorageError(RuntimeError):
    pass


@runtime_checkable
class BlobStore(Protocol):
    """Per-tenant object storage."""

    async def put(
        self, tenant_id: str, key: str, data: bytes, content_type: str
    ) -> StoredObject:
        """Write an object. The key must already lie inside the tenant's prefix."""

    async def get(self, tenant_id: str, key: str) -> bytes:
        """Read an object, refusing any key outside the tenant's prefix."""

    async def list(self, tenant_id: str, prefix: str) -> list[StoredObject]:
        """List objects under a prefix, which is itself confined to the tenant."""

    async def delete(self, tenant_id: str, key: str) -> None:
        """Remove one object. Used when a family withdraws material."""

    async def delete_tenant(self, tenant_id: str) -> int:
        """Remove everything belonging to a tenant. Returns the count.

        This exists because deletion is a legal obligation, not a convenience:
        consent can be revoked, and revocation that leaves the photographs on
        disk is not revocation.
        """

    async def upload_url(
        self, tenant_id: str, key: str, content_type: str, ttl: timedelta = DEFAULT_UPLOAD_TTL
    ) -> str:
        """A short-lived URL the browser can PUT one object to.

        Direct-to-store upload keeps twenty-five full-resolution photographs
        from passing through the gateway, which would otherwise need to buffer
        them and would become the bottleneck at any real concurrency.
        """

    async def download_url(
        self, tenant_id: str, key: str, ttl: timedelta = DEFAULT_DOWNLOAD_TTL
    ) -> str:
        """A short-lived URL for reading one object."""
