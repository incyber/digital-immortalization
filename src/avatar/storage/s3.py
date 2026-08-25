"""S3-compatible BlobStore.

One backend for AWS S3, Cloudflare R2, Backblaze B2 and MinIO - they speak the
same API, so the hosting decision is an endpoint URL rather than a rewrite.

boto3 is synchronous, so every call runs in a worker thread. The alternative
is another async dependency for the same API surface.

Presigned URLs are what make this scale: twenty-five full-resolution
photographs go browser-to-bucket without passing through the gateway, which
would otherwise buffer them and become the bottleneck.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from functools import cached_property

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from avatar.storage.base import (
    DEFAULT_DOWNLOAD_TTL,
    DEFAULT_UPLOAD_TTL,
    StorageError,
    StoredObject,
)
from avatar.storage.keys import belongs_to, tenant_prefix


class S3BlobStore:
    def __init__(
        self,
        bucket: str,
        region: str = "auto",
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ):
        self._bucket = bucket
        self._region = region
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key

    @cached_property
    def _client(self):
        return boto3.client(
            "s3",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            # SigV4 is required by R2 and by newer S3 regions.
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )

    def _check(self, tenant_id: str, key: str) -> str:
        if not belongs_to(key, tenant_id):
            raise StorageError("key does not belong to this tenant")
        return key

    async def put(
        self, tenant_id: str, key: str, data: bytes, content_type: str
    ) -> StoredObject:
        self._check(tenant_id, key)
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            # Server-side encryption at rest. This bucket holds photographs of
            # dead people; it is not the place to accept a default.
            ServerSideEncryption="AES256",
        )
        return StoredObject(key=key, size=len(data), content_type=content_type)

    async def get(self, tenant_id: str, key: str) -> bytes:
        self._check(tenant_id, key)
        try:
            response = await asyncio.to_thread(
                self._client.get_object, Bucket=self._bucket, Key=key
            )
            return await asyncio.to_thread(response["Body"].read)
        except ClientError as exc:
            raise StorageError(f"no object at {key}") from exc

    async def list(self, tenant_id: str, prefix: str) -> list[StoredObject]:
        if not belongs_to(prefix, tenant_id):
            raise StorageError("prefix does not belong to this tenant")

        def fetch() -> list[StoredObject]:
            objects: list[StoredObject] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    objects.append(
                        StoredObject(
                            key=item["Key"],
                            size=item["Size"],
                            content_type="application/octet-stream",
                        )
                    )
            return objects

        return await asyncio.to_thread(fetch)

    async def delete(self, tenant_id: str, key: str) -> None:
        self._check(tenant_id, key)
        await asyncio.to_thread(self._client.delete_object, Bucket=self._bucket, Key=key)

    async def delete_tenant(self, tenant_id: str) -> int:
        prefix = tenant_prefix(tenant_id)

        def purge() -> int:
            removed = 0
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                batch = [{"Key": item["Key"]} for item in page.get("Contents", [])]
                if not batch:
                    continue
                # delete_objects takes at most 1000 keys per call.
                for chunk in (batch[i : i + 1000] for i in range(0, len(batch), 1000)):
                    self._client.delete_objects(
                        Bucket=self._bucket, Delete={"Objects": chunk}
                    )
                    removed += len(chunk)
            return removed

        return await asyncio.to_thread(purge)

    async def upload_url(
        self, tenant_id: str, key: str, content_type: str, ttl: timedelta = DEFAULT_UPLOAD_TTL
    ) -> str:
        self._check(tenant_id, key)
        return await asyncio.to_thread(
            self._client.generate_presigned_url,
            "put_object",
            Params={
                "Bucket": self._bucket,
                "Key": key,
                "ContentType": content_type,
                "ServerSideEncryption": "AES256",
            },
            ExpiresIn=int(ttl.total_seconds()),
        )

    async def download_url(
        self, tenant_id: str, key: str, ttl: timedelta = DEFAULT_DOWNLOAD_TTL
    ) -> str:
        self._check(tenant_id, key)
        return await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=int(ttl.total_seconds()),
        )
