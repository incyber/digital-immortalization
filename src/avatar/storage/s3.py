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
import io
import threading
import time
from datetime import timedelta

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, SSLError
from loguru import logger

from avatar.storage.base import (
    DEFAULT_DOWNLOAD_TTL,
    DEFAULT_UPLOAD_TTL,
    StorageError,
    StoredObject,
)
from avatar.storage.keys import belongs_to, tenant_prefix

# Above this, an object is uploaded in parts, and each part is this size.
#
# Four megabytes, from measurement rather than convention. Uploading a
# customer's video to R2 failed reliably with "SSLV3_ALERT_BAD_RECORD_MAC" -
# reproduced with plain boto3 and no application code involved, so it is the
# transfer and not anything here. Bisecting by size: 1MB and 4MB succeed, 8MB
# and 16MB fail, consistently. The default part size is 8MB, which is why the
# first attempt at chunking failed on part three rather than fixing anything.
#
# Whatever sits between this machine and Cloudflare, a smaller part is
# universally safe and costs only a few more round trips. Chunking is also
# simply correct for an object this size: a failed part is retried on its own
# rather than losing the whole upload, and somebody's only recording of a
# person who has died should not have to be sent twice.
MULTIPART_THRESHOLD = 4 * 1024 * 1024
MULTIPART_CHUNK = 4 * 1024 * 1024

# How many times a large upload is restarted before the customer is told.
UPLOAD_ATTEMPTS = 4


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
        # See _client: a boto3 client belongs to exactly one thread.
        self._local = threading.local()

    @property
    def _client(self):
        """One client per thread, never one shared between them.

        boto3 clients are not thread-safe: the underlying connection pool and
        its TLS state are not guarded, and two threads writing through one
        client interleave records on the same socket. It surfaces as
        "SSLV3_ALERT_BAD_RECORD_MAC" partway through a transfer, looks random,
        and hits large uploads hardest because they are in flight longest.

        This was latent until the ingest work moved uploads onto threads to
        keep them off the event loop - the correct fix for a different bug -
        and a real 72MB video then failed every time.

        Clients are cheap to construct and expensive to share. One per thread,
        kept for that thread's lifetime.
        """
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._build_client()
            self._local.client = client
        return client

    def _build_client(self):
        return boto3.client(
            "s3",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            # SigV4 is required by R2 and by newer S3 regions.
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )

    def _put_multipart(self, key: str, data: bytes, content_type: str) -> StoredObject:
        """Upload a large object in parts rather than as one stream.

        A single PUT of a customer's video failed reliably against R2 with
        "SSLV3_ALERT_BAD_RECORD_MAC" - reproduced with plain boto3 and no
        application code in the way, so it is the transfer itself and not
        anything here. Small objects were unaffected; the failure scales with
        how long one TLS stream stays open.

        Chunking is the correct way to move an object this size regardless: a
        failed part is retried on its own instead of losing sixty megabytes,
        and a customer's only recording of someone who has died is not a thing
        to make them upload twice.
        """
        transfer = TransferConfig(
            multipart_threshold=MULTIPART_THRESHOLD,
            multipart_chunksize=MULTIPART_CHUNK,
            # Parts go one at a time. The concern here is a fragile connection,
            # not throughput, and several parallel streams is the condition the
            # single large stream already failed under.
            max_concurrency=1,
            use_threads=False,
        )
        # Retried here rather than left to botocore. Its retry policy covers
        # a failed HTTP response; a TLS stream that breaks mid-part raises
        # SSLError out of the transfer manager instead, and the upload simply
        # stops. Observed repeatedly against R2 on a marginal connection, at
        # a different part each time.
        #
        # The whole upload is restarted rather than the part resumed, because
        # the transfer manager owns the part bookkeeping and reaching into it
        # would couple this to its internals. A restart is cheap next to
        # telling somebody to upload the only recording of their father again.
        last: Exception | None = None
        for attempt in range(UPLOAD_ATTEMPTS):
            try:
                self._client.upload_fileobj(
                    io.BytesIO(data),
                    self._bucket,
                    key,
                    ExtraArgs={"ContentType": content_type},
                    Config=transfer,
                )
                return StoredObject(key=key, size=len(data), content_type=content_type)
            except (SSLError, EndpointConnectionError, ConnectionError) as exc:
                last = exc
                logger.warning(
                    f"upload of {key} failed on attempt {attempt + 1} "
                    f"of {UPLOAD_ATTEMPTS}: {type(exc).__name__}"
                )
                # A fresh client, because the broken one holds the broken
                # connection and the next attempt would reuse it.
                self._local.client = None
                time.sleep(2**attempt)

        raise StorageError(
            f"could not upload {key} after {UPLOAD_ATTEMPTS} attempts: {last}"
        ) from last

    def _check(self, tenant_id: str, key: str) -> str:
        if not belongs_to(key, tenant_id):
            raise StorageError("key does not belong to this tenant")
        return key

    async def put(
        self, tenant_id: str, key: str, data: bytes, content_type: str
    ) -> StoredObject:
        self._check(tenant_id, key)
        if len(data) >= MULTIPART_THRESHOLD:
            return await asyncio.to_thread(
                self._put_multipart, key, data, content_type
            )
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
