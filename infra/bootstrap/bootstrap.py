"""Fetch the application from private storage, verify it, run it.

This file is the only thing of ours baked into the published images, and it is
written to be worth nothing to a reader: download an archive, check its hash,
unpack it, exec what is inside. Everything specific to this product - the
handler, the renderer service, the licence-clean face detector, the lease
supervisor - lives in the archive, in a private bucket, and never enters a
container registry.

That is the whole point. The images can be public, because a public image of
open-source models plus this file gives away nothing that was not already
downloadable from the projects it is built on.

The hash is not decoration. Object storage credentials sit in the worker's
environment, so an attacker who reaches them could otherwise replace the code
a GPU runs. BUNDLE_SHA256 is supplied by whoever creates the endpoint and is
compared before anything is unpacked or executed; a mismatch is fatal.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

DEST = Path(os.environ.get("APP_DIR", "/opt/app"))

# Signed GET, or any URL that serves the archive. Presigned is preferred: it
# means the container never holds a long-lived credential at all.
BUNDLE_URL = os.environ.get("BUNDLE_URL", "")

# The S3-compatible route, for when a presigned URL would expire before the
# worker is next started.
BUNDLE_ENDPOINT = os.environ.get("BUNDLE_ENDPOINT_URL", "")
BUNDLE_BUCKET = os.environ.get("BUNDLE_BUCKET", "")
BUNDLE_KEY = os.environ.get("BUNDLE_KEY", "")
BUNDLE_ACCESS_KEY = os.environ.get("BUNDLE_ACCESS_KEY", "")
BUNDLE_SECRET_KEY = os.environ.get("BUNDLE_SECRET_KEY", "")

EXPECTED_SHA256 = os.environ.get("BUNDLE_SHA256", "")

# What to run once the archive is in place, relative to DEST.
ENTRYPOINT = os.environ.get("APP_ENTRYPOINT", "handler.py")

MAX_BYTES = 64 * 1024 * 1024


class BootstrapError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[bootstrap] {message}", flush=True)


def fetch() -> bytes:
    if BUNDLE_URL:
        log("fetching bundle over https")
        with urllib.request.urlopen(BUNDLE_URL, timeout=120) as response:
            data = response.read(MAX_BYTES + 1)
    elif BUNDLE_BUCKET and BUNDLE_KEY:
        log(f"fetching s3://{BUNDLE_BUCKET}/{BUNDLE_KEY}")
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=BUNDLE_ENDPOINT or None,
            aws_access_key_id=BUNDLE_ACCESS_KEY or None,
            aws_secret_access_key=BUNDLE_SECRET_KEY or None,
            region_name="auto",
        )
        data = client.get_object(Bucket=BUNDLE_BUCKET, Key=BUNDLE_KEY)["Body"].read()
    else:
        raise BootstrapError(
            "no bundle source: set BUNDLE_URL, or BUNDLE_BUCKET and BUNDLE_KEY"
        )

    if len(data) > MAX_BYTES:
        raise BootstrapError(f"bundle exceeds {MAX_BYTES} bytes")
    return data


def verify(data: bytes) -> None:
    """Refuse to run code whose hash was not stated in advance.

    Missing is treated the same as wrong. An unverified bundle from a bucket
    whose credentials are in this container's environment is exactly the thing
    the hash exists to prevent, and defaulting to 'allow' when the variable is
    absent would make it decorative.
    """
    digest = hashlib.sha256(data).hexdigest()
    if not EXPECTED_SHA256:
        raise BootstrapError("BUNDLE_SHA256 is not set; refusing to run unverified code")
    if digest != EXPECTED_SHA256.lower():
        raise BootstrapError(f"bundle hash {digest} does not match the expected value")
    log(f"bundle verified: {digest[:16]}...")


def unpack(data: bytes) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
        handle.write(data)
        archive = Path(handle.name)

    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                # A member named ../../etc/passwd would otherwise be written
                # wherever it liked. Checked rather than trusted even though we
                # build the archive ourselves, because "we built it" stops
                # being true the moment the bucket is not what we think it is.
                target = (DEST / member.name).resolve()
                if not str(target).startswith(str(DEST.resolve())):
                    raise BootstrapError(f"archive member escapes the target: {member.name}")
                if member.issym() or member.islnk():
                    raise BootstrapError(f"archive contains a link: {member.name}")
            tar.extractall(DEST)
    finally:
        archive.unlink(missing_ok=True)

    log(f"unpacked to {DEST}")


def main() -> int:
    try:
        data = fetch()
        verify(data)
        unpack(data)
    except Exception as exc:  # noqa: BLE001
        # Fail loudly and immediately. On serverless this ends the job in
        # seconds; on a pod the supervisor never starts, so nothing renders and
        # the lease expires unrenewed.
        log(f"FATAL: {exc}")
        return 1

    entry = DEST / ENTRYPOINT
    if not entry.exists():
        log(f"FATAL: {entry} is not in the bundle")
        return 1

    log(f"exec {entry}")
    os.execv(sys.executable, [sys.executable, "-u", str(entry)])
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
