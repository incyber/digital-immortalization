"""Publish the private half of a GPU worker.

    python -m avatar.cli.bundle publish serverless
    python -m avatar.cli.bundle publish musetalk

Packs the files a worker needs, uploads them to the operator's own bucket, and
prints the environment a worker needs to find and verify them. The digest is
the part that matters: the bootstrap in the public image refuses to execute an
archive whose hash was not stated in advance, so publishing and configuring are
deliberately two steps rather than one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3

from avatar.gpu.bundle import BUNDLES, build

ROOT = Path(__file__).resolve().parents[3]


def client():
    for name in ("S3_ENDPOINT_URL", "S3_BUCKET", "S3_ACCESS_KEY", "S3_SECRET_KEY"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} is not set")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="auto",
    )


def cmd_publish(args: argparse.Namespace) -> None:
    bundle = build(args.name, ROOT)
    bucket = os.environ["S3_BUCKET"]

    client().put_object(
        Bucket=bucket,
        Key=bundle.key,
        Body=bundle.data,
        ContentType="application/gzip",
    )

    print(f"\npublished s3://{bucket}/{bundle.key}")
    print(f"{len(bundle.data) // 1024}KB, sha256 {bundle.sha256}")
    print("\nWorker environment:\n")
    for line in (
        f"BUNDLE_ENDPOINT_URL={os.environ['S3_ENDPOINT_URL']}",
        f"BUNDLE_BUCKET={bucket}",
        f"BUNDLE_KEY={bundle.key}",
        f"BUNDLE_SHA256={bundle.sha256}",
        "BUNDLE_ACCESS_KEY=<the read-only key>",
        "BUNDLE_SECRET_KEY=<the read-only secret>",
    ):
        print(f"  {line}")


def cmd_digest(args: argparse.Namespace) -> None:
    """The hash, without uploading anything. Useful for checking a deployment."""
    bundle = build(args.name, ROOT)
    print(bundle.sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    publish = sub.add_parser("publish", help="pack and upload a bundle")
    publish.add_argument("name", choices=sorted(BUNDLES))
    digest = sub.add_parser("digest", help="print the hash without uploading")
    digest.add_argument("name", choices=sorted(BUNDLES))

    args = parser.parse_args()
    {"publish": cmd_publish, "digest": cmd_digest}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
