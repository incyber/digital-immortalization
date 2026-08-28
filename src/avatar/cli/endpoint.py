"""Create, verify and remove the serverless endpoint.

Separate from `avatar.cli.gpu`, which reports and stops. This one changes
things, and the split is deliberate: the tool you reach for when you are
worried about spend should not be the tool that can create something that
spends.

    python -m avatar.cli.endpoint create   build it, then verify what was stored
    python -m avatar.cli.endpoint verify   re-read the settings and check them
    python -m avatar.cli.endpoint delete   remove it

`verify` is worth running on its own. Provisioning checks the settings once at
creation, but a console click can change workersMin afterwards and nothing in
the application would notice.
"""

from __future__ import annotations

import argparse
import os

from avatar.gpu.provision import Provisioner
from avatar.gpu.serverless import ServerlessError, assert_endpoint_is_safe

DEFAULT_NAME = "avatar-gpu-worker"
DEFAULT_IMAGE = "ghcr.io/incyber/digital-immortalization/gpu-worker:v1"


def key() -> str:
    value = os.environ.get("RUNPOD_API_KEY", "")
    if not value:
        raise SystemExit("RUNPOD_API_KEY is not set")
    return value


def endpoint_id() -> str:
    value = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    if not value:
        raise SystemExit("RUNPOD_ENDPOINT_ID is not set")
    return value


def cmd_create(args: argparse.Namespace) -> None:
    provisioner = Provisioner(key())

    auth_id = None
    token = os.environ.get("GHCR_TOKEN", "")
    if token:
        auth_id = provisioner.create_registry_auth(
            f"{args.name}-ghcr", os.environ.get("GHCR_USERNAME", ""), token
        )
    else:
        print("no GHCR_TOKEN set; assuming the image is public")

    result = provisioner.provision(args.name, args.image, registry_auth_id=auth_id)

    print(f"\nendpoint: {result.endpoint_id}")
    print("verified against the platform:")
    for name, value in result.verified.items():
        print(f"  {name}: {value}")
    print(f"\nRUNPOD_ENDPOINT_ID={result.endpoint_id}")
    print("\nNothing is running. A worker starts when the first job is queued.")


def cmd_verify(_args: argparse.Namespace) -> None:
    stored = Provisioner(key()).read_endpoint(endpoint_id())
    problems = assert_endpoint_is_safe(stored)

    for name in ("workersMin", "workersMax", "idleTimeout", "executionTimeoutMs"):
        print(f"  {name}: {stored.get(name)}")

    if problems:
        # Non-zero exit so this is usable as a check rather than only as a
        # thing to read.
        raise SystemExit("\nUNSAFE:\n  " + "\n  ".join(problems))
    print("\nsafe: nothing is allocated between jobs")


def cmd_delete(_args: argparse.Namespace) -> None:
    Provisioner(key()).delete_endpoint(endpoint_id())
    print("deleted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="create and verify the endpoint")
    create.add_argument("--name", default=DEFAULT_NAME)
    create.add_argument("--image", default=DEFAULT_IMAGE)
    sub.add_parser("verify", help="re-read the stored settings and check them")
    sub.add_parser("delete", help="remove the endpoint")

    args = parser.parse_args()
    try:
        {"create": cmd_create, "verify": cmd_verify, "delete": cmd_delete}[
            args.command
        ](args)
    except ServerlessError as exc:
        raise SystemExit(f"runpod: {exc}") from exc


if __name__ == "__main__":
    main()
