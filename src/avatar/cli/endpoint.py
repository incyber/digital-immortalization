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


# The environment a splat worker needs. Split in two on purpose: the bundle
# variables tell the bootstrap what code to run, and the S3 variables tell that
# code where to put what it produces. Both are read from this machine's
# environment so the values live in one place rather than in the platform's
# console where nobody can diff them.
BUNDLE_VARS = (
    "BUNDLE_ENDPOINT_URL",
    "BUNDLE_BUCKET",
    "BUNDLE_KEY",
    "BUNDLE_SHA256",
    "BUNDLE_ACCESS_KEY",
    "BUNDLE_SECRET_KEY",
)
STORAGE_VARS = ("S3_ENDPOINT_URL", "S3_BUCKET", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_REGION")


def worker_env(bundle_name: str) -> dict[str, str]:
    """Build the worker environment from this machine's, deriving the digest.

    The digest is computed rather than read, because a stale one is the worst
    of the failure modes available here: the bootstrap refuses the archive and
    the worker exits with nothing useful to show for it.
    """
    from avatar.cli.bundle import ROOT
    from avatar.gpu.bundle import build

    bundle = build(bundle_name, ROOT)

    env = {
        "BUNDLE_ENDPOINT_URL": os.environ.get("S3_ENDPOINT_URL", ""),
        "BUNDLE_BUCKET": os.environ.get("S3_BUCKET", ""),
        "BUNDLE_KEY": bundle.key,
        "BUNDLE_SHA256": bundle.sha256,
        "BUNDLE_ACCESS_KEY": os.environ.get("S3_ACCESS_KEY", ""),
        "BUNDLE_SECRET_KEY": os.environ.get("S3_SECRET_KEY", ""),
    }
    for name in STORAGE_VARS:
        env[name] = os.environ.get(name, "auto" if name == "S3_REGION" else "")

    # The platform's SDK runs system fitness checks before it starts polling for
    # work, and one of them imports torch to verify CUDA. On this image that
    # import takes about ten seconds, the worker has not pinged in that time,
    # and the platform stops it - every time, forever, with jobs sitting in the
    # queue behind a worker that was killed on its way to becoming useful.
    #
    # Skipped rather than sped up, because the check is in the wrong place
    # rather than merely slow: this worker already has a `health` task that
    # reports whether torch, CUDA and the weights are present, and it runs when
    # somebody asks instead of on the path to accepting a job.
    env["RUNPOD_SKIP_AUTO_SYSTEM_CHECKS"] = "true"

    empty = [name for name, value in env.items() if not value]
    if empty:
        raise SystemExit("not set in this environment: " + ", ".join(empty))
    return env


def cmd_retarget(args: argparse.Namespace) -> None:
    provisioner = Provisioner(key())
    template_id = Provisioner(key()).read_endpoint(endpoint_id())["templateId"]
    env = worker_env(args.bundle)

    provisioner.retarget_template(template_id, args.image, env)

    print(f"template {template_id} now runs {args.image}")
    print(f"  BUNDLE_KEY:    {env['BUNDLE_KEY']}")
    print(f"  BUNDLE_SHA256: {env['BUNDLE_SHA256']}")
    print("\nThe next job queued starts a worker on it.")


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

    retarget = sub.add_parser("retarget", help="point the endpoint at a new image")
    retarget.add_argument("image")
    retarget.add_argument("--bundle", default="splat")

    args = parser.parse_args()
    try:
        {
            "create": cmd_create,
            "verify": cmd_verify,
            "delete": cmd_delete,
            "retarget": cmd_retarget,
        }[args.command](args)
    except ServerlessError as exc:
        raise SystemExit(f"runpod: {exc}") from exc


if __name__ == "__main__":
    main()
