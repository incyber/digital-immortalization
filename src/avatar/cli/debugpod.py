"""Rent one GPU, run commands on it, give it back.

    python -m avatar.cli.debugpod start
    python -m avatar.cli.debugpod run 'nvidia-smi'
    python -m avatar.cli.debugpod run --file scratch/fit.py
    python -m avatar.cli.debugpod stop

The splat worker had never run anywhere anybody could watch it. Serverless
gives no shell and no reachable log, so every question about it cost a
fifteen-minute round trip through a bundle publish, a template retarget and a
fresh image pull, and came back with about one line of evidence. This rents an
ordinary pod on the same image, where the answer takes seconds.

Commands travel through the operator's own bucket rather than a port. See
infra/splatworker/agent.py for why.

BILLING
=======
A pod bills for every minute it exists, including while it is doing nothing and
including after the process inside it has exited. That is the whole difference
from serverless and it is the reason this file exists as a tool rather than as
a thing anybody does by hand in a console: `stop` is one command, `status`
shows what is running, and `start` refuses to create a second pod while one is
already up.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import boto3
import httpx

REST = "https://rest.runpod.io/v1"

# The same image the serverless worker runs. Debugging a different image would
# answer a question nobody asked.
DEFAULT_IMAGE = "ghcr.io/incyber/digital-immortalization/splatworker-reconstruct:v4"

# Where the pod id is remembered between invocations, so `stop` does not need
# to be told what to stop. In the state directory rather than the repository:
# it is a fact about this machine's rented hardware, not about the code.
STATE = Path.home() / ".cache" / "digital-immortalization" / "debugpod.json"

POD_NAME = "avatar-splat-debug"
GPU_TYPES = ("NVIDIA RTX A5000", "NVIDIA L4")
DISK_GB = 60


def key() -> str:
    value = os.environ.get("RUNPOD_API_KEY", "")
    if not value:
        raise SystemExit("RUNPOD_API_KEY is not set")
    return value


def api(method: str, path: str, **kwargs) -> dict:
    response = httpx.request(
        method,
        f"{REST}{path}",
        headers={"Authorization": f"Bearer {key()}"},
        timeout=60,
        **kwargs,
    )
    if response.status_code >= 400:
        raise SystemExit(f"runpod {method} {path} -> {response.status_code}: {response.text[:400]}")
    return response.json() if response.content else {}


def storage():
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


def load_state() -> dict:
    if not STATE.exists():
        return {}
    import json

    return json.loads(STATE.read_text())


def save_state(state: dict) -> None:
    import json

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))


# --------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> None:
    state = load_state()
    if state.get("pod_id"):
        raise SystemExit(
            f"pod {state['pod_id']} is already recorded as running. "
            "Run 'stop' first, or 'status' to see it."
        )

    from avatar.cli.bundle import ROOT
    from avatar.cli.endpoint import worker_env
    from avatar.gpu.bundle import build
    from avatar.gpu.provision import Provisioner

    bundle = build("debug", ROOT)
    storage().put_object(
        Bucket=os.environ["S3_BUCKET"],
        Key=bundle.key,
        Body=bundle.data,
        ContentType="application/gzip",
    )
    print(f"published {bundle.key} ({len(bundle.data) // 1024}KB)")

    session = uuid.uuid4().hex[:8]
    env = worker_env("debug")
    env["DEBUG_SESSION"] = session

    auth_id = None
    token = os.environ.get("GHCR_TOKEN", "")
    if token:
        auth_id = Provisioner(key()).create_registry_auth(
            f"{POD_NAME}-ghcr", os.environ.get("GHCR_USERNAME", ""), token
        )

    body = {
        "name": POD_NAME,
        "imageName": args.image,
        "computeType": "GPU",
        "gpuTypeIds": list(GPU_TYPES),
        "gpuCount": 1,
        "containerDiskInGb": DISK_GB,
        # Nothing here outlives the session, and a volume keeps charging after
        # the pod is gone.
        "volumeInGb": 0,
        "ports": [],
        "env": env,
    }
    if auth_id:
        body["containerRegistryAuthId"] = auth_id

    created = api("POST", "/pods", json=body)
    pod_id = created.get("id")
    if not pod_id:
        raise SystemExit(f"pod creation returned no id: {created}")

    save_state({"pod_id": pod_id, "session": session, "started": time.time()})
    print(f"\npod {pod_id} starting, session {session}")
    print(f"cost {created.get('costPerHr', '?')}/hr - it bills until you run 'stop'")
    print("\nThe image is large; the first command will wait for the pull.")


def cmd_run(args: argparse.Namespace) -> None:
    state = load_state()
    if not state.get("pod_id"):
        raise SystemExit("no pod is recorded as running; 'start' one first")

    script = Path(args.file).read_text() if args.file else args.script
    if not script:
        raise SystemExit("nothing to run")

    s3 = storage()
    bucket = os.environ["S3_BUCKET"]
    session = state["session"]
    name = f"{int(time.time() * 1000)}.sh"

    s3.put_object(
        Bucket=bucket, Key=f"debug/{session}/in/{name}", Body=script.encode(), ContentType="text/plain"
    )
    print(f"sent {name}; waiting up to {args.wait}s", file=sys.stderr)

    deadline = time.time() + args.wait
    out_key = f"debug/{session}/out/{name}"
    while time.time() < deadline:
        try:
            print(s3.get_object(Bucket=bucket, Key=out_key)["Body"].read().decode())
            return
        except s3.exceptions.NoSuchKey:
            time.sleep(2)

    raise SystemExit(f"no answer within {args.wait}s; the pod may still be pulling the image")


def cmd_push(args: argparse.Namespace) -> None:
    """Put the working copy of the worker's code on the running pod.

    The pod fetched its code once, at start. Without this, every edit would
    mean terminating the pod and pulling that very large image again, which is
    the fifteen-minute loop this tool exists to escape.
    """
    from avatar.cli.bundle import ROOT
    from avatar.gpu.bundle import build

    bundle = build("debug", ROOT)
    storage().put_object(
        Bucket=os.environ["S3_BUCKET"],
        Key=bundle.key,
        Body=bundle.data,
        ContentType="application/gzip",
    )
    print(f"published {bundle.key}")

    # Unpacked over /opt/app, deliberately not into a fresh directory: the
    # agent is running from there and the next command must see the new code.
    args.script = f"""
python - <<'PUSH'
import os, io, tarfile, boto3
s3 = boto3.client("s3", endpoint_url=os.environ["BUNDLE_ENDPOINT_URL"],
                  aws_access_key_id=os.environ["BUNDLE_ACCESS_KEY"],
                  aws_secret_access_key=os.environ["BUNDLE_SECRET_KEY"],
                  region_name="auto")
blob = s3.get_object(Bucket=os.environ["BUNDLE_BUCKET"], Key="{bundle.key}")["Body"].read()
with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
    tar.extractall("/opt/app")
print("updated", sorted(os.listdir("/opt/app")))
PUSH
"""
    args.file = None
    cmd_run(args)


def cmd_status(_args: argparse.Namespace) -> None:
    state = load_state()
    if not state.get("pod_id"):
        print("no pod recorded")
        return

    pod = api("GET", f"/pods/{state['pod_id']}")
    minutes = (time.time() - state.get("started", time.time())) / 60
    print(f"pod        {state['pod_id']}")
    print(f"status     {pod.get('desiredStatus')} / {pod.get('lastStatusChange', '')}")
    print(f"machine    {pod.get('machine', {}).get('gpuTypeId', '?')}")
    print(f"running    {minutes:.0f} minutes")
    print(f"cost/hr    {pod.get('costPerHr', '?')}")


def cmd_stop(_args: argparse.Namespace) -> None:
    state = load_state()
    pod_id = state.get("pod_id")
    if not pod_id:
        print("no pod recorded; nothing to stop")
        return

    api("DELETE", f"/pods/{pod_id}")
    minutes = (time.time() - state.get("started", time.time())) / 60
    save_state({})
    print(f"pod {pod_id} terminated after {minutes:.0f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    start = sub.add_parser("start", help="rent a GPU and put the agent on it")
    start.add_argument("--image", default=DEFAULT_IMAGE)

    run = sub.add_parser("run", help="run one command on it and print the output")
    run.add_argument("script", nargs="?", default="")
    run.add_argument("--file", help="run a local script instead of a command string")
    run.add_argument("--wait", type=int, default=900)

    push = sub.add_parser("push", help="send the working copy of the code to the pod")
    push.add_argument("--wait", type=int, default=300)

    sub.add_parser("status", help="what is running and for how long")
    sub.add_parser("stop", help="terminate it")

    args = parser.parse_args()
    {
        "start": cmd_start,
        "run": cmd_run,
        "push": cmd_push,
        "status": cmd_status,
        "stop": cmd_stop,
    }[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
