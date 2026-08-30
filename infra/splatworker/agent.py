"""A shell on a GPU, over object storage.

WHY THIS EXISTS
===============
The splat worker had never run. Not once, anywhere anybody could watch it. Every
attempt to find out why was a fifteen-minute round trip: publish a bundle,
retarget a template, recycle a worker, wait for a fresh host to pull a very
large image, then read whatever breadcrumbs survived. The platform's own worker
logs are not reachable from its API, so each cycle returned about one line of
evidence, and a wrong guess cost the same as a right one.

This turns that loop into seconds. It runs as the entrypoint of an ordinary
GPU pod built from the same image the worker uses, polls a prefix in the
operator's own bucket for shell scripts, runs each one, and writes back its
output. No inbound port is opened and no new credential is minted: it uses the
same read/write key the worker already has, scoped to the same bucket.

It is a debugging tool and it is written like one - it will run whatever it is
handed. The safety is in the delivery: the only way to reach it is to write an
object into a private bucket, which is a thing only somebody holding the
operator's own storage key can do.

DELIBERATELY NOT A NETWORK SERVICE
==================================
The obvious alternative was an HTTP endpoint on the pod's public proxy, which
would have been quicker to write and would have put a remote execution service
on the open internet behind nothing but an unguessable hostname. Polling a
private bucket has no listening socket at all.

USAGE
=====
    python -m avatar.cli.debugpod start        create the pod
    python -m avatar.cli.debugpod run 'ls /'   run a command, print its output
    python -m avatar.cli.debugpod stop         terminate it

A pod bills for every minute it exists, so `stop` is not optional and `start`
arms a watchdog that stops it regardless.
"""

from __future__ import annotations

import os
import subprocess
import time
import traceback

import boto3
from botocore.config import Config

BUCKET = os.environ.get("BUNDLE_BUCKET") or os.environ.get("S3_BUCKET", "")
ENDPOINT = os.environ.get("BUNDLE_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT_URL", "")
ACCESS_KEY = os.environ.get("BUNDLE_ACCESS_KEY") or os.environ.get("S3_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("BUNDLE_SECRET_KEY") or os.environ.get("S3_SECRET_KEY", "")

# Where commands arrive and answers go back. One prefix per session so a stale
# command from an earlier pod cannot be picked up by a later one.
SESSION = os.environ.get("DEBUG_SESSION", "default")
INBOX = f"debug/{SESSION}/in/"
OUTBOX = f"debug/{SESSION}/out/"

POLL_S = 2.0

# Nothing here should run for an hour. A command that does is a mistake, and
# killing it leaves the agent able to accept the next one instead of requiring
# the pod to be recreated.
COMMAND_TIMEOUT_S = 1800

# The pod bills whether or not anybody is using it. If no command arrives for
# this long the agent stops asking for work and says so in its last message;
# the operator's watchdog is what actually terminates the pod, because a
# process cannot be trusted to bill-manage the machine it runs on.
IDLE_LIMIT_S = 3600


def client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT or None,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="auto",
        config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 3}),
    )


def say(s3, name: str, text: str) -> None:
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{OUTBOX}{name}",
        Body=text.encode()[:4_000_000],
        ContentType="text/plain",
    )


def run(script: str) -> str:
    """Run one script, returning everything it said and how it ended."""
    started = time.time()
    try:
        finished = subprocess.run(
            script,
            shell=True,  # running arbitrary shell is the entire point of this
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
            cwd="/opt/app",
        )
        body = finished.stdout + finished.stderr
        status = f"exit {finished.returncode}"
    except subprocess.TimeoutExpired as expired:
        body = (expired.stdout or "") + (expired.stderr or "")
        if isinstance(body, bytes):
            body = body.decode(errors="replace")
        status = f"timed out after {COMMAND_TIMEOUT_S}s"
    except Exception:  # noqa: BLE001 - the agent must outlive a bad command
        body = traceback.format_exc()
        status = "the agent itself failed to run it"

    return f"[{status} in {time.time() - started:.1f}s]\n{body}"


def main() -> int:
    if not BUCKET:
        print("[agent] no bucket configured; nothing to poll", flush=True)
        return 1

    s3 = client()
    say(s3, "0000-ready.txt", f"agent up on {os.uname().nodename}\n")
    print(f"[agent] polling s3://{BUCKET}/{INBOX}", flush=True)

    seen: set[str] = set()
    last_command = time.time()

    while True:
        if time.time() - last_command > IDLE_LIMIT_S:
            say(s3, "zzzz-idle.txt", "no command for an hour; agent stopped asking\n")
            print("[agent] idle limit reached", flush=True)
            return 0

        try:
            listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=INBOX).get("Contents", [])
        except Exception as exc:  # noqa: BLE001 - a blip must not end the session
            print(f"[agent] list failed: {exc}", flush=True)
            time.sleep(POLL_S)
            continue

        fresh = sorted(o["Key"] for o in listing if o["Key"] not in seen)
        if not fresh:
            time.sleep(POLL_S)
            continue

        for key in fresh:
            seen.add(key)
            name = key[len(INBOX) :]
            script = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()
            print(f"[agent] running {name}", flush=True)
            say(s3, name, run(script))
            last_command = time.time()


if __name__ == "__main__":
    raise SystemExit(main())
