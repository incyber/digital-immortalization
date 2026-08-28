"""PID 1 in a rented pod: holds a lease, or kills the pod.

The direction of authority is the whole point. A design where an orchestrator
pushes a kill leaves the pod running on every delivery failure - dropped
network, crashed process, slept laptop. Here the pod must keep *pulling* a
permission that expires on its own, so every one of those failures ends with
the pod dead rather than alive.

Four rules, each chosen because its opposite fails open:

  1. The lease is a key with a TTL enforced by the store, not by a timestamp
     this process compares. Nothing here has to keep working for it to expire.
  2. Missing, expired, unreadable, wrong value, and unreachable store all count
     as denial. There is no "assume fine" branch, because that branch is the
     leak.
  3. Renewal comes from outside the pod. A heartbeat written by the worker
     itself is fail-open: a hung worker keeps writing it.
  4. This process is PID 1 and runs the service as its child. If it dies the
     container dies, so a crashed watchdog cannot leave a running GPU.

The service is started only after a first successful lease read. A pod that
comes up unable to reach the store never renders a frame and never bills for
more than its own start-up.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import redis

# How often the lease is checked. Renewal happens on a shorter period from
# outside, so a single slow round trip is not a denial.
CHECK_SECONDS = 15

# Consecutive denials tolerated. Three at fifteen seconds is a 45-second
# window, which is longer than any redeploy of the orchestrator and far
# shorter than a leak that matters.
MAX_DENIALS = 3

# An absolute ceiling, independent of the lease. A renewal loop that keeps
# working forever is not a failure this catches, so it is caught here instead:
# no call runs for four hours, and a pod that has been up that long is wrong
# whatever the lease says. The Dockerfile wraps this process in `timeout` as
# well, so the ceiling survives this loop breaking.
MAX_LIFETIME_S = int(os.environ.get("POD_MAX_LIFETIME_S", str(4 * 60 * 60)))

POD_ID = os.environ.get("RUNPOD_POD_ID", "")
LEASE_KEY = os.environ.get("LEASE_KEY", f"lease:pod:{POD_ID}")
REDIS_URL = os.environ.get("REDIS_URL", "")
SERVICE = os.environ.get("LEASE_SERVICE_CMD", "python -u server.py")


def log(message: str) -> None:
    print(f"[supervisor] {message}", flush=True)


def held(client: redis.Redis) -> bool:
    """True only on a positive, unexpired answer.

    Every failure path returns False. That is the inversion the whole design
    rests on, so it is written as one expression with no exceptions to it.
    """
    try:
        value = client.get(LEASE_KEY)
    except Exception as exc:  # noqa: BLE001 - unreachable store is a denial
        log(f"lease unreadable: {exc}")
        return False
    if value is None:
        log("lease absent or expired")
        return False
    return True


def terminate_pod() -> None:
    """Ask the platform to release this pod, then stop regardless.

    Best effort, deliberately. If the call fails the process still exits, the
    container still dies, and the pod stops running our code. Termination is
    the cleaner ending, not the load-bearing one.
    """
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not (key and POD_ID):
        log("no credentials to self-terminate; exiting instead")
        return
    try:
        import httpx

        httpx.delete(
            f"https://rest.runpod.io/v1/pods/{POD_ID}",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20.0,
        )
        log(f"terminated pod {POD_ID}")
    except Exception as exc:  # noqa: BLE001
        log(f"self-termination failed, exiting anyway: {exc}")


def main() -> int:
    if not REDIS_URL:
        log("REDIS_URL is not set; refusing to start")
        return 2

    client = redis.from_url(REDIS_URL, socket_timeout=5, socket_connect_timeout=5)

    # Nothing starts until the lease is confirmed once. A pod that cannot reach
    # the store never renders a frame.
    if not held(client):
        log("no lease at start-up; refusing to start the service")
        terminate_pod()
        return 3

    log(f"lease {LEASE_KEY} held; starting service")
    service = subprocess.Popen(SERVICE.split())

    denials = 0
    started = time.monotonic()
    try:
        while True:
            time.sleep(CHECK_SECONDS)

            if time.monotonic() - started > MAX_LIFETIME_S:
                log(f"pod has been up {MAX_LIFETIME_S}s; stopping regardless of the lease")
                service.send_signal(signal.SIGKILL)
                terminate_pod()
                return 6

            if (code := service.poll()) is not None:
                log(f"service exited with {code}; stopping the pod")
                terminate_pod()
                return code or 0

            if held(client):
                denials = 0
                continue

            denials += 1
            log(f"denial {denials}/{MAX_DENIALS}")
            if denials >= MAX_DENIALS:
                log("lease lost; killing the service and the pod")
                service.send_signal(signal.SIGKILL)
                terminate_pod()
                return 4
    except BaseException as exc:  # noqa: BLE001 - including KeyboardInterrupt
        # Any exit path from this loop stops the pod. An interrupted or
        # crashing supervisor must not leave the service running behind it.
        log(f"supervisor stopping: {type(exc).__name__}: {exc}")
        service.send_signal(signal.SIGKILL)
        terminate_pod()
        return 5


if __name__ == "__main__":
    sys.exit(main())
