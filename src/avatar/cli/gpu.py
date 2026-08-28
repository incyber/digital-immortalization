"""Manual control over rented GPUs.

Exists so there is always a way to see what is running and stop it, without
opening a browser and without trusting that a previous run cleaned up. It is
the one tool here that is deliberately outside the system it inspects: it
believes nothing the application says and asks the platform directly.

    python -m avatar.cli.gpu status     what is running, and what it is costing
    python -m avatar.cli.gpu stop       terminate everything this project started

Reports on both shapes. Serverless endpoints are what the project now uses and
bill only while a worker runs; Pods bill until something stops them and should
not exist at all. A Pod appearing in `status` is a finding, not a reading.
"""

from __future__ import annotations

import argparse
import os

import httpx

from avatar.gpu.runpod import POD_TAG, GpuError, RunPodClient

REST = "https://rest.runpod.io/v1"


def endpoints(key: str) -> list[dict]:
    """Serverless endpoints on the account, straight from the platform."""
    with httpx.Client(timeout=30.0) as http:
        response = http.get(
            f"{REST}/endpoints", headers={"Authorization": f"Bearer {key}"}
        )
        if response.status_code >= 400:
            raise GpuError(f"listing endpoints -> {response.status_code}")
        return response.json() or []


def report_endpoints(key: str) -> None:
    found = endpoints(key)
    if not found:
        print("\nno serverless endpoints")
        return

    print(f"\n{len(found)} serverless endpoint(s):")
    for endpoint in found:
        idle = endpoint.get("idleTimeout")
        execution = int(endpoint.get("executionTimeoutMs") or 0) // 1000
        workers_min = endpoint.get("workersMin")
        # Flagged rather than merely printed. workersMin above zero is the one
        # setting that bills with no job running at all, and it is the setting
        # a console click can change without anything here noticing.
        warning = "  <-- BILLS WHILE IDLE" if workers_min else ""
        print(
            f"  {endpoint.get('id'):<20} {endpoint.get('name','?'):<24} "
            f"min={workers_min} max={endpoint.get('workersMax')} "
            f"idle={idle}s exec={execution}s{warning}"
        )


def client() -> RunPodClient:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    return RunPodClient(key)


def cmd_status() -> None:
    api = client()
    balance = api.balance()
    print(f"balance: ${balance:.2f}" if balance >= 0 else "balance: unavailable")

    report_endpoints(os.environ["RUNPOD_API_KEY"])

    pods = api.list_pods()
    if not pods:
        print("\nno pods - nothing is allocated")
        return

    print("\nPODS ARE RUNNING. This project uses serverless; a pod here is a leak.")

    print(f"\n{len(pods)} pod(s) running:")
    for pod in pods:
        env = pod.get("env") or {}
        mine = env.get("AVATAR_TAG") == POD_TAG or pod.get("name", "").startswith(POD_TAG)
        print(
            f"  {pod.get('id'):<24} {pod.get('name','?'):<28} "
            f"{pod.get('desiredStatus','?'):<10} {'(ours)' if mine else '(not ours)'}"
        )
    print("\nAnything listed here is being billed. `stop` terminates ours.")


def cmd_stop() -> None:
    api = client()
    killed = api.sweep()
    if killed:
        print(f"terminated {len(killed)}: {', '.join(killed)}")
    else:
        print("nothing of ours was running")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="what is running and what it costs")
    sub.add_parser("stop", help="terminate every pod this project started")

    args = parser.parse_args()
    try:
        {"status": cmd_status, "stop": cmd_stop}[args.command]()
    except GpuError as exc:
        raise SystemExit(f"runpod: {exc}") from exc


if __name__ == "__main__":
    main()
