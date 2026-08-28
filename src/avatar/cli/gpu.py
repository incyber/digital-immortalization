"""Manual control over rented GPUs.

Exists so there is always a way to see what is running and stop it, without
opening a browser and without trusting that a previous run cleaned up.

    python -m avatar.cli.gpu status     what is running, and what it is costing
    python -m avatar.cli.gpu stop       terminate everything this project started
"""

from __future__ import annotations

import argparse
import os

from avatar.gpu.runpod import POD_TAG, GpuError, RunPodClient


def client() -> RunPodClient:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    return RunPodClient(key)


def cmd_status() -> None:
    api = client()
    balance = api.balance()
    print(f"balance: ${balance:.2f}" if balance >= 0 else "balance: unavailable")

    pods = api.list_pods()
    if not pods:
        print("no pods running - nothing is being billed")
        return

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
