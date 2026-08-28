"""Answers, and nothing else."""

import runpod


def handler(job):
    return {"ok": True, "echo": job.get("input")}


runpod.serverless.start({"handler": handler})
