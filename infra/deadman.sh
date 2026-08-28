#!/bin/sh
# Deadman switch. Runs inside the pod and terminates it from the inside.
#
# Every other safeguard in this project assumes some process outside the pod
# is still alive to stop it. That assumption fails on SIGKILL, on a laptop
# going to sleep, on a crashed orchestrator, on a dropped network. This one
# does not depend on anything outside the pod, because it is the pod.
#
# Two independent triggers:
#
#   Hard ceiling  - after MAX_MINUTES the pod terminates itself, whatever it
#                   was doing. A job that outruns this has hung.
#
#   Heartbeat     - the work touches /tmp/heartbeat while it is alive. If that
#                   file stops being updated for IDLE_MINUTES, the work is gone
#                   and the pod is being billed for nothing.
#
# It terminates via the RunPod API using the pod's own id, so it works even if
# the container's main process is already dead.
set -eu

MAX_MINUTES="${DEADMAN_MAX_MINUTES:-30}"
IDLE_MINUTES="${DEADMAN_IDLE_MINUTES:-10}"
HEARTBEAT="${DEADMAN_HEARTBEAT:-/tmp/heartbeat}"
POD_ID="${RUNPOD_POD_ID:-}"
API_KEY="${RUNPOD_API_KEY:-}"

log() { echo "[deadman] $*" >&2; }

terminate() {
  log "terminating pod ${POD_ID}: $1"
  if [ -n "$POD_ID" ] && [ -n "$API_KEY" ]; then
    curl -s -X DELETE "https://rest.runpod.io/v1/pods/${POD_ID}" \
         -H "Authorization: Bearer ${API_KEY}" >/dev/null 2>&1 || true
  fi
  # If the API call failed - no key, no network - stop the container anyway.
  # A stopped container is not a stopped pod, but it is a smaller bill and it
  # makes the leak visible in the console.
  sleep 5
  kill -TERM 1 2>/dev/null || true
  exit 0
}

log "armed: hard ceiling ${MAX_MINUTES}m, idle ceiling ${IDLE_MINUTES}m"
[ -n "$POD_ID" ] || log "WARNING: RUNPOD_POD_ID unset; can only stop the container"
[ -n "$API_KEY" ] || log "WARNING: RUNPOD_API_KEY unset; can only stop the container"

touch "$HEARTBEAT" 2>/dev/null || true
START=$(date +%s)

while true; do
  sleep 30
  NOW=$(date +%s)

  ELAPSED_MIN=$(( (NOW - START) / 60 ))
  if [ "$ELAPSED_MIN" -ge "$MAX_MINUTES" ]; then
    terminate "hard ceiling of ${MAX_MINUTES} minutes reached"
  fi

  if [ -f "$HEARTBEAT" ]; then
    LAST=$(stat -c %Y "$HEARTBEAT" 2>/dev/null || stat -f %m "$HEARTBEAT" 2>/dev/null || echo "$NOW")
    IDLE_MIN=$(( (NOW - LAST) / 60 ))
    if [ "$IDLE_MIN" -ge "$IDLE_MINUTES" ]; then
      terminate "no heartbeat for ${IDLE_MIN} minutes; the work is gone"
    fi
  fi
done
