#!/bin/sh
# Exercises the deadman switch against a fake terminate.
#
# The point of these is that they do not use the real API: what is being tested
# is whether the switch fires, not whether RunPod answers.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT="$HERE/../../infra/deadman.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
FAILURES=0

check() {
  if [ "$2" = "$3" ]; then echo "  ok   $1"
  else echo "  FAIL $1: expected '$3', got '$2'"; FAILURES=$((FAILURES+1)); fi
}

echo "deadman switch:"

# The hard ceiling fires even when the heartbeat is healthy - a job that runs
# too long has hung, and a healthy heartbeat does not make it not-hung.
HB="$WORK/hb"; touch "$HB"
( while true; do touch "$HB"; sleep 1; done ) & TOUCHER=$!
DEADMAN_MAX_MINUTES=0 DEADMAN_IDLE_MINUTES=99 DEADMAN_HEARTBEAT="$HB" \
  RUNPOD_POD_ID= RUNPOD_API_KEY= sh "$SCRIPT" >"$WORK/out1" 2>&1 || true
kill $TOUCHER 2>/dev/null || true
grep -q "hard ceiling" "$WORK/out1" && R=fired || R=silent
check "fires on the hard ceiling despite a live heartbeat" "$R" "fired"

# A stale heartbeat fires the idle ceiling: the work is gone but the GPU is
# still rented.
HB2="$WORK/hb2"; touch -t 200001010000 "$HB2" 2>/dev/null || touch "$HB2"
DEADMAN_MAX_MINUTES=99 DEADMAN_IDLE_MINUTES=0 DEADMAN_HEARTBEAT="$HB2" \
  RUNPOD_POD_ID= RUNPOD_API_KEY= sh "$SCRIPT" >"$WORK/out2" 2>&1 || true
grep -q "heartbeat" "$WORK/out2" && R=fired || R=silent
check "fires when the heartbeat goes stale" "$R" "fired"

# Missing credentials must not stop it from acting. A switch that refuses to
# fire because it cannot reach the API is worse than one that stops the
# container and leaves evidence.
grep -q "WARNING" "$WORK/out2" && R=warned || R=quiet
check "warns when it cannot reach the API but still acts" "$R" "warned"

# It must say what it is doing on start-up: a silent watchdog is one nobody
# notices has stopped.
grep -q "armed" "$WORK/out1" && R=announced || R=silent
check "announces its ceilings on start" "$R" "announced"

echo
[ "$FAILURES" -eq 0 ] && echo "all deadman checks passed" || { echo "$FAILURES failed"; exit 1; }
