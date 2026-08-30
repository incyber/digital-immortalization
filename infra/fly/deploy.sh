#!/usr/bin/env bash
# Create, credential and deploy the gateway on Fly.
#
# Everything this needs is either in fly.toml or in .env. No credential is
# written into this file, and none is printed: the values are read out of .env
# at run time and handed straight to `fly secrets set`, which is the only place
# they are stored.
#
# Safe to run repeatedly. App creation, volume creation and the session secret
# are all skipped if they already exist - re-running must not roll the session
# secret, because that would sign every customer out.
#
# Prerequisite this script cannot satisfy for you: the Fly organisation needs a
# payment method. Without one the API refuses to create an app at all.
set -euo pipefail

cd "$(dirname "$0")/../.."

APP="$(awk -F'"' '/^app *=/ {print $2; exit}' fly.toml)"
REGION="$(awk -F'"' '/^primary_region *=/ {print $2; exit}' fly.toml)"
VOLUME="assets"
VOLUME_GB=10

[ -f .env ] || { echo ".env not found; it holds the credentials this reads" >&2; exit 1; }

# Read one value out of .env without sourcing it. Sourcing would execute
# whatever is in there, and a comment containing a backtick is enough to make
# that interesting.
env_value() {
  awk -F= -v key="$1" '
    $1 == key { sub(/^[^=]*=/, ""); print; exit }
  ' .env
}

echo "==> app"
# Fly indents its table, so anchoring at the start of the line never
# matched an existing app and this tried to create one that was already
# there. Idempotence was the point of the check.
if ! flyctl apps list 2>/dev/null | grep -qE "(^|[[:space:]])${APP}([[:space:]]|$)"; then
  flyctl apps create "$APP" --org personal
else
  echo "    ${APP} already exists"
fi

echo "==> volume"
if ! flyctl volumes list -a "$APP" 2>/dev/null | grep -q "$VOLUME"; then
  flyctl volumes create "$VOLUME" -a "$APP" -r "$REGION" -s "$VOLUME_GB" -y
else
  echo "    volume ${VOLUME} already exists"
fi

echo "==> secrets"
ARGS=()
for key in \
  LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET \
  LLM_BASE_URL LLM_MODEL LLM_API_KEY \
  FALLBACK_LLM_PROVIDER_NAME FALLBACK_LLM_BASE_URL FALLBACK_LLM_API_KEY FALLBACK_LLM_MODEL \
  S3_ENDPOINT_URL S3_BUCKET S3_ACCESS_KEY S3_SECRET_KEY CLOUDFLARE_ACCOUNT_ID \
  RUNPOD_API_KEY RUNPOD_ENDPOINT_ID \
  CRISIS_LINES_VERIFIED
do
  value="$(env_value "$key")"
  if [ -z "$value" ]; then
    echo "    ${key} is empty in .env; refusing to set it blank" >&2
    exit 1
  fi
  ARGS+=("${key}=${value}")
done

# Generated here, once, and never reused from .env: the development default is
# published in config.py and anyone holding it can forge a session cookie for
# any account. gateway/sessions.py refuses to boot on it, which is the check
# doing its job rather than an obstacle.
if flyctl secrets list -a "$APP" 2>/dev/null | grep -q '^SESSION_SECRET'; then
  echo "    SESSION_SECRET already set; leaving it alone"
else
  ARGS+=("SESSION_SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')")
fi

# --stage so the machine is not restarted once per key; the deploy below picks
# all of them up in one go.
flyctl secrets set -a "$APP" --stage "${ARGS[@]}" >/dev/null
echo "    $(( ${#ARGS[@]} )) values staged"

echo "==> deploy"
# Remote builders: this workstation is arm64 with no Docker daemon, and Fly
# builds amd64 natively. It also means no registry credentials anywhere.
flyctl deploy --remote-only -a "$APP"

# Pinned, not assumed. A second machine would serve sign-in from one SQLite
# file and the photograph upload from another, with the call agent spawned on
# whichever host happened to answer and reading a volume the assets are not on.
echo "==> pin to one machine"
flyctl scale count 1 -a "$APP" -y

echo "==> health"
curl -fsS "https://${APP}.fly.dev/health" && echo
