#!/bin/sh
# Piper first, then the gateway in the foreground so the container's lifetime is
# the gateway's lifetime. If Piper dies the calls lose their voice; if the
# gateway dies the container should stop rather than sit there answering
# nothing.
set -e

python /opt/piper/server.py &
PIPER=$!

# Give it a moment to bind before the first call asks for speech. A failure
# here is not fatal: the gateway reports it per request rather than refusing to
# start, which is more useful than a container that will not come up.
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -sf http://127.0.0.1:7002/health >/dev/null 2>&1 && break
  sleep 1
done

trap 'kill $PIPER 2>/dev/null' EXIT
exec python -m uvicorn avatar.gateway.app:app --host 0.0.0.0 --port "${PORT:-8000}"
