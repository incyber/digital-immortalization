#!/bin/bash
# macOS launcher for the DI web prototype - the equivalent of Start-Web.bat / server.ps1
# on Windows. Double-click this file in Finder, or run ./start-web.command from a terminal.
#
# What it does, in order:
#   1. Starts the Ollama background service if it is not already answering on
#      localhost:11434. index.html talks to that service directly from the browser.
#   2. Serves this directory over http://localhost:8787 with Python's built-in
#      static file server. The server exists only to give the page a secure
#      origin - browsers refuse microphone access to a file:// page. It never
#      sees the conversation and never reaches the internet.
#   3. Opens the app in the default browser.
#
# Stop the app by pressing Ctrl+C here, or closing this terminal window.
#
# Related files:
#   index.html  - the whole app (UI, persona prompt, guardrail, memory, voice, Ollama call)
#   server.ps1  - the Windows counterpart of step 2

set -e
PORT=8787
MODEL_HOST="http://localhost:11434"

cd "$(dirname "$0")"

# Step 1 - make sure the local model service is up.
if ! curl -s -m 3 "$MODEL_HOST/api/tags" >/dev/null 2>&1; then
  echo "Starting Ollama..."
  open -a Ollama 2>/dev/null || ollama serve >/dev/null 2>&1 &
  for _ in $(seq 1 20); do
    curl -s -m 2 "$MODEL_HOST/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
fi

if ! curl -s -m 3 "$MODEL_HOST/api/tags" >/dev/null 2>&1; then
  echo "Could not reach Ollama at $MODEL_HOST."
  echo "Install it from ollama.com, then run: ollama pull llama3.2:1b"
  exit 1
fi

# The model name index.html defaults to. Pulled on first run so a fresh machine
# does not hit a 404 from the model service on the first message.
DEFAULT_MODEL="llama3.2:1b"
if ! ollama list 2>/dev/null | grep -q "^${DEFAULT_MODEL}"; then
  echo "Pulling $DEFAULT_MODEL (one time, ~1.3GB)..."
  ollama pull "$DEFAULT_MODEL"
fi

# Step 2 + 3 - serve this folder, then open the page once the port answers.
echo "Serving http://localhost:$PORT/ - press Ctrl+C to stop."
( for _ in $(seq 1 20); do
    curl -s -m 1 -o /dev/null "http://localhost:$PORT/index.html" && break
    sleep 0.5
  done
  open "http://localhost:$PORT/index.html" ) &

exec python3 -m http.server "$PORT" --bind 127.0.0.1
