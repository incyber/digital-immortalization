# DI web prototype (voice, local, free)

A local, single-page web app for the DI avatar. No PowerShell, no terminal, no API key, no cost — for the person actually talking to it. Runs entirely on this machine via **Ollama**.

## One-time setup (do this once)

1. **Install Ollama**: download the Windows installer from ollama.com and run it (normal app install, no account needed). It runs quietly in the background afterward — you won't need to open it again.
2. **Pull a small model** — open any terminal once and run:
   ```
   ollama pull llama3.2:3b
   ```
   This machine has 16GB RAM and a weak GPU (2GB VRAM), so a small model (3B or under) is the right choice — it'll run on CPU and stay reasonably responsive. If replies feel slow, try an even smaller one: `ollama pull llama3.2:1b`, then set the model name in the app's Settings to match.

That's the only terminal step, ever. Everything after this is double-click and talk.

## Every time after that

Double-click **`Start-Web.bat`**. It quietly starts a tiny local server and opens your browser straight to the app.

Tap the mic, talk. It listens, thinks, replies out loud, then automatically starts listening again — like a real call. Tap the mic again to end it.

## Why there's a local server at all, if it's "just a web page"

Browsers only allow microphone access from a secure address (`https://` or `localhost`), not a bare double-clicked file. `server.ps1` serves the page at `http://localhost:8787/` to satisfy that. It only ever serves that one file — it never sees your conversation, never talks to the internet. Everything (voice, memory, the model itself) stays on this computer.

## Settings

- **Ollama model** — must match what you pulled (default `llama3.2:3b`). The panel shows live whether Ollama is detected and which models are available.
- **Voice language** — for both speech recognition and the spoken reply.
- **Clear conversation** — wipes the local memory log.

## What's actually implemented

- Voice in (Web Speech API `SpeechRecognition` — Chrome/Edge; falls back to the text box if unsupported) and voice out (`speechSynthesis`).
- Continuous hands-free conversation: one tap starts a call, it keeps listening/replying until you tap again.
- The same `prompt-v2.txt` persona logic as the earlier prototypes, rendered with an embedded demo profile, sent to Ollama as the system message.
- The same keyword-based safety guardrail, checked **before** any message reaches the model — on a match it shows the fixed safety message, speaks it, and skips the model call entirely.
- A visible disclosure banner at the top of the page, always present before any conversation starts.
- Memory: last ~12 turns, kept in the browser's local storage, fed back in as `[HISTORICAL_CONTEXT]`.

## Before using this with a real person's data

Edit the `PROFILE` object near the top of the `<script>` in `index.html` — it currently has fictional demo data ("James Whitfield"), on purpose. Also set a real `crisis_line_name` / `crisis_line_number` there; it currently has placeholders.

## Known limitations

- **Model quality is lower than a frontier hosted model** (Claude, GPT, Gemini) — that's the trade for zero cost and zero limits. A 3B local model can hold a warm, in-character conversation but will be less sharp, especially on nuance or long context.
- **Speed depends entirely on this machine.** On this hardware (CPU-bound, weak GPU), expect a few seconds of "Thinking..." per reply with a 3B model, more with anything bigger.
- The crisis guardrail is a keyword list, not a trained classifier — a stand-in for the real thing, not a final safety layer.
- No real consent tracking or biometric detection — text/voice only, one browser, one person.
- Closing the `server.ps1` console window (visible minimized in the taskbar) stops the app.

## If you ever want higher quality later

Nothing here rules out switching to a hosted model later (Anthropic, Google's free-tier Gemini, etc.) if quality matters more than zero-cost at some point — that would mean swapping `callOllama()` in `index.html` for a hosted API call again. Not needed now; this version is fully self-contained.

## Files

| File | Purpose |
|---|---|
| `Start-Web.bat` | Double-click this — starts the server and opens the browser |
| `index.html` | The entire app: UI, prompt, profile, guardrail, memory, voice, Ollama call |
| `server.ps1` | Tiny local static file server (secure context for the mic) |
