# Running this project on macOS

The launchers committed in `web/` and `mvp/` are Windows-only (`.bat` / `.ps1`).
This file covers the macOS equivalents. Nothing in the original prototypes was
changed - the two files below sit alongside the Windows ones.

| Added file | Replaces | Prototype |
|---|---|---|
| `web/start-web.command` | `web/Start-Web.bat` + `web/server.ps1` | Voice app, local model |
| `vercel-app/dev-server.mjs` | `vercel dev` | Web app, hosted model |

---

## `web/` - voice prototype, runs fully offline

The one that works with no API key and no account.

**Requirements:** Ollama (`brew install --cask ollama` or ollama.com), Python 3
(ships with macOS via Xcode command line tools).

```bash
cd web
./start-web.command
```

The script starts Ollama if it isn't running, pulls `llama3.2:1b` on first run,
serves the folder at <http://localhost:8787/>, and opens the browser. Ctrl+C stops it.

Notes:
- Use **Chrome or Edge** - voice input needs the Web Speech API, which Safari
  does not implement. The typed-input box works in any browser.
- The model name in Settings must match a model that is actually pulled.
  `llama3.2:1b` is the default in `index.html`; `llama3.2:3b` is better on
  historical detail and roughly twice as slow.
- The local server exists only so the browser will grant microphone access -
  a `file://` page is not a secure origin.

---

## `vercel-app/` - web prototype, hosted model

Same UI, but replies come from Gemini through the serverless function in
`api/chat.js` instead of a local model.

**Requirements:** Node 20+ and a Google AI Studio API key.

```bash
cd vercel-app
echo 'GEMINI_API_KEY=your-key-here' > .env.local
node dev-server.mjs                       # http://localhost:3000
PORT=3001 node dev-server.mjs             # if 3000 is taken
```

`dev-server.mjs` serves `public/` and routes `POST /api/chat` to the exported
handler in `api/chat.js` - the same file that runs on the Edge runtime in
production, unmodified. It exists so the app can run locally without linking the
folder to a Vercel project; production never reads it.

`.env.local` is already covered by `.gitignore` (`vercel-app/.env*`).

Without a key, the page loads and `/api/chat` returns:
`{"error":"Server misconfigured: GEMINI_API_KEY is not set"}`.

If Gemini rejects the model name, override it without touching `api/chat.js`:

```bash
echo 'GEMINI_MODEL=gemini-2.5-flash' >> .env.local
```

---

## `mvp/` - text chat prototype

Not runnable as-is on macOS: `Start-Chat.ps1` is a PowerShell script and
`Start-Chat.bat` is a Windows batch file. Two options if it's needed:

1. Install PowerShell (`brew install --cask powershell`), then
   `export ANTHROPIC_API_KEY=...` and `pwsh ./Start-Chat.ps1`. The script may
   still need edits - it was written against Windows PowerShell, not the
   cross-platform build.
2. Port the loop to Node. It is a small script: render `prompt-v2.txt` with
   `profile.json`, check `crisis-keywords.json` before each turn, call the
   Messages API, append to `memory.jsonl`.

The `web/` prototype already exercises the same persona prompt and the same
keyword guardrail, so this is only worth doing if the CLI flow specifically matters.
