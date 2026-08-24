# DI prototype (text chat MVP)

A real, runnable chat prototype for the "digital immortalization" avatar project. No Python or Node required — it runs on Windows PowerShell, calling the Anthropic API directly.

## What it actually does

- Renders `../prompt-v2.txt` with the data in `profile.json` into a real system prompt (no leftover placeholders).
- Runs a chat loop against the Anthropic Messages API.
- Checks every message against `crisis-keywords.json` **before** it reaches the model. On a match, it skips the persona entirely, logs the event to `crisis-log.jsonl`, and shows a fixed safety message with the crisis line from `profile.json` — this does not depend on the model deciding to break character.
- Keeps a local `memory.jsonl` log and feeds the last few turns back in as `[HISTORICAL_CONTEXT]`, the way `prompt-v2.txt` expects.

## Run it

Windows won't run `.ps1` files by double-clicking them (they open in an editor, or the console flashes and closes instantly) — use one of these instead:

**Option A — from PowerShell (recommended for a first run, so you can see errors):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
cd C:\Users\2112\Dev\DI\mvp
.\Start-Chat.ps1
```

**Option B — double-click `Start-Chat.bat`.** This works even from Explorer, but only if the API key is set *permanently*, not just for one PowerShell session:
```powershell
setx ANTHROPIC_API_KEY "sk-ant-..."
```
Run that once in any PowerShell window, then close and reopen any terminal (or just double-click `Start-Chat.bat` — new processes pick up `setx` variables automatically). The `.bat` also keeps the window open with `pause` at the end so errors are actually readable instead of flashing by.

Type `exit` to end the call.

## Before using this with a real person's data

1. Replace everything in `profile.json` — it currently ships with a fictional example ("Mateo Duarte"), on purpose. Only fill it in with data you have documented consent to use, per the P0 items in the audit report.
2. Replace `crisis_line_name` / `crisis_line_number` with a real local crisis line. The script warns on startup if you haven't.
3. Read the limitations below — this is a prototype for testing the prompt and the guardrail mechanism, not a product.

## Known limitations (by design, for an MVP)

- **Memory** is "last 6 turns from a flat file," not real retrieval. Fine for a short demo call; won't hold up over months of history.
- **Crisis guardrail** is a keyword list, not a trained classifier — it will miss indirect distress and can false-positive on ordinary phrases like "I want to be with you forever" said fondly. It exists to prove the mechanism (deterministic check outside the LLM), not to be the final safety layer.
- **No voice/video/biometric detection.** Text only. The multimodal protocol in the prompt isn't exercised by this script.
- **No consent enforcement.** The prompt asks the model to decline if consent is missing, but nothing here actually checks or stores consent records — that has to live in whatever system sits above this prototype.

## Files

| File | Purpose |
|---|---|
| `Start-Chat.ps1` | Main script — render prompt, chat loop, guardrail, memory |
| `profile.json` | The legend/legacy person's data — fictional example, replace it |
| `crisis-keywords.json` | Keyword list for the safety guardrail (ES/EN) |
| `memory.jsonl` | Created on first run — conversation log used as memory |
| `crisis-log.jsonl` | Created if the guardrail ever triggers |
