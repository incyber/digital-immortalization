"""Piper speech synthesis as an HTTP service.

This file and this container are GPL-3.0-or-later, the same licence as the
piper-tts library it imports. That is the entire point of it existing: the
application that calls this service does not import Piper, does not link
against it, and communicates only over HTTP. Aggregation, not linking.

Exposes the request shape Pipecat's PiperHttpTTSService sends:

    POST /  {"text": "...", "voice": "es_ES-davefx-medium"}  ->  audio/wav

SPDX-License-Identifier: GPL-3.0-or-later
"""

import io
import os
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from json import loads
from pathlib import Path

from piper import PiperVoice

VOICES_DIR = Path(os.environ.get("PIPER_VOICES_DIR", "/voices"))
DEFAULT_VOICE = os.environ.get("PIPER_VOICE", "es_ES-davefx-medium")
PORT = int(os.environ.get("PIPER_PORT", "5050"))

# Voices are loaded once and kept. Loading is the expensive part; synthesis of
# a sentence is tens of milliseconds.
_loaded: dict[str, PiperVoice] = {}


def voice(name: str) -> PiperVoice:
    if name not in _loaded:
        _loaded[name] = PiperVoice.load(VOICES_DIR / f"{name}.onnx")
    return _loaded[name]


def synthesize(text: str, name: str) -> bytes:
    v = voice(name)
    pcm = b"".join(chunk.audio_int16_bytes for chunk in v.synthesize(text))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(v.config.sample_rate)
        out.writeframes(pcm)
    return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path == "/health":
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("content-length", 0))
            body = loads(self.rfile.read(length) or b"{}")
            audio = synthesize(body.get("text", ""), body.get("voice") or DEFAULT_VOICE)
        except Exception as exc:  # noqa: BLE001
            self.send_error(500, str(exc))
            return

        self.send_response(200)
        self.send_header("content-type", "audio/wav")
        self.send_header("content-length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def log_message(self, *args):
        pass  # quiet; the application logs its own timings


if __name__ == "__main__":
    print(f"piper http on :{PORT}, voices from {VOICES_DIR}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
