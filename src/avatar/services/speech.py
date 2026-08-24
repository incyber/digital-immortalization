"""Speech services: which STT and TTS backend this process uses.

Both are chosen by configuration, not by import, so the same pipeline code
runs on a laptop and on a GPU server. See the execution matrix in the design
document.

Neither service is written here. Pipecat 1.7.0 already ships local, in-process
implementations of both, and reimplementing them would mean maintaining a copy
of somebody else's streaming and metrics plumbing for no gain.
"""

from __future__ import annotations

from pathlib import Path

from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from avatar.config import Settings


def build_stt(cfg: Settings) -> STTService:
    """Speech to text.

    mlx  - Whisper through MLX, which runs on the Metal GPU. The local default.
    faster - CTranslate2 Whisper. CPU or CUDA. The cloud default.
    """
    if cfg.stt_backend == "mlx":
        from pipecat.services.whisper.stt import WhisperSTTServiceMLX

        return WhisperSTTServiceMLX(model=cfg.stt_model)

    if cfg.stt_backend == "faster":
        from pipecat.services.whisper.stt import WhisperSTTService

        return WhisperSTTService(model=cfg.stt_model)

    raise ValueError(f"unknown stt_backend {cfg.stt_backend!r}; expected 'mlx' or 'faster'")


def build_tts(cfg: Settings) -> TTSService:
    """Text to speech.

    Piper for now: small, fast on CPU, MIT, and cross-platform, so local and
    cloud sound identical. Chatterbox replaces it in sub-project 3, when the
    voice becomes a clone of a specific person rather than a stock voice. Both
    satisfy Pipecat's TTSService contract, so that swap touches this function
    and nothing else.
    """
    return PiperTTSService(
        settings=PiperTTSService.Settings(voice=cfg.tts_voice),
        download_dir=Path(cfg.voices_dir),
    )
