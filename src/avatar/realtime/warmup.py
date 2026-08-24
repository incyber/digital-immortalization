"""Load every model before the first person speaks.

Whisper, the language model and Piper all load lazily on first use. Measured
on an M3 Max, that put roughly a second of loading inside the first turn:

    import mlx_whisper      1791 ms   (once per process)
    first transcribe         868 ms
    subsequent transcribes    72 ms

Ollama behaves the same way - seconds cold, ~150 ms warm. So the first turn of
every call was paying for model loading, and because the latency test measured
exactly one turn, that cost was being read as the steady-state cost of speech
recognition. It is not.

This runs after the agent joins the room and before the pipeline starts, which
is dead time anyway: the person is still looking at a connecting screen.

Nothing here can fail the session. A warm-up failure costs latency on the first
turn; failing to start the call costs the call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx
from loguru import logger

from avatar.config import Settings

# One second of silence at the rate the pipeline runs speech recognition at.
# Silence is enough: the cost being removed is loading weights and compiling
# kernels, neither of which depends on the content.
_WARMUP_SAMPLE_RATE = 16000
_WARMUP_SILENCE = b"\x00\x00" * _WARMUP_SAMPLE_RATE

# Short and unambiguous, so a voice with a slow first synthesis still returns
# promptly.
_WARMUP_TEXT = "Hola."


@dataclass
class WarmupReport:
    """What was warmed, how long it took, and what failed."""

    elapsed_ms: dict[str, float] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"{name} {ms:.0f}ms" for name, ms in sorted(self.elapsed_ms.items())]
        if self.failures:
            parts += [f"{name} FAILED ({err})" for name, err in sorted(self.failures.items())]
        return ", ".join(parts)


async def _timed(report: WarmupReport, name: str, coro: Callable[[], Awaitable[None]]) -> None:
    loop = asyncio.get_running_loop()
    start = loop.time()
    try:
        await coro()
    except Exception as exc:  # noqa: BLE001 - every failure is handled the same way
        report.failures[name] = str(exc)
        logger.warning(f"warm-up for {name} failed; first turn will be slower: {exc}")
    finally:
        report.elapsed_ms[name] = (loop.time() - start) * 1000


async def default_llm_ping(cfg: Settings) -> None:
    """Force the language model resident with the smallest possible request.

    One token, because the purpose is to make the server load weights, not to
    get an answer.
    """
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{cfg.llm_base_url.rstrip('/')}/chat/completions",
            headers={"authorization": f"Bearer {cfg.llm_api_key}"},
            json={
                "model": cfg.llm_model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "stream": False,
            },
        )
        response.raise_for_status()


async def warm(
    cfg: Settings,
    stt,
    tts,
    llm_ping: Callable[[], Awaitable[None]] | None = None,
) -> WarmupReport:
    """Exercise every model once. Never raises.

    Concurrent rather than sequential: this sits between the agent joining and
    the person speaking, so its wall time is real.
    """
    report = WarmupReport()

    async def warm_stt() -> None:
        async for _ in stt.run_stt(_WARMUP_SILENCE):
            pass

    async def warm_tts() -> None:
        async for _ in tts.run_tts(_WARMUP_TEXT, "warmup"):
            pass

    async def warm_llm() -> None:
        if llm_ping is not None:
            await llm_ping()
        else:
            await default_llm_ping(cfg)

    await asyncio.gather(
        _timed(report, "stt", warm_stt),
        _timed(report, "tts", warm_tts),
        _timed(report, "llm", warm_llm),
    )

    logger.info(f"warm-up complete: {report.summary()}")
    return report
