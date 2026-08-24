"""Warm-up must be thorough and must never be able to break a call.

Every model in the pipeline loads lazily on first use, which put roughly a
second of model loading inside the first turn a person experiences. These
tests pin both halves: that each service is actually exercised, and that a
warm-up failure costs latency rather than the session.
"""

import asyncio

import pytest

from avatar.config import Settings
from avatar.realtime.warmup import WarmupReport, warm


class FakeSTT:
    def __init__(self, fail=False):
        self.calls = 0
        self._fail = fail

    async def run_stt(self, audio: bytes):
        self.calls += 1
        if self._fail:
            raise RuntimeError("no model")
        yield None


class FakeTTS:
    def __init__(self, fail=False):
        self.calls = 0
        self._fail = fail

    async def run_tts(self, text: str, context_id: str):
        self.calls += 1
        if self._fail:
            raise RuntimeError("no voice")
        yield None


class FakeLLM:
    def __init__(self, fail=False):
        self.calls = 0
        self._fail = fail

    async def ping(self):
        self.calls += 1
        if self._fail:
            raise RuntimeError("no server")


@pytest.fixture
def cfg():
    return Settings(_env_file=None)


async def test_every_service_is_exercised(cfg):
    stt, tts, llm = FakeSTT(), FakeTTS(), FakeLLM()
    report = await warm(cfg, stt=stt, tts=tts, llm_ping=llm.ping)
    assert stt.calls == 1
    assert tts.calls == 1
    assert llm.calls == 1
    assert isinstance(report, WarmupReport)


async def test_a_failing_service_does_not_prevent_a_call(cfg):
    # A warm-up failure costs latency on the first turn. It must never be the
    # reason a person cannot start a conversation.
    report = await warm(
        cfg, stt=FakeSTT(fail=True), tts=FakeTTS(), llm_ping=FakeLLM().ping
    )
    assert report.failures, "the failure should be recorded"
    assert "stt" in report.failures


async def test_all_services_failing_still_returns(cfg):
    report = await warm(
        cfg,
        stt=FakeSTT(fail=True),
        tts=FakeTTS(fail=True),
        llm_ping=FakeLLM(fail=True).ping,
    )
    assert set(report.failures) == {"stt", "tts", "llm"}


async def test_report_records_timings(cfg):
    report = await warm(cfg, stt=FakeSTT(), tts=FakeTTS(), llm_ping=FakeLLM().ping)
    assert set(report.elapsed_ms) == {"stt", "tts", "llm"}
    assert all(v >= 0 for v in report.elapsed_ms.values())


async def test_services_are_warmed_concurrently(cfg):
    # Serially this would be three sleeps; concurrently it is one. Warm-up sits
    # between the agent joining and the person speaking, so its wall time is
    # real and worth keeping short.
    class Slow(FakeSTT):
        async def run_stt(self, audio: bytes):
            await asyncio.sleep(0.15)
            yield None

    class SlowTTS(FakeTTS):
        async def run_tts(self, text: str, context_id: str):
            await asyncio.sleep(0.15)
            yield None

    async def slow_ping():
        await asyncio.sleep(0.15)

    loop = asyncio.get_running_loop()
    start = loop.time()
    await warm(cfg, stt=Slow(), tts=SlowTTS(), llm_ping=slow_ping)
    assert (loop.time() - start) < 0.35
