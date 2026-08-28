"""The MuseTalk renderer, seen from the call.

The GPU service publishes one continuous stream of frames - rendered while
there is speech, prepared cycle frames while there is not - so this class does
not choose between two sources. It keeps one socket open, one reader filling a
queue, and both `idle()` and `render()` draw from that same queue.

That is what keeps a call looking like a call. The alternative, a renderer that
returns frames only when speaking, forces the caller to splice two video
sources together at exactly the moment a person starts talking, which is the
moment nobody is looking anywhere else.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import cv2
import numpy as np
import websockets
from loguru import logger

from avatar.renderer.base import AudioChunk, VideoFrame

# What the service emits. Fixed rather than negotiated: the service paces its
# own output, and a client that guessed differently would drift.
FPS = 25
SAMPLE_RATE = 16000

# Frames buffered locally. Two seconds is enough to ride out a slow round trip
# and short enough that a cancel does not leave stale mouth shapes on screen.
MAX_BUFFERED = FPS * 2

CONNECT_TIMEOUT_S = 30


class RendererUnavailable(RuntimeError):
    pass


class MuseTalkRenderer:
    """RendererStage over the live GPU service."""

    def __init__(self, base_url: str, avatar_id: str = "", size: tuple[int, int] = (512, 512)):
        # ws:// or wss://, the service root. http:// is accepted and rewritten
        # because that is what everything else in the configuration uses.
        self._base = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self._avatar_id = avatar_id
        self._size = size
        self._socket = None
        self._frames: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=MAX_BUFFERED)
        self._reader: asyncio.Task | None = None
        # Bumped by cancel(). A render generator captures it when it starts and
        # stops the moment it changes, which is what makes cancel abandon
        # frames already promised rather than merely stop new ones.
        self._generation = 0

    @property
    def fps(self) -> int:
        return FPS

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    # ------------------------------------------------------------------
    async def prepare(self, avatar_id: str) -> None:
        """Open the stream for this avatar. Once per session, before any render."""
        self._avatar_id = avatar_id
        url = f"{self._base}/avatars/{avatar_id}/stream"

        try:
            self._socket = await asyncio.wait_for(
                websockets.connect(url, max_size=None), timeout=CONNECT_TIMEOUT_S
            )
        except Exception as exc:
            raise RendererUnavailable(f"could not reach the renderer at {url}: {exc}") from exc

        self._reader = asyncio.create_task(self._read())
        logger.info(f"renderer stream open for {avatar_id}")

    async def _read(self) -> None:
        """Decode frames as they arrive, dropping the oldest when behind.

        Dropping is deliberate. A call that falls behind should lose a frame
        and stay in sync with the audio; holding everything would grow a
        permanent gap between what is heard and what is seen.
        """
        try:
            async for message in self._socket:
                if not isinstance(message, bytes):
                    continue
                image = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                frame = VideoFrame(
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes(),
                    width=image.shape[1],
                    height=image.shape[0],
                )
                if self._frames.full():
                    self._frames.get_nowait()
                self._frames.put_nowait(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"renderer stream ended: {exc}")

    # ------------------------------------------------------------------
    async def idle(self) -> AsyncIterator[VideoFrame]:
        """Frames while nothing is being said. Endless by contract."""
        while True:
            yield await self._frames.get()

    async def render(self, audio: AudioChunk) -> AsyncIterator[VideoFrame]:
        """Send one chunk of speech and yield the frames it produces.

        The count is derived from the audio's own duration rather than from
        anything the service reports, so a dropped frame cannot desynchronise
        the caller's idea of when this utterance ends.
        """
        if self._socket is None:
            raise RendererUnavailable("prepare() has not been called")

        generation = self._generation
        await self._socket.send(audio.pcm)
        await self._socket.send(json.dumps({"op": "flush"}))

        for _ in range(max(1, round(audio.duration_s * FPS))):
            if self._generation != generation:
                return
            yield await self._frames.get()

    async def cancel(self) -> None:
        """Barge-in. Drops queued frames here and at the service.

        Local first. The socket round trip is what makes this slow, and the
        frames already on this side are the ones about to be shown.
        """
        self._generation += 1
        while not self._frames.empty():
            self._frames.get_nowait()

        if self._socket is not None:
            try:
                await asyncio.wait_for(
                    self._socket.send(json.dumps({"op": "cancel"})), timeout=0.05
                )
            except Exception as exc:  # noqa: BLE001 - includes the timeout
                # The 100ms contract matters more than the service hearing it.
                # Stale frames on the far side are dropped as they arrive.
                logger.warning(f"cancel not delivered: {exc}")

    async def aclose(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
        if self._socket is not None:
            await self._socket.close()
