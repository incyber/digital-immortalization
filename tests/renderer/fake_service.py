"""A stand-in for the GPU renderer service, so the client can be held to the
same contract as every other backend.

It imitates the one behaviour the client depends on and nothing else: a
continuous stream of JPEG frames at a fixed rate, sped up while audio is
arriving. No model, no weights, no GPU - the contract suite is about counts,
sizes and timing, and those are the parts that have to be right regardless of
what is drawing the pixels.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import cv2
import numpy as np
import websockets

FPS = 25
SAMPLE_RATE = 16000


def _jpeg(size: tuple[int, int], value: int) -> bytes:
    width, height = size
    image = np.full((height, width, 3), value, dtype=np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


class FakeRendererService:
    def __init__(self, size: tuple[int, int] = (128, 128)):
        self.size = size
        self.cancels = 0
        self._server = None

    @property
    def url(self) -> str:
        port = self._server.sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    async def start(self) -> FakeRendererService:
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        return self

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, socket) -> None:
        queued = asyncio.Queue()

        async def receive() -> None:
            async for message in socket:
                if isinstance(message, bytes):
                    # Frames for the audio's own duration, which is what the
                    # real service produces and what the client counts on.
                    seconds = len(message) / 2 / SAMPLE_RATE
                    for _ in range(round(seconds * FPS)):
                        await queued.put(_jpeg(self.size, 200))
                elif json.loads(message).get("op") == "cancel":
                    self.cancels += 1
                    while not queued.empty():
                        queued.get_nowait()

        async def send() -> None:
            while True:
                try:
                    frame = queued.get_nowait()
                except asyncio.QueueEmpty:
                    frame = _jpeg(self.size, 60)
                await socket.send(frame)
                await asyncio.sleep(1.0 / FPS)

        receiver = asyncio.create_task(receive())
        sender = asyncio.create_task(send())
        try:
            await receiver
        except websockets.ConnectionClosed:
            pass
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
