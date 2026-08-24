"""Publishes the rendered avatar as a LiveKit video track.

This exists because Pipecat 1.7.0's LiveKit transport does not implement video
output: LiveKitOutputTransport.write_video_frame returns False unconditionally,
and register_video_destination is a no-op. Video input works; video output was
never wired up. Verified by reading the installed source, not inferred from a
missing track.

So the renderer's frames are published directly through the LiveKit SDK. The
split is deliberate: RendererProcessor stays transport-agnostic and emits
OutputImageRawFrame, this processor is the single place that knows LiveKit
exists, and swapping transports later touches one file.
"""

from __future__ import annotations

from collections.abc import Callable

from livekit import rtc
from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, Frame, OutputImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# LiveKit takes RGBA; the renderer produces RGB24. One alpha byte per pixel is
# appended per frame rather than having the renderer carry an alpha channel it
# has no use for.
_RGBA = rtc.VideoBufferType.RGBA


class _NotConnectedYet(Exception):
    """The room exists but has not finished connecting."""


class LiveKitVideoPublisher(FrameProcessor):
    """Turns OutputImageRawFrame into a published video track."""

    def __init__(
        self,
        room_provider: Callable[[], rtc.Room],
        width: int,
        height: int,
        fps: int,
        track_name: str = "avatar",
    ):
        super().__init__()
        # A callable, not a room: the transport has no room object until it has
        # connected, and the pipeline is assembled before that happens.
        self._room_provider = room_provider
        self._width = width
        self._height = height
        self._fps = fps
        self._track_name = track_name
        self._source: rtc.VideoSource | None = None
        self._published = False

    async def _ensure_track(self) -> None:
        """Create and publish the track once, on the first frame.

        Publishing lazily rather than at start means the track appears when
        there is something to show, and a renderer that never produces a frame
        surfaces as a missing track rather than a black one.
        """
        if self._published:
            return

        try:
            room = self._room_provider()
            participant = room.local_participant
        except Exception as exc:
            raise _NotConnectedYet(str(exc)) from exc

        self._source = rtc.VideoSource(self._width, self._height)
        track = rtc.LocalVideoTrack.create_video_track(self._track_name, self._source)
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            video_encoding=rtc.VideoEncoding(
                max_framerate=self._fps,
                max_bitrate=1_500_000,
            ),
        )
        await participant.publish_track(track, options)
        self._published = True
        logger.info(
            f"published avatar video track {self._width}x{self._height} @ {self._fps}fps"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputImageRawFrame):
            try:
                await self._capture(frame)
            except _NotConnectedYet:
                # The renderer starts publishing idle frames as soon as the
                # pipeline starts, which is a moment before the room finishes
                # connecting. Dropping those first few frames is correct; there
                # is nobody to send them to yet.
                pass
            except Exception as exc:  # noqa: BLE001
                # Video is the degradable half of the call. Losing it must not
                # take the audio with it.
                logger.error(f"could not publish video frame: {exc}")
            # Not forwarded: the frame has been consumed. Passing it on would
            # reach a transport whose write_video_frame discards it anyway.
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._published = False
            self._source = None

        await self.push_frame(frame, direction)

    async def _capture(self, frame: OutputImageRawFrame) -> None:
        await self._ensure_track()
        assert self._source is not None

        width, height = frame.size
        rgba = bytearray(width * height * 4)
        rgba[0::4] = frame.image[0::3]
        rgba[1::4] = frame.image[1::3]
        rgba[2::4] = frame.image[2::3]
        rgba[3::4] = b"\xff" * (width * height)

        self._source.capture_frame(rtc.VideoFrame(width, height, _RGBA, bytes(rgba)))
