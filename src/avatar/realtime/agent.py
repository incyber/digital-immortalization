"""The realtime agent.

One process per session. Joins a LiveKit room as a participant and runs the
conversation pipeline. Everything it uses is chosen by configuration, so the
same file runs against a CPU renderer on a laptop and a GPU renderer in cloud.

Pipeline order matters in two places:

  CrisisProcessor sits before the aggregator, so a crisis utterance never
  enters the model's context at all.

  RendererProcessor sits after TTS and before transport output, so the video
  track is derived from the audio actually being sent rather than produced in
  parallel with it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from avatar.config import Settings, get_settings
from avatar.marking.declare import declare
from avatar.marking.manifest import (
    ModelRef,
    SessionManifest,
    watermark_payload_for,
)
from avatar.persona import Persona, build_system_prompt, persona_from_avatar
from avatar.realtime.video_publisher import LiveKitVideoPublisher
from avatar.realtime.warmup import warm
from avatar.renderer.base import RendererStage
from avatar.renderer.plates import AvatarAssets, synthetic_assets
from avatar.renderer.processor import RendererProcessor
from avatar.renderer.viseme import VisemeRenderer
from avatar.safety.processor import CrisisProcessor
from avatar.services.llm_fallback import FallbackLLMService, build_providers
from avatar.services.speech import build_stt, build_tts
from avatar.vision.processor import VisionSampler
from avatar.vision.state import SceneState


def build_renderer(cfg: Settings, assets_path: str | None = None) -> RendererStage:
    """Pick a renderer backend.

    musetalk arrives in sub-project 2 and satisfies the same contract tests as
    the CPU backend, so nothing else in this file changes when it lands.
    """
    if cfg.renderer_backend == "musetalk":
        raise NotImplementedError(
            "the MuseTalk backend is sub-project 2; it requires an NVIDIA GPU "
            "and is not runnable on this machine"
        )

    if cfg.renderer_backend != "viseme":
        raise ValueError(f"unknown renderer_backend {cfg.renderer_backend!r}")

    if assets_path and Path(assets_path).exists():
        assets = AvatarAssets.load(Path(assets_path))
    else:
        # No source material yet. The generated stand-in keeps the call
        # runnable end to end, which is what lets the rest of the system be
        # developed before any customer has uploaded a clip.
        logger.warning(f"no assets at {assets_path!r}; using the generated stand-in avatar")
        assets = synthetic_assets(
            size=(cfg.video_width, cfg.video_height), fps=cfg.video_fps
        )
    return VisemeRenderer(assets)


async def load_persona(cfg: Settings, avatar_id: str) -> Persona:
    """Read the character from the database.

    The agent is a separate process, so it fetches rather than being handed a
    file path. Nothing about the character lives on disk with the application.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from avatar.gateway.db import init_engine
    from avatar.gateway.models import Avatar
    from avatar.safety.crisis_lines import parse_attested

    factory = async_sessionmaker(init_engine(cfg), expire_on_commit=False)
    async with factory() as db:
        avatar = await db.get(Avatar, avatar_id)
        if avatar is None:
            raise ValueError(f"no avatar {avatar_id}")
        return persona_from_avatar(avatar, parse_attested(cfg.crisis_lines_verified))


def build_pipeline(
    cfg: Settings,
    transport: LiveKitTransport,
    persona: Persona,
    stage: RendererStage,
    room_name: str = "unknown",
) -> tuple[Pipeline, PipelineTask, SceneState, dict]:
    """Assemble the conversation graph."""
    scene = SceneState()

    stt = build_stt(cfg)
    tts = build_tts(cfg)
    # Falls through to the configured backup provider when the primary runs
    # out of quota, rather than leaving somebody mid-conversation.
    llm = FallbackLLMService(build_providers(cfg))

    profile = persona.as_dict()
    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(persona, scene)}]
    )
    # Voice activity detection lives on the user aggregator in Pipecat 1.7,
    # not on transport params. LiveKitParams has no vad_analyzer field and
    # silently ignores one, which produces a call that hears nothing: audio
    # arrives, no turn is ever detected, and no transcription happens.
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    async def refresh_system_prompt(_text: str) -> None:
        """Rewrite the system message when the camera observation changes.

        The prompt is rebuilt rather than appended to, so a stale observation
        is replaced instead of accumulating alongside the current one.
        """
        messages = context.get_messages()
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = build_system_prompt(persona, scene)
            context.set_messages(messages)
            logger.info("scene observation injected into system prompt")

    pipeline = Pipeline(
        [
            transport.input(),
            VisionSampler(
                scene, cfg, on_observation=refresh_system_prompt, locale=persona.locale
            ),
            stt,
            CrisisProcessor(profile),
            aggregators.user(),
            llm,
            tts,
            RendererProcessor(stage, avatar_id=persona.avatar_id),
            # Publishes video directly: Pipecat's LiveKit transport implements
            # video input only. See video_publisher.py.
            # No watermark_payload here on purpose. WebRTC re-encodes every
            # frame and a watermark faint enough to be invisible does not
            # survive VP8 - measured against this pipeline, thirty consecutive
            # received frames decoded to nothing. Marking the live stream
            # happens out of band; see marking/declare.py. The pixel watermark
            # is for media this system encodes itself.
            LiveKitVideoPublisher(
                lambda: transport._client.room,
                width=cfg.video_width,
                height=cfg.video_height,
                fps=cfg.video_fps,
            ),
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    # Returned so the caller can warm them before the pipeline starts.
    services = {"stt": stt, "tts": tts, "llm": llm}
    return pipeline, task, scene, services


async def run_agent(
    room: str,
    token: str,
    avatar_id: str,
    assets_path: str | None,
    consent_record_id: str | None = None,
    rights_holder: str | None = None,
) -> None:
    cfg = get_settings()
    persona = await load_persona(cfg, avatar_id)
    stage = build_renderer(cfg, assets_path)

    transport = LiveKitTransport(
        url=cfg.livekit_url.replace("ws://", "ws://").replace("wss://", "wss://"),
        token=token,
        room_name=room,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            video_in_enabled=True,      # the camera the vision channel samples
            video_out_enabled=True,     # the rendered likeness
            video_out_is_live=True,
            video_out_width=cfg.video_width,
            video_out_height=cfg.video_height,
            video_out_framerate=cfg.video_fps,
            video_out_color_format="RGB",
        ),
    )

    _, task, _, services = build_pipeline(cfg, transport, persona, stage, room_name=room)

    @transport.event_handler("on_first_participant_joined")
    async def _on_join(transport, participant):
        logger.info(f"participant joined: {participant}")

    @transport.event_handler("on_participant_disconnected")
    async def _on_leave(transport, participant, reason=None):
        logger.info(f"participant left ({reason}); ending session")
        await task.cancel()

    # Load every model before the person speaks. They are still looking at a
    # connecting screen, so this time is free; inside the first turn it is not.
    await warm(cfg, stt=services["stt"], tts=services["tts"])

    manifest = SessionManifest(
        session_id=room,
        avatar_id=persona.avatar_id,
        avatar_display_name=persona.display_name,
        consent_record_id=consent_record_id or "unknown",
        rights_holder=rights_holder or "unknown",
        started_at=datetime.now(UTC).isoformat(),
        models=[
            ModelRef(name=cfg.stt_model, role="speech-to-text"),
            ModelRef(name=cfg.llm_model, role="language"),
            ModelRef(name=cfg.tts_voice, role="text-to-speech"),
            ModelRef(name=cfg.vlm_model, role="vision"),
            ModelRef(name=cfg.renderer_backend, role="renderer"),
        ],
        watermark_payload=watermark_payload_for(room).hex(),
    )

    logger.info(f"agent joining room {room}")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)

    # Declare the stream synthetic before any media flows. This travels on the
    # signalling channel, so unlike a pixel watermark it is not destroyed by
    # the video codec - see marking/declare.py.
    async def declare_when_connected() -> None:
        for _ in range(60):
            try:
                room_obj = transport._client.room
                await declare(room_obj.local_participant, manifest)
                return
            except Exception:  # noqa: BLE001 - retried below
                await asyncio.sleep(0.25)
        logger.error("never managed to declare this stream as synthetic")

    declare_task = asyncio.create_task(declare_when_connected())

    try:
        await runner.run()
    finally:
        declare_task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Join a room and run the avatar pipeline")
    parser.add_argument("--room", required=True)
    parser.add_argument("--token", default=os.environ.get("LIVEKIT_TOKEN", ""))
    parser.add_argument("--avatar", required=True, help="avatar id to embody")
    parser.add_argument("--assets", default=None)
    parser.add_argument("--consent-record", default=None)
    parser.add_argument("--rights-holder", default=None)
    args = parser.parse_args()

    if not args.token:
        print("a room token is required (--token or LIVEKIT_TOKEN)", file=sys.stderr)
        raise SystemExit(2)

    asyncio.run(
        run_agent(
            args.room,
            args.token,
            args.avatar,
            args.assets,
            consent_record_id=args.consent_record,
            rights_holder=args.rights_holder,
        )
    )


if __name__ == "__main__":
    main()
