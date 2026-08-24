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
from pathlib import Path

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from avatar.config import Settings, get_settings
from avatar.persona import build_system_prompt, load_profile
from avatar.renderer.base import RendererStage
from avatar.renderer.plates import AvatarAssets, synthetic_assets
from avatar.renderer.processor import RendererProcessor
from avatar.renderer.viseme import VisemeRenderer
from avatar.safety.processor import CrisisProcessor
from avatar.services.speech import build_stt, build_tts
from avatar.vision.processor import VisionSampler
from avatar.realtime.video_publisher import LiveKitVideoPublisher
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


def build_pipeline(
    cfg: Settings,
    transport: LiveKitTransport,
    profile: dict,
    stage: RendererStage,
) -> tuple[Pipeline, PipelineTask, SceneState]:
    """Assemble the conversation graph."""
    scene = SceneState()

    stt = build_stt(cfg)
    tts = build_tts(cfg)
    llm = OpenAILLMService(
        api_key=cfg.llm_api_key, base_url=cfg.llm_base_url, model=cfg.llm_model
    )

    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(profile, scene)}]
    )
    aggregators = LLMContextAggregatorPair(context)

    async def refresh_system_prompt(_text: str) -> None:
        """Rewrite the system message when the camera observation changes.

        The prompt is rebuilt rather than appended to, so a stale observation
        is replaced instead of accumulating alongside the current one.
        """
        messages = context.get_messages()
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = build_system_prompt(profile, scene)
            context.set_messages(messages)

    pipeline = Pipeline(
        [
            transport.input(),
            VisionSampler(scene, cfg, on_observation=refresh_system_prompt),
            stt,
            CrisisProcessor(profile),
            aggregators.user(),
            llm,
            tts,
            RendererProcessor(stage, avatar_id=profile["id"]),
            # Publishes video directly: Pipecat's LiveKit transport implements
            # video input only. See video_publisher.py.
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
    return pipeline, task, scene


async def run_agent(room: str, token: str, profile_path: str, assets_path: str | None) -> None:
    cfg = get_settings()
    profile = load_profile(profile_path)
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
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    _, task, _ = build_pipeline(cfg, transport, profile, stage)

    @transport.event_handler("on_first_participant_joined")
    async def _on_join(transport, participant):
        logger.info(f"participant joined: {participant}")

    @transport.event_handler("on_participant_disconnected")
    async def _on_leave(transport, participant, reason=None):
        logger.info(f"participant left ({reason}); ending session")
        await task.cancel()

    logger.info(f"agent joining room {room}")
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    await runner.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Join a room and run the avatar pipeline")
    parser.add_argument("--room", required=True)
    parser.add_argument("--token", default=os.environ.get("LIVEKIT_TOKEN", ""))
    parser.add_argument("--profile", default="src/avatar/profiles/colon.json")
    parser.add_argument("--assets", default=None)
    args = parser.parse_args()

    if not args.token:
        print("a room token is required (--token or LIVEKIT_TOKEN)", file=sys.stderr)
        raise SystemExit(2)

    asyncio.run(run_agent(args.room, args.token, args.profile, args.assets))


if __name__ == "__main__":
    main()
