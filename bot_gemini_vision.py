import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnMessageAddedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams, VideoInSourceParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_in_sources={
            "camera": VideoInSourceParams(framerate=1),
            "screenVideo": VideoInSourceParams(framerate=1),
        },
    ),
    "livekit": lambda: LiveKitParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_in_sources={
            "camera": VideoInSourceParams(framerate=1),
            "screenVideo": VideoInSourceParams(framerate=1),
        },
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_in_sources={
            "camera": VideoInSourceParams(framerate=1),
            "screenVideo": VideoInSourceParams(framerate=1),
        },
    ),
}

SYSTEM_INSTRUCTION = """
Tu es un assistant vidéo et vocal intelligent, amical et très réactif.
Tu vois la caméra en direct de l'utilisateur et tu l'entends.
Tes réponses seront prononcées à haute voix : réponds toujours en français de manière naturelle, concise et vivante en 1 ou 2 phrases courtes.
N'hésite pas à commenter ou décrire ce que tu vois quand l'utilisateur te le demande ou te montre quelque chose.
Évite les puces, listes ou émoticônes.
"""


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Démarrage du bot Gemini Live Vision + Audio...")

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("Aucune clé GOOGLE_API_KEY ou GEMINI_API_KEY trouvée dans .env !")
        return

    llm = GeminiLiveLLMService(
        api_key=api_key,
        settings=GeminiLiveLLMService.Settings(
            system_instruction=SYSTEM_INSTRUCTION,
            voice="Aoede",  # Options: Aoede, Charon, Fenrir, Kore, Puck
        ),
    )

    context = LLMContext(
        [
            {
                "role": "developer",
                "content": "Salue l'utilisateur chaleureusement en français et dis-lui que tu le vois et l'entends très bien.",
            },
        ]
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connecté : {client}")
        await worker.queue_frames([LLMRunFrame()])
        await asyncio.sleep(2)
        llm.set_audio_input_paused(False)
        llm.set_video_input_paused(False)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client déconnecté")
        await runner.cancel()

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message: UserTurnMessageAddedMessage):
        logger.info(f"🎤 Utilisateur : {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        logger.info(f"🤖 IA : {message.content}")

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
