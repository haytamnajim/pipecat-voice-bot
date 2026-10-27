import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
    ),
    "livekit": lambda: LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Démarrage du bot Groq + Kokoro")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY non trouvée dans .env")
        return

    stt = GroqSTTService(
        api_key=groq_api_key,
        settings=GroqSTTService.Settings(model="whisper-large-v3-turbo", language="fr"),
    )

    llm = GroqLLMService(
        api_key=groq_api_key,
        settings=GroqLLMService.Settings(
            model="openai/gpt-oss-20b",
            system_instruction="Tu es un assistant vocal amical et réactif. Tes réponses seront lues à voix haute : réponds toujours en français de façon naturelle, concise et vivante en 1 ou 2 phrases courtes maximum. N'utilise pas d'émojis ni de puces.",
        ),
    )

    logger.info("Chargement de la voix Kokoro TTS...")
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice="ff_siwis",  # Voix française féminine native
            language="fr-fr",
        )
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
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
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connecté ! Envoi du message d'accueil...")
        context.add_message(
            {"role": "developer", "content": "Dis bonjour et demande en quoi tu peux aider."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client déconnecté")
        await runner.cancel()

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message: UserTurnMessageAddedMessage):
        logger.info(f"🎤 Utilisateur a dit : {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        logger.info(f"🤖 Réponse IA : {message.content}")

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
