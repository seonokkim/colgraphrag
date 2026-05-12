"""
ColGraphRAG Multi-Dataset Demo Backend Server.

A read-only viewer for WebQA and MultimodalQA pipeline results.

Usage:
    cd demo/be
    python server.py              # 127.0.0.1:8000
    # or
    uvicorn server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

_BE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BE_DIR))

from services.logging_setup import configure_demo_file_logging

_SESSION_LOG = configure_demo_file_logging(_BE_DIR)
print(f"[demo-be] Session log: {_SESSION_LOG}", flush=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.resolve_demo_paths import (
    demo_paths_as_dict,
    load_all_demo_paths,
    DemoPaths,
    MmqaDemoPaths,
    MultiDatasetPaths,
)
from routers.health_router import router as health_router, set_run_id
from routers.run_router import router as run_router
from routers.questions_router import router as questions_router
from routers.graph_router import router as graph_router
from routers.images_router import router as images_router
from routers.chat_router import router as chat_router
from routers.datasets_router import router as datasets_router
from services.run_service import RunService
from services.question_service import QuestionService
from services.graph_service import GraphService
from services.image_service import ImageService
from services.llm_service import LLMService, OllamaE4BLLMService, OllamaLLMService

logger = logging.getLogger(__name__)


_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Lifespan handler: load paths and initialize services on startup."""
    try:
        all_paths = load_all_demo_paths()

        # WebQA
        _init_webqa_services(application, all_paths.webqa)
        logger.info("WebQA run_id=%s", all_paths.webqa.run_id)
        for k, v in demo_paths_as_dict(all_paths.webqa).items():
            logger.info("webqa.paths.%s=%s", k, v)

        # MMQA
        if all_paths.mmqa is not None:
            _init_mmqa_services(application, all_paths.mmqa)
            application.state.mmqa_paths = all_paths.mmqa
            logger.info("MMQA run_id=%s", all_paths.mmqa.run_id)
        else:
            logger.info("No MMQA run directory found — MMQA endpoints will return 404.")

    except FileNotFoundError as e:
        logger.warning("Could not load paths: %s", e)
        logger.warning("Some endpoints may not work until result data exists.")
    yield


def _create_app() -> FastAPI:
    application = FastAPI(
        title="ColGraphRAG Demo API",
        description=(
            "Read-only viewer for ColGraphRAG pipeline results. "
            "Supports WebQA and MultimodalQA datasets."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health_router)
    application.include_router(datasets_router)
    application.include_router(run_router)
    application.include_router(questions_router)
    application.include_router(graph_router)
    application.include_router(images_router)
    application.include_router(chat_router)

    @application.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": "ColGraphRAG Demo API",
            "docs": "/docs",
            "health": "/health",
            "datasets": "/api/datasets",
            "run_info": "/api/run/info?dataset=webqa|mmqa",
            "questions": "/api/questions?dataset=webqa|mmqa",
            "graphs": "/api/graphs/{qid}?dataset=webqa|mmqa",
            "images": "/api/images/{image_id}?dataset=webqa|mmqa",
            "chat": "/api/chat",
        }

    return application


def _init_webqa_services(application: FastAPI, paths: DemoPaths) -> None:
    application.state.demo_paths = paths
    application.state.run_service = RunService(paths.result_run_dir)
    application.state.question_service = QuestionService(
        result_run_dir=paths.result_run_dir,
        webqa_questions_jsonl=paths.webqa_questions_jsonl,
        phase4_graphs_out=paths.phase4_graphs_out,
        dataset="webqa",
    )
    application.state.graph_service = GraphService(paths.phase4_graphs_out)
    application.state.image_service = ImageService(paths.webqa_imgs_dir)
    application.state.llm_service = LLMService(paths.phase4_graphs_out)
    application.state.ollama_llm_service = OllamaLLMService(paths.phase4_graphs_out)
    application.state.ollama_e4b_llm_service = OllamaE4BLLMService(paths.phase4_graphs_out)
    set_run_id(paths.run_id)


def _init_mmqa_services(application: FastAPI, paths: MmqaDemoPaths) -> None:
    application.state.mmqa_run_service = RunService(paths.result_run_dir)
    application.state.mmqa_question_service = QuestionService(
        result_run_dir=paths.result_run_dir,
        webqa_questions_jsonl=paths.mmqa_questions_jsonl,
        phase4_graphs_out=paths.phase4_graphs_out,
        dataset="mmqa",
    )
    application.state.mmqa_graph_service = GraphService(paths.phase4_graphs_out)
    application.state.mmqa_image_service = ImageService(paths.mmqa_imgs_dir)
    application.state.mmqa_llm_service = LLMService(paths.phase4_graphs_out)
    application.state.mmqa_ollama_llm_service = OllamaLLMService(paths.phase4_graphs_out)
    application.state.mmqa_ollama_e4b_llm_service = OllamaE4BLLMService(paths.phase4_graphs_out)


def init_demo_services(application: FastAPI, all_paths: MultiDatasetPaths) -> None:
    """Wire WebQA and optional MMQA into ``application.state`` (for tests or tooling)."""
    _init_webqa_services(application, all_paths.webqa)
    if all_paths.mmqa is not None:
        _init_mmqa_services(application, all_paths.mmqa)
        application.state.mmqa_paths = all_paths.mmqa


app = _create_app()


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="ColGraphRAG Demo BE")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (use 0.0.0.0 for RunPod / LAN)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Listen port")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (dev only)",
    )
    args = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=True,
    )
