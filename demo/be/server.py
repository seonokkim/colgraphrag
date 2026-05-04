"""
ColGraphRAG WebQA Demo Backend Server.

A read-only viewer for WebQA pipeline results. No external search or LLM APIs needed.

Usage:
    cd demo/be
    python server.py              # 127.0.0.1:8000
    # or
    uvicorn server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_BE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BE_DIR))

from config.resolve_demo_paths import load_demo_paths, DemoPaths
from routers.health_router import router as health_router, set_run_id
from routers.run_router import router as run_router
from routers.questions_router import router as questions_router
from routers.graph_router import router as graph_router
from routers.images_router import router as images_router
from routers.chat_router import router as chat_router
from services.run_service import RunService
from services.question_service import QuestionService
from services.graph_service import GraphService
from services.image_service import ImageService
from services.llm_service import LLMService


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
        paths = load_demo_paths()
        _init_services(application, paths)
        print(f"[demo-be] Loaded run: {paths.run_id}")
        print(f"[demo-be] Result dir: {paths.result_run_dir}")
    except FileNotFoundError as e:
        print(f"[demo-be] Warning: Could not load paths: {e}")
        print("[demo-be] Some endpoints may not work until result data exists.")
    yield


def _create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    application = FastAPI(
        title="ColGraphRAG WebQA Demo API",
        description=(
            "Read-only viewer for ColGraphRAG WebQA pipeline results. "
            "Serves questions, predictions, evaluation scores, and knowledge graphs."
        ),
        version="0.1.0",
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
    application.include_router(run_router)
    application.include_router(questions_router)
    application.include_router(graph_router)
    application.include_router(images_router)
    application.include_router(chat_router)

    @application.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": "ColGraphRAG WebQA Demo API",
            "docs": "/docs",
            "health": "/health",
            "run_info": "/api/run/info",
            "run_scores": "/api/run/scores",
            "questions": "/api/questions",
            "graphs": "/api/graphs/{qid}",
            "images": "/api/images/{image_id}",
            "chat": "/api/chat",
        }

    return application


def _init_services(application: FastAPI, paths: DemoPaths) -> None:
    """Initialize services and attach to app state."""
    application.state.demo_paths = paths
    application.state.run_service = RunService(paths.result_run_dir)
    application.state.question_service = QuestionService(
        result_run_dir=paths.result_run_dir,
        webqa_questions_jsonl=paths.webqa_questions_jsonl,
        phase4_graphs_out=paths.phase4_graphs_out,
    )
    application.state.graph_service = GraphService(paths.phase4_graphs_out)
    application.state.image_service = ImageService(paths.webqa_imgs_dir)
    application.state.llm_service = LLMService(paths.phase4_graphs_out)
    set_run_id(paths.run_id)


app = _create_app()


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="ColGraphRAG WebQA Demo BE")
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
    )
