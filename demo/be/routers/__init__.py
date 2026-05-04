"""API routers for the demo backend."""

from routers.health_router import router as health_router
from routers.run_router import router as run_router
from routers.questions_router import router as questions_router
from routers.graph_router import router as graph_router
from routers.images_router import router as images_router
from routers.chat_router import router as chat_router

__all__ = [
    "health_router",
    "run_router",
    "questions_router",
    "graph_router",
    "images_router",
    "chat_router",
]
