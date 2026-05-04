"""Graph endpoints for knowledge graph visualization."""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse

from schemas.graph import GraphData


router = APIRouter(prefix="/api/graphs", tags=["graphs"])


@router.get("/{qid}", response_model=GraphData)
async def get_graph(qid: str, request: Request) -> GraphData:
    """Return graph nodes and edges for visualization."""
    graph_service = request.app.state.graph_service
    graph = graph_service.get_graph(qid)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Graph for {qid} not found")
    return graph


@router.get("/{qid}/graphml")
async def get_graphml_file(qid: str, request: Request) -> FileResponse:
    """Download raw GraphML file."""
    graph_service = request.app.state.graph_service
    path = graph_service.get_graphml_path(qid)
    if path is None:
        raise HTTPException(status_code=404, detail=f"GraphML for {qid} not found")
    return FileResponse(
        path,
        media_type="application/xml",
        filename=f"{qid}_graph.graphml",
    )
