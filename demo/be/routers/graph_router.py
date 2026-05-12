"""Graph endpoints for knowledge graph visualization."""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import FileResponse

from schemas.graph import GraphData


router = APIRouter(prefix="/api/graphs", tags=["graphs"])


def _get_graph_service(request: Request, dataset: str):
    if dataset == "mmqa":
        svc = getattr(request.app.state, "mmqa_graph_service", None)
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="MultimodalQA demo not initialized (check demo/be/config).",
            )
        return svc
    svc = getattr(request.app.state, "graph_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="WebQA demo not initialized (check demo/be/config).",
        )
    return svc


@router.get("/{qid}", response_model=GraphData)
async def get_graph(
    qid: str,
    request: Request,
    dataset: str = Query(default="webqa"),
) -> GraphData:
    """Return graph nodes and edges for visualization."""
    graph_service = _get_graph_service(request, dataset)
    graph = graph_service.get_graph(qid)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Graph for {qid} not found")
    return graph


@router.get("/{qid}/graphml")
async def get_graphml_file(
    qid: str,
    request: Request,
    dataset: str = Query(default="webqa"),
) -> FileResponse:
    """Download raw GraphML file."""
    graph_service = _get_graph_service(request, dataset)
    path = graph_service.get_graphml_path(qid)
    if path is None:
        raise HTTPException(status_code=404, detail=f"GraphML for {qid} not found")
    return FileResponse(
        path,
        media_type="application/xml",
        filename=f"{qid}_graph.graphml",
    )
