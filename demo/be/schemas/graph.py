"""Pydantic models for graph data."""

from __future__ import annotations

from pydantic import BaseModel


class GraphNode(BaseModel):
    """Node in the knowledge graph."""

    id: str
    entity_name: str | None = None
    node_type: str | None = None
    description: str | None = None
    source_id: str | None = None


class GraphEdge(BaseModel):
    """Edge in the knowledge graph."""

    source: str
    target: str
    weight: float | None = None
    description: str | None = None
    source_id: str | None = None


class GraphData(BaseModel):
    """Complete graph structure for visualization."""

    qid: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    node_count: int
    edge_count: int
