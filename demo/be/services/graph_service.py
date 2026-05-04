"""Service for loading and parsing GraphML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.graph import GraphData, GraphNode, GraphEdge


class GraphService:
    """Parses GraphML files into JSON-friendly structures."""

    def __init__(self, phase4_graphs_out: Path) -> None:
        self._graphs_dir = phase4_graphs_out

    def _get_graphml_path(self, qid: str) -> Path:
        return self._graphs_dir / f"{qid}_graph.graphml"

    def has_graph(self, qid: str) -> bool:
        return self._get_graphml_path(qid).exists()

    def get_graphml_path(self, qid: str) -> Path | None:
        path = self._get_graphml_path(qid)
        return path if path.exists() else None

    def get_graph(self, qid: str) -> GraphData | None:
        """Parse GraphML and return structured graph data."""
        path = self._get_graphml_path(qid)
        if not path.exists():
            return None

        try:
            import networkx as nx

            G = nx.read_graphml(path)
        except Exception:
            return None

        nodes: list[GraphNode] = []
        for node_id, attrs in G.nodes(data=True):
            nodes.append(
                GraphNode(
                    id=str(node_id),
                    entity_name=attrs.get("entity_name"),
                    node_type=attrs.get("type"),
                    description=attrs.get("description"),
                    source_id=attrs.get("source_id"),
                )
            )

        edges: list[GraphEdge] = []
        for u, v, attrs in G.edges(data=True):
            edges.append(
                GraphEdge(
                    source=str(u),
                    target=str(v),
                    weight=attrs.get("weight"),
                    description=attrs.get("description"),
                    source_id=attrs.get("source_id"),
                )
            )

        return GraphData(
            qid=qid,
            nodes=nodes,
            edges=edges,
            node_count=len(nodes),
            edge_count=len(edges),
        )
