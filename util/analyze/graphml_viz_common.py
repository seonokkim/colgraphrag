"""
Shared helpers for GraphML 2D/3D visualizers.

Aligns with `.dev_document/txt/20260320_1312.txt`: type colors, LCC guard, labels.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

# Tableau-like palette; keys are UPPER case node `type` values from Phase-4 exports.
TYPE_COLOR: dict[str, str] = {
    "PERSON": "#4C78A8",
    "TABLE": "#F58518",
    "EVENT": "#54A24B",
    "CHARACTER": "#E45756",
    "MOVIE": "#72B7B2",
    "LOCATION": "#B279A2",
    "ORGANIZATION": "#FF9DA6",
    "RACE": "#00CC96",
    "ANIMAL": "#AB63FA",
    "EMPTY": "#BAB0AC",
    "UNKNOWN": "#9D9D9D",
}


def with_timestamp_prefix(out_path: str | Path, *, enabled: bool = True) -> Path:
    """
    Return ``out_path`` with local ``YYYYMMDD_HHMMSS_`` prepended to the filename.

    Analyze/visualize CLIs use this for every artifact path so outputs sort by run time.
    Pass ``enabled=False`` only for rare tests or programmatic overrides.
    """
    p = Path(out_path)
    if not enabled:
        return p
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return p.with_name(f"{ts}_{p.name}")


def shorten(text: Any, max_len: int = 18) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def normalized_type(attrs: dict[str, Any]) -> str:
    t = str(attrs.get("type", "") or "").strip().upper()
    return t if t else "EMPTY"


def get_node_label(attrs: dict[str, Any], *, max_len: int = 20) -> str:
    return shorten(
        attrs.get("entity_name")
        or attrs.get("description")
        or attrs.get("type")
        or "UNK",
        max_len=max_len,
    )


def get_node_color(attrs: dict[str, Any]) -> str:
    return TYPE_COLOR.get(normalized_type(attrs), TYPE_COLOR["UNKNOWN"])


def simplify_multigraph(g: nx.Graph) -> nx.Graph:
    if isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        h: nx.Graph = nx.Graph()
        for n, d in g.nodes(data=True):
            h.add_node(n, **d)
        for u, v, d in g.edges(data=True):
            if not h.has_edge(u, v):
                h.add_edge(u, v, **d)
        return h
    return g


def load_graph(
    graphml_path: str | Path,
    *,
    max_nodes: int | None = 30,
) -> nx.Graph:
    """
    Read GraphML; optionally keep only the largest (weakly) connected component
    when |V| > max_nodes (doc: readability guard for large exports).

    Pass ``max_nodes <= 0`` or ``None`` to disable trimming.
    """
    path = Path(graphml_path)
    g: nx.Graph = nx.read_graphml(path)
    g = simplify_multigraph(g)

    if max_nodes is not None and max_nodes > 0 and g.number_of_nodes() > max_nodes:
        if nx.is_directed(g):
            largest = max(nx.weakly_connected_components(g), key=len)
        else:
            largest = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest).copy()

    return g


def graph_summary(G: nx.Graph) -> dict[str, Any]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n <= 1:
        dens = 0.0
    else:
        dens = float(2 * m / (n * (n - 1))) if not nx.is_directed(G) else float(m / (n * (n - 1)))
    comps = list(nx.connected_components(G)) if not nx.is_directed(G) else list(nx.weakly_connected_components(G))
    n_comp = len(comps)
    largest = max((len(c) for c in comps), default=0)
    ratio = (largest / n) if n else 0.0
    return {
        "nodes": n,
        "edges": m,
        "density": dens,
        "components": n_comp,
        "largest_component_ratio": ratio,
    }
