#!/usr/bin/env python3
"""
Interactive 3D GraphML visualization (Plotly).

Implements `.dev_document/txt/20260320_1312.txt`:
  - layered: z-layers (query / entities / evidence / answer heuristics)
  - constellation: components separated in 3D (fragmented constellation)
  - spring3d: plain 3D spring layout (doc mockup; good for quick structure view)

Edge styling: ``weight`` -> line width; ``description`` length -> opacity (doc:
thickness / transparency).

Usage (from `query-driven_mm_graph_rag`):
  python util/analyze/visualize_graphml_3d.py result/.../foo_graph.graphml -o viz.html --mode layered
  # Always writes YYYYMMDD_HHMMSS_viz.html (local time prefix on the basename).
  python util/analyze/visualize_graphml_3d.py result/.../foo_graph.graphml -o viz.html --mode spring3d --show-labels
  python util/analyze/visualize_graphml_3d.py ... -o out.html --standalone
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Plotly is required. Install with: pip install plotly>=5.18.0"
    ) from e

from graphml_viz_common import (
    get_node_color,
    get_node_label,
    graph_summary,
    load_graph,
    normalized_type,
    shorten,
    with_timestamp_prefix,
)


def plotly_interaction_config() -> dict[str, Any]:
    """
    Plotly.js config for strong in-browser interaction (orbit, zoom, hover, export).

    Open the exported HTML locally: drag to rotate, scroll to zoom, double-click resets
    the camera; toolbar supports reset axes and PNG download.
    """
    return {
        "responsive": True,
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "doubleClick": "reset",
        "showTips": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "graph_3d",
            "height": 900,
            "width": 1200,
            "scale": 2,
        },
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }


def _trunc(s: str | None, max_len: int = 240) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _node_attrs(G: nx.Graph, n: Any) -> dict[str, str]:
    d = G.nodes[n]
    return {k: (v if isinstance(v, str) else str(v)) for k, v in d.items()}


def infer_hub_nodes(G: nx.Graph, question_override: set[Any] | None = None) -> set[Any]:
    if question_override:
        return question_override
    candidates: list[tuple[int, Any]] = []
    for n in G.nodes:
        t = (_node_attrs(G, n).get("type") or "").strip()
        if t:
            continue
        candidates.append((G.degree(n), n))
    if not candidates:
        return set()
    m = max(candidates, key=lambda x: x[0])[0]
    return {n for deg, n in candidates if deg == m and m > 0}


def node_layer_z(G: nx.Graph, n: Any, hub_nodes: set[Any]) -> float:
    a = _node_attrs(G, n)
    t = (a.get("type") or "").strip().upper()
    en = (a.get("entity_name") or "").strip().lower()
    src = a.get("source_id") or ""
    comma_parts = src.count(",") + 1 if src.strip() else 0

    if t in ("QUESTION", "QUERY") or "query" in en:
        return 0.0
    if "ANSWER" in t or "answer" in en:
        return 3.0
    if "TABLE" in t or t == "TABLE":
        return 2.0
    if n in hub_nodes:
        return 0.25
    if comma_parts >= 3:
        return 1.75
    return 1.0


def layout_layered(
    G: nx.Graph,
    *,
    layer_scale: float = 2.2,
    xy_scale: float = 1.0,
    seed: int = 0,
    question_nodes: set[Any] | None = None,
) -> dict[Any, tuple[float, float, float]]:
    hub = infer_hub_nodes(G, question_override=question_nodes)
    zmap = {n: node_layer_z(G, n, hub) for n in G}
    pos2 = nx.spring_layout(G, dim=2, seed=seed)
    out: dict[Any, tuple[float, float, float]] = {}
    for n in G:
        x, y = pos2[n]
        out[n] = (float(x) * xy_scale, float(y) * xy_scale, float(zmap[n]) * layer_scale)
    return out


def layout_constellation(
    G: nx.Graph,
    *,
    spread: float = 4.0,
    intra_scale: float = 0.85,
    seed: int = 0,
) -> dict[Any, tuple[float, float, float]]:
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    rng = np.random.default_rng(seed)
    pos3: dict[Any, tuple[float, float, float]] = {}
    n_comp = len(comps)
    for i, comp in enumerate(comps):
        sub = G.subgraph(comp).copy()
        p2 = nx.spring_layout(sub, dim=2, seed=seed + i + 1)
        if i == 0:
            cx = cy = cz = 0.0
        else:
            theta = 2.0 * math.pi * (i - 1) / max(n_comp - 1, 1)
            r = spread * (1.0 + 0.12 * i)
            cx = r * math.cos(theta)
            cy = r * math.sin(theta)
            cz = float(0.25 * rng.standard_normal())
        for n in sub:
            x, y = p2[n]
            pos3[n] = (
                cx + float(x) * intra_scale,
                cy + float(y) * intra_scale,
                cz,
            )
    return pos3


def layout_spring3d(
    G: nx.Graph,
    *,
    seed: int = 42,
    k: float | None = 0.9,
) -> dict[Any, tuple[float, float, float]]:
    pos = nx.spring_layout(G, dim=3, seed=seed, k=k)
    return {n: (float(pos[n][0]), float(pos[n][1]), float(pos[n][2])) for n in G}


def _edge_weight(ed: dict[str, Any]) -> float:
    w = ed.get("weight")
    try:
        return float(w) if w is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


def _edge_linestyle(ed: dict[str, Any], w_min: float, w_max: float) -> tuple[float, float]:
    """
    Return (width, opacity) for Plotly 3D lines.

    Phase-4 exports are sparse (few edges, many isolates). Thin gray lines disappear in 3D,
    so we use a higher width/alpha floor than for dense 2D plots.
    """
    w = _edge_weight(ed)
    if w_max <= w_min:
        width = 3.5
    else:
        t = (w - w_min) / (w_max - w_min)
        width = 2.5 + t * 5.0
    desc = (ed.get("description") or "").strip()
    if not desc:
        alpha = 0.55
    else:
        alpha = min(0.92, 0.55 + min(len(desc), 120) / 200.0)
    return width, alpha


def _edge_relation_label(edge_data: dict[str, Any]) -> str:
    """Phase-4 edges store the relation string in ``description`` (e.g. ``participates_in``)."""
    return (edge_data.get("description") or edge_data.get("entity_name") or "").strip()


def build_figure(
    G: nx.Graph,
    pos: dict[Any, tuple[float, float, float]],
    *,
    title: str,
    show_labels: bool = False,
    show_relation_labels: bool = True,
    max_relation_labels: int = 48,
) -> go.Figure:
    nodes = list(G.nodes())
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    zs = [pos[n][2] for n in nodes]
    node_colors = [get_node_color(G.nodes[n]) for n in nodes]

    hover_lines: list[str] = []
    for n in nodes:
        a = _node_attrs(G, n)
        name = html.escape(_trunc(a.get("entity_name") or str(n), 120))
        typ = html.escape(_trunc(a.get("type"), 80))
        desc = html.escape(_trunc(a.get("description"), 400))
        sid = html.escape(_trunc(a.get("source_id"), 200))
        deg = G.degree[n]
        hover_lines.append(
            f"<b>{name}</b><br>type: {typ}<br>degree: {deg}<br><br>{desc}<br><br>source_id: {sid}"
        )

    edge_data = [(u, v, d) for u, v, d in G.edges(data=True)]
    weights = [_edge_weight(d) for _, _, d in edge_data]
    w_min = min(weights) if weights else 0.0
    w_max = max(weights) if weights else 1.0

    node_trace_kw: dict[str, Any] = dict(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text" if show_labels else "markers",
        marker=dict(
            size=[6 + min(10, G.degree(n)) for n in nodes],
            color=node_colors,
            line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
        ),
        hovertext=hover_lines,
        hovertemplate="%{hovertext}<extra></extra>",
        name="nodes",
    )
    if show_labels:
        node_trace_kw["text"] = [get_node_label(G.nodes[n], max_len=24) for n in nodes]
        node_trace_kw["textposition"] = "top center"
    traces: list[Any] = [go.Scatter3d(**node_trace_kw)]

    # Edges after nodes so relation lines draw on top (easier to see in sparse graphs).
    for u, v, d in edge_data:
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        width, alpha = _edge_linestyle(d, w_min, w_max)
        desc = (d.get("description") or "").strip()
        en = (d.get("entity_name") or "").strip()
        sid_e = _trunc(d.get("source_id"), 120)
        hover_e = html.escape(
            f"edge<br>description: {_trunc(desc, 200)}<br>entity_name: {en}<br>weight: {_edge_weight(d)}<br>source_id: {sid_e}"
        )
        traces.append(
            go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode="lines",
                line=dict(
                    width=max(2.0, width),
                    color=f"rgba(25,25,35,{alpha:.3f})",
                ),
                hovertext=hover_e,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
            )
        )

    if show_relation_labels and len(edge_data) <= max_relation_labels:
        lx: list[float] = []
        ly: list[float] = []
        lz: list[float] = []
        ltxt: list[str] = []
        for u, v, d in edge_data:
            rel = _edge_relation_label(d)
            if not rel:
                continue
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            lx.append((x0 + x1) / 2.0)
            ly.append((y0 + y1) / 2.0)
            lz.append((z0 + z1) / 2.0)
            ltxt.append(shorten(rel, 36))
        if ltxt:
            traces.append(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="text",
                    text=ltxt,
                    textposition="middle center",
                    textfont=dict(size=11, color="rgb(20,20,28)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        hovermode="closest",
        autosize=True,
        height=820,
        scene=dict(
            dragmode="orbit",
            xaxis=dict(showbackground=False, title="x"),
            yaxis=dict(showbackground=False, title="y"),
            zaxis=dict(showbackground=False, title="z"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def _legend_type_colors(G: nx.Graph) -> list[go.Scatter3d]:
    """Legend-only traces listing node ``type`` values present in ``G``."""
    seen: set[str] = set()
    traces: list[go.Scatter3d] = []
    for n in G.nodes():
        t = normalized_type(G.nodes[n])
        if t in seen:
            continue
        seen.add(t)
        c = get_node_color(G.nodes[n])
        traces.append(
            go.Scatter3d(
                x=[0.0],
                y=[0.0],
                z=[0.0],
                mode="markers",
                marker=dict(size=10, color=c, opacity=1.0),
                name=t,
                showlegend=True,
                visible="legendonly",
            )
        )
    return traces


def visualize_graphml(
    graphml_path: Path,
    output_path: Path,
    *,
    mode: str,
    seed: int,
    layer_scale: float,
    spread: float,
    max_nodes: int,
    show_labels: bool,
    question_node: list[str] | None,
    type_legend: bool,
    standalone_html: bool = False,
    show_relation_labels: bool = True,
) -> None:
    mn = max_nodes if max_nodes > 0 else 0
    G = load_graph(graphml_path, max_nodes=mn if mn > 0 else None)
    if G.number_of_nodes() == 0:
        raise SystemExit(f"No nodes in {graphml_path}")

    qset: set[Any] | None = None
    if question_node:
        qset = set()
        for q in question_node:
            if q in G:
                qset.add(q)
            else:
                raise SystemExit(f"--question-node not in graph: {q!r}")

    summ = graph_summary(G)
    if mode == "layered":
        pos = layout_layered(G, layer_scale=layer_scale, seed=seed, question_nodes=qset)
        subtitle = "layered: z ~ query / entity / table-evidence / answer heuristics"
    elif mode == "constellation":
        pos = layout_constellation(G, spread=spread, seed=seed)
        subtitle = (
            f"constellation: {summ['components']} components; "
            f"largest_cc_ratio={summ['largest_component_ratio']:.2f}"
        )
    elif mode == "spring3d":
        pos = layout_spring3d(G, seed=seed)
        subtitle = "spring3d: 3D force layout (doc mockup)"
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    title = (
        f"{graphml_path.name}<br><sup>{subtitle}</sup><br>"
        f"<sup>n={summ['nodes']}, m={summ['edges']}, "
        f"density={summ['density']:.4f}</sup>"
    )
    fig = build_figure(
        G,
        pos,
        title=title,
        show_labels=show_labels,
        show_relation_labels=show_relation_labels,
    )
    if type_legend:
        for t in _legend_type_colors(G):
            fig.add_trace(t)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    include_js: bool | str = True if standalone_html else "cdn"
    fig.write_html(
        output_path,
        include_plotlyjs=include_js,
        config=plotly_interaction_config(),
        div_id="graph-3d-interactive",
    )
    print(
        f"Wrote {output_path} ({summ['nodes']} nodes, {summ['edges']} edges, "
        f"{summ['components']} components)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="3D Plotly visualization for Phase-4 GraphML.")
    ap.add_argument("graphml", type=Path, help="Path to *_graph.graphml")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .html path",
    )
    ap.add_argument(
        "--mode",
        choices=("layered", "constellation", "spring3d"),
        default="layered",
        help="layered | constellation | spring3d (plain 3D spring)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--layer-scale",
        type=float,
        default=2.2,
        help="Z separation multiplier for layered mode",
    )
    ap.add_argument(
        "--spread",
        type=float,
        default=4.0,
        help="Radial spread between components (constellation mode)",
    )
    ap.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="If >0, keep largest connected component only when |V| exceeds this (doc guard)",
    )
    ap.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw entity_name labels in 3D (can clutter)",
    )
    ap.add_argument(
        "--question-node",
        action="append",
        default=[],
        metavar="NODE_ID",
        help="GraphML node id to pin to query layer (repeatable)",
    )
    ap.add_argument(
        "--type-legend",
        action="store_true",
        help="Add legend entries for node types (legend-only proxy traces)",
    )
    ap.add_argument(
        "--standalone",
        action="store_true",
        help="Embed full plotly.js in the HTML (larger file; works offline; max interaction)",
    )
    ap.add_argument(
        "--hide-relation-labels",
        action="store_true",
        help="Do not draw edge relation text (GraphML edge description) at segment midpoints",
    )
    args = ap.parse_args()
    output_path = with_timestamp_prefix(args.output.resolve())
    visualize_graphml(
        args.graphml.resolve(),
        output_path,
        mode=args.mode,
        seed=args.seed,
        layer_scale=args.layer_scale,
        spread=args.spread,
        max_nodes=args.max_nodes,
        show_labels=args.show_labels,
        question_node=args.question_node or None,
        type_legend=args.type_legend,
        standalone_html=args.standalone,
        show_relation_labels=not args.hide_relation_labels,
    )


if __name__ == "__main__":
    main()
