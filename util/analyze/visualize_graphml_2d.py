#!/usr/bin/env python3
"""
Static 2D GraphML figures (matplotlib) for papers / slides.

See `.dev_document/txt/20260320_1312.txt` § "1) 2D 코드".

Usage (from `query-driven_mm_graph_rag`):
  python util/analyze/visualize_graphml_2d.py --graphml result/.../foo_graph.graphml --out fig.png
  # Always writes YYYYMMDD_HHMMSS_fig.png (local time prefix on the basename).
  python util/analyze/visualize_graphml_2d.py --graphml ... --out fig.png --layout kamada_kawai
  # Relation text on edges (GraphML ``description``) is on by default; use --hide-edge-labels to omit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from graphml_viz_common import (
    get_node_color,
    get_node_label,
    load_graph,
    shorten,
    with_timestamp_prefix,
)


def draw_graph_2d(
    graphml_path: str | Path,
    out_path: str | Path,
    *,
    layout: str = "spring",
    figsize: tuple[float, float] = (10.0, 8.0),
    show_edge_labels: bool = True,
    max_nodes: int = 30,
    seed: int = 42,
    max_edges_for_labels: int = 40,
) -> None:
    path = Path(graphml_path)
    g = load_graph(path, max_nodes=max_nodes)

    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(g)
    elif layout == "circular":
        pos = nx.circular_layout(g)
    else:
        pos = nx.spring_layout(g, seed=seed, k=0.9)

    node_colors = [get_node_color(g.nodes[n]) for n in g.nodes()]
    node_labels = {n: get_node_label(g.nodes[n]) for n in g.nodes()}

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_colors,
        node_size=1100,
        alpha=0.95,
        linewidths=1.0,
        edgecolors="black",
    )

    nx.draw_networkx_edges(
        g,
        pos,
        width=1.5,
        alpha=0.45,
    )

    nx.draw_networkx_labels(
        g,
        pos,
        labels=node_labels,
        font_size=8,
        font_weight="bold",
    )

    if show_edge_labels and g.number_of_edges() <= max_edges_for_labels:
        edge_labels: dict[tuple[Any, Any], str] = {}
        for u, v, d in g.edges(data=True):
            desc = (d.get("description") or d.get("entity_name") or "").strip()
            if desc:
                edge_labels[(u, v)] = shorten(desc, 28)

        if edge_labels:
            nx.draw_networkx_edge_labels(
                g,
                pos,
                edge_labels=edge_labels,
                font_size=7,
                font_color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.78),
            )

    plt.title(path.stem, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="2D matplotlib export for Phase-4 GraphML.")
    parser.add_argument("--graphml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--layout",
        type=str,
        default="spring",
        choices=["spring", "kamada_kawai", "circular"],
    )
    parser.add_argument(
        "--hide-edge-labels",
        action="store_true",
        help="Omit relation labels on edges (GraphML edge description / entity_name)",
    )
    parser.add_argument("--max-nodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = with_timestamp_prefix(args.out.resolve())
    draw_graph_2d(
        graphml_path=args.graphml.resolve(),
        out_path=out_path,
        layout=args.layout,
        show_edge_labels=not args.hide_edge_labels,
        max_nodes=args.max_nodes,
        seed=args.seed,
    )
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
