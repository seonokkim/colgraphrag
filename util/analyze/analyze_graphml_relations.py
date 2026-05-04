#!/usr/bin/env python3
"""
Aggregate edge-level information from Phase-4 GraphML exports.

This codebase does not store a fixed ontology of relation *types*; edges use
free-text `description` (and sometimes `entity_name`). This script counts edges,
unique descriptions, and top frequent strings for reporting.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from graphml_viz_common import with_timestamp_prefix


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def parse_graphml_edge_stats(path: Path) -> tuple[list[dict], dict[str, str]]:
    """Return list of edge records {description, entity_name} and key id -> attr.name map."""
    tree = ET.parse(path)
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    key_map: dict[str, str] = {}
    for el in root.iter():
        if _strip_ns(el.tag) != "key":
            continue
        kid = el.get("id")
        aname = el.get("attr.name")
        if kid and aname:
            key_map[kid] = aname

    edges: list[dict] = []
    for el in root.iter():
        if _strip_ns(el.tag) != "edge":
            continue
        desc, ename = "", ""
        for child in el:
            if _strip_ns(child.tag) != "data":
                continue
            k = child.get("key")
            if not k:
                continue
            name = key_map.get(k, k)
            text = (child.text or "").strip()
            if name == "description":
                desc = text
            elif name == "entity_name":
                ename = text
        edges.append({"description": desc, "entity_name": ename})
    return edges, key_map


def normalize_desc(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s[:500]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "graph_dir",
        type=Path,
        help="Directory containing *_graph.graphml files",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Write JSON summary here",
    )
    args = ap.parse_args()

    files = sorted(args.graph_dir.glob("*_graph.graphml"))
    if not files:
        raise SystemExit(f"No *_graph.graphml under {args.graph_dir}")

    all_desc: Counter[str] = Counter()
    all_entity_name: Counter[str] = Counter()
    empty_desc = 0
    total_edges = 0
    desc_lengths: list[int] = []

    per_graph: list[dict] = []

    for fp in files:
        edges, _ = parse_graphml_edge_stats(fp)
        g_n = len(edges)
        total_edges += g_n
        g_empty = sum(1 for e in edges if not e["description"])
        empty_desc += g_empty
        for e in edges:
            d = e["description"]
            if d:
                nd = normalize_desc(d)
                all_desc[nd] += 1
                desc_lengths.append(len(d))
            en = e["entity_name"].strip()
            if en:
                all_entity_name[en] += 1
        per_graph.append(
            {
                "file": fp.name,
                "edge_count": g_n,
                "empty_description_edges": g_empty,
            }
        )

    top_desc = all_desc.most_common(40)
    top_ename = all_entity_name.most_common(30)

    def _pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 2) if d else 0.0

    summary = {
        "source_dir": str(args.graph_dir.resolve()),
        "graph_files": len(files),
        "total_edges": total_edges,
        "edges_with_empty_description": empty_desc,
        "pct_edges_empty_description": _pct(empty_desc, total_edges),
        "unique_normalized_descriptions": len(all_desc),
        "unique_entity_name_values": len(all_entity_name),
        "description_length_chars": {
            "min": min(desc_lengths) if desc_lengths else 0,
            "max": max(desc_lengths) if desc_lengths else 0,
            "mean": round(sum(desc_lengths) / len(desc_lengths), 2)
            if desc_lengths
            else 0.0,
        },
        "note": (
            "GraphML edges in this pipeline store natural-language `description` "
            "and optional `entity_name` (e.g. table title), not a closed set of "
            "typed relation labels."
        ),
        "top_edge_descriptions_by_frequency": [
            {"description": d, "count": c} for d, c in top_desc
        ],
        "top_edge_entity_name_by_frequency": [
            {"entity_name": n, "count": c} for n, c in top_ename
        ],
        "per_graph_edge_counts": per_graph,
    }

    out_path = with_timestamp_prefix(args.output.resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({total_edges} edges from {len(files)} graphs)")


if __name__ == "__main__":
    main()
