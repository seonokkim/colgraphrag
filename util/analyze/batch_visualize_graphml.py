#!/usr/bin/env python3
"""
Batch-export Phase-4 GraphML folders (e.g. ``phase4_graphs_real_100``) to 2D PNG and/or 3D HTML.

Usage (from ``query-driven_mm_graph_rag``)::

  python util/analyze/batch_visualize_graphml.py --graph-dir result/20260319/phase4_graphs_real_100 \\
    --out-dir figures/batch_export --formats both

Each output file is named ``YYYYMMDD_HHMMSS_<stem>.png`` / ``.html`` (local time prefix).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graphml_viz_common import with_timestamp_prefix
from visualize_graphml_2d import draw_graph_2d
from visualize_graphml_3d import visualize_graphml


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch 2D/3D export for directories of *_graph.graphml.")
    ap.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Directory containing *_graph.graphml",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output root (2d/ and 3d/ subfolders are created per format)",
    )
    ap.add_argument(
        "--formats",
        type=str,
        default="2d",
        choices=("2d", "3d", "both"),
        help="Export PNG (matplotlib), HTML (plotly), or both",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N files (sorted by name)",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error instead of skipping failed graphs",
    )
    # 2D
    ap.add_argument(
        "--layout",
        default="spring",
        choices=("spring", "kamada_kawai", "circular"),
    )
    ap.add_argument(
        "--hide-edge-labels",
        action="store_true",
        help="2D: no edge relation text; 3D: no midpoint relation labels",
    )
    ap.add_argument("--max-nodes-2d", type=int, default=30)
    ap.add_argument("--seed-2d", type=int, default=42)
    # 3D
    ap.add_argument(
        "--mode-3d",
        choices=("layered", "constellation", "spring3d"),
        default="layered",
    )
    ap.add_argument("--seed-3d", type=int, default=0)
    ap.add_argument("--layer-scale", type=float, default=2.2)
    ap.add_argument("--spread", type=float, default=4.0)
    ap.add_argument("--max-nodes-3d", type=int, default=0)
    ap.add_argument("--show-labels", action="store_true")
    ap.add_argument("--type-legend", action="store_true")
    ap.add_argument("--standalone", action="store_true")
    args = ap.parse_args()

    graph_dir = args.graph_dir.resolve()
    if not graph_dir.is_dir():
        raise SystemExit(f"Not a directory: {graph_dir}")

    files = sorted(graph_dir.glob("*_graph.graphml"))
    if not files:
        raise SystemExit(f"No *_graph.graphml under {graph_dir}")
    if args.limit > 0:
        files = files[: args.limit]

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    sub_2d = out_root / "2d"
    sub_3d = out_root / "3d"

    want_2d = args.formats in ("2d", "both")
    want_3d = args.formats in ("3d", "both")

    ok = 0
    failed: list[tuple[Path, str]] = []
    for fp in files:
        try:
            if want_2d:
                sub_2d.mkdir(parents=True, exist_ok=True)
                out_png = with_timestamp_prefix(sub_2d / f"{fp.stem}.png")
                draw_graph_2d(
                    fp,
                    out_png,
                    layout=args.layout,
                    show_edge_labels=not args.hide_edge_labels,
                    max_nodes=args.max_nodes_2d,
                    seed=args.seed_2d,
                )
            if want_3d:
                sub_3d.mkdir(parents=True, exist_ok=True)
                out_html = with_timestamp_prefix(sub_3d / f"{fp.stem}.html")
                visualize_graphml(
                    fp.resolve(),
                    out_html,
                    mode=args.mode_3d,
                    seed=args.seed_3d,
                    layer_scale=args.layer_scale,
                    spread=args.spread,
                    max_nodes=args.max_nodes_3d,
                    show_labels=args.show_labels,
                    question_node=None,
                    type_legend=args.type_legend,
                    standalone_html=args.standalone,
                    show_relation_labels=not args.hide_edge_labels,
                )
            ok += 1
        except Exception as e:  # noqa: BLE001 — batch: collect errors
            failed.append((fp, f"{type(e).__name__}: {e}"))
            if args.fail_fast:
                raise
            print(f"[skip] {fp.name}: {e}", file=sys.stderr)

    print(f"Done: {ok}/{len(files)} graphs -> {out_root}")
    if failed:
        print(f"Failed: {len(failed)}", file=sys.stderr)
        for fp, msg in failed[:20]:
            print(f"  {fp.name}: {msg}", file=sys.stderr)
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more", file=sys.stderr)


if __name__ == "__main__":
    main()
