#!/usr/bin/env python
"""
Run full WebQA pipeline with configurable query count.

This is a convenience wrapper around tests/test_pipeline.py that defaults
to running ALL toy dataset queries (66) instead of just 5.

Usage:
    # Real models + CUDA (default; MMGRAPHRAG_STRICT_REAL=1 in children unless --dry-run)
    python scripts/run_full_pipeline.py

    # Subset of queries
    python scripts/run_full_pipeline.py -n 10

    # Wiring-only (forces MMGRAPHRAG_STRICT_REAL=0 via test_pipeline)
    python scripts/run_full_pipeline.py --dry-run

    # Custom run ID
    python scripts/run_full_pipeline.py --run-id my_experiment

    Optional flags are forwarded: ``--dataset mmqa``, ``--webqa-data-dir``,
    ``--mmqa-data-dir``, ``--webqa-slice-dir``, ``--mmqa-slice-dir``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Default to all 66 toy dataset queries
DEFAULT_N_QUERIES = 66


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    rr = str(repo_root)
    if rr not in sys.path:
        sys.path.insert(0, rr)
    from util.pipeline_session_log import run_with_session_stdio_tee

    def _run() -> int:
        parser = argparse.ArgumentParser(
            description="Run full WebQA GraphRAG pipeline (wrapper for tests/test_pipeline.py)"
        )
        parser.add_argument(
            "-n", "--n-queries",
            type=int,
            default=DEFAULT_N_QUERIES,
            help=f"Number of queries to process (default: {DEFAULT_N_QUERIES} = all toy queries)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Dry-run mode (placeholder LLM, no GPU required)",
        )
        parser.add_argument(
            "--run-id",
            type=str,
            default="",
            help="Custom run ID (default: auto-generated timestamp)",
        )
        parser.add_argument(
            "--no-toy",
            action="store_true",
            help="Use full WebQA export instead of toy dataset",
        )
        parser.add_argument(
            "--dataset",
            choices=("webqa", "mmqa"),
            default="webqa",
            help="Forwarded to tests/test_pipeline.py (default: webqa).",
        )
        parser.add_argument(
            "--webqa-data-dir",
            type=str,
            default="",
            help="Forwarded: WebQA bundle root (default data/webqa in repo).",
        )
        parser.add_argument(
            "--webqa-slice-dir",
            type=str,
            default="",
            help="Forwarded: pre-built webqa_slice/ for WebQA.",
        )
        parser.add_argument(
            "--mmqa-data-dir",
            type=str,
            default="",
            help="Forwarded: MMQA export root when --dataset mmqa.",
        )
        parser.add_argument(
            "--mmqa-slice-dir",
            type=str,
            default="",
            help="Forwarded: pre-built mmqa_slice/ for MMQA.",
        )
        args = parser.parse_args()

        test_script = repo_root / "tests" / "test_pipeline.py"

        if not test_script.is_file():
            print(f"[ERROR] test_pipeline.py not found: {test_script}")
            return 1

        cmd = [
            sys.executable,
            str(test_script),
            "--dataset",
            args.dataset,
            "-n",
            str(args.n_queries),
        ]

        if args.dry_run:
            cmd.append("--dry-run")
        if args.run_id:
            cmd.extend(["--run-id", args.run_id])
        if args.no_toy:
            cmd.append("--no-toy")
        if args.webqa_data_dir.strip():
            cmd.extend(["--webqa-data-dir", args.webqa_data_dir.strip()])
        if args.webqa_slice_dir.strip():
            cmd.extend(["--webqa-slice-dir", args.webqa_slice_dir.strip()])
        if args.mmqa_data_dir.strip():
            cmd.extend(["--mmqa-data-dir", args.mmqa_data_dir.strip()])
        if args.mmqa_slice_dir.strip():
            cmd.extend(["--mmqa-slice-dir", args.mmqa_slice_dir.strip()])

        print(f"{'='*60}")
        print("GraphRAG Full Pipeline Runner (tests/test_pipeline.py)")
        print(f"{'='*60}")
        print(f"Dataset: {args.dataset}")
        print(f"Queries: {args.n_queries}")
        print(f"Dry-run: {args.dry_run}")
        print(f"Run ID: {args.run_id or '(auto)'}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        return subprocess.run(cmd, cwd=str(repo_root)).returncode

    return run_with_session_stdio_tee(repo_root, "run_full_pipeline", _run)


if __name__ == "__main__":
    sys.exit(main())
