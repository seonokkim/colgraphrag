#!/usr/bin/env python
"""
Run full WebQA pipeline with configurable query count.

This is a convenience wrapper around tests/test_pipeline.py that defaults
to running ALL toy dataset queries (66) instead of just 5.

Usage:
    # Run all 66 toy queries (default)
    python scripts/run_full_pipeline.py

    # Run specific number of queries
    python scripts/run_full_pipeline.py -n 10

    # Dry-run (no GPU, placeholder LLM)
    python scripts/run_full_pipeline.py --dry-run

    # Custom run ID
    python scripts/run_full_pipeline.py --run-id my_experiment
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Default to all 66 toy dataset queries
DEFAULT_N_QUERIES = 66


def main() -> int:
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
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    test_script = base_dir / "tests" / "test_pipeline.py"

    if not test_script.is_file():
        print(f"[ERROR] test_pipeline.py not found: {test_script}")
        return 1

    cmd = [sys.executable, str(test_script), "-n", str(args.n_queries)]

    if args.dry_run:
        cmd.append("--dry-run")
    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.no_toy:
        cmd.append("--no-toy")

    print(f"{'='*60}")
    print("WebQA Full Pipeline Runner")
    print(f"{'='*60}")
    print(f"Queries: {args.n_queries}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Run ID: {args.run_id or '(auto)'}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    return subprocess.run(cmd, cwd=str(base_dir)).returncode


if __name__ == "__main__":
    sys.exit(main())
