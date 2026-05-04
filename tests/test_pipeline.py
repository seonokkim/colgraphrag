#!/usr/bin/env python
"""
Pipeline Smoke Test for colgraphrag_webqa (clean version).

Runs the complete graph-based pipeline (export -> pattern -> extraction ->
construct -> inference -> QA eval) for N WebQA questions.

Usage:
    # Default: 5 queries, real Gemma LLM on GPU (CUDA required), toy dataset
    python tests/test_pipeline.py

    # Fast dry-run test (no LLM calls, no GPU required)
    python tests/test_pipeline.py --dry-run

    # Override number of queries
    python tests/test_pipeline.py -n 10

    # Use full WebQA export instead of toy dataset
    python tests/test_pipeline.py --no-toy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
_LOCAL_DATA = _BASE_DIR / "data" / "webqa"


def _resolve_shard14_toy_slice() -> str:
    local = _LOCAL_DATA / "webqa_shard14_toy" / "webqa_slice"
    if local.is_dir():
        return str(local)
    return "/workspace/data/webqa/WebQA_imgs_7z_chunks/webqa_shard14_toy/webqa_slice"


def _resolve_shard14_imgs() -> str:
    local = _LOCAL_DATA / "WebQA_imgs_7z_chunks" / "imgs" / "all_png" / "shard_00014"
    if local.is_dir():
        return str(local)
    return "/workspace/data/webqa/WebQA_imgs_7z_chunks/imgs/all_png/shard_00014"


SHARD14_TOY_SLICE_DIR = _resolve_shard14_toy_slice()
SHARD14_TOY_IMGS_DIR = _resolve_shard14_imgs()


def run_cmd(cmd: list[str], env: dict[str, str], cwd: Path, desc: str) -> int:
    print(f"\n{'='*60}")
    print(f"[{desc}]")
    print(f"CMD: {' '.join(cmd)}")
    print(f"CWD: {cwd}")
    print(f"{'='*60}\n")

    merged_env = {**os.environ, **env}
    proc = subprocess.run(cmd, cwd=str(cwd), env=merged_env)

    if proc.returncode != 0:
        print(f"[ERROR] {desc} failed with exit code {proc.returncode}")
    else:
        print(f"[OK] {desc} completed successfully")

    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="WebQA pipeline test (clean)")
    parser.add_argument(
        "--n-queries", "-n", type=int, default=5,
        help="Number of queries to process (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use dry-run mode (placeholder LLM responses, no GPU required)",
    )
    parser.add_argument(
        "--run-id", type=str, default="",
        help="Custom run ID (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--gemma-model-path", type=str, default="",
        help="Override GEMMA4_E4B_IT_MODEL_PATH.",
    )
    parser.add_argument(
        "--toy", action="store_true", default=True,
        help="Use shard_00014 toy dataset (default: True).",
    )
    parser.add_argument(
        "--no-toy", action="store_true",
        help="Disable --toy and use export_webqa_slice.py instead.",
    )
    parser.add_argument(
        "--webqa-slice-dir", type=str, default="",
        help="Path to pre-built webqa_slice directory.",
    )
    args = parser.parse_args()

    if args.no_toy:
        args.toy = False

    if args.toy and args.webqa_slice_dir:
        print("[ERROR] Choose either --toy or --webqa-slice-dir, not both.")
        return 1

    prebuilt_slice_dir = None
    if args.toy:
        prebuilt_slice_dir = Path(SHARD14_TOY_SLICE_DIR)
        if not prebuilt_slice_dir.is_dir():
            print(f"[ERROR] Toy dataset not found: {SHARD14_TOY_SLICE_DIR}")
            return 1
    elif args.webqa_slice_dir:
        prebuilt_slice_dir = Path(args.webqa_slice_dir)
        if not prebuilt_slice_dir.is_dir():
            print(f"[ERROR] webqa_slice_dir not found: {args.webqa_slice_dir}")
            return 1

    base_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(base_dir))
    from util.llm_defaults import DEFAULT_GEMMA4_E4B_IT_MODEL_PATH
    from util.run_id import stamped_run_id

    run_id = args.run_id or stamped_run_id("5query_test")
    result_dir = base_dir / "result" / run_id

    py = sys.executable
    n_queries = args.n_queries
    dry_run = "1" if args.dry_run else "0"

    print(f"{'='*60}")
    print(f"colgraphrag_webqa Pipeline Test (clean)")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Base dir: {base_dir}")
    print(f"Result dir: {result_dir}")
    print(f"Python: {py}")
    print(f"Queries: {n_queries}")
    if args.toy:
        print(f"Data: --toy (shard_00014 toy dataset)")
    elif prebuilt_slice_dir:
        print(f"Data: --webqa-slice-dir {prebuilt_slice_dir}")
    else:
        print(f"Data: placeholder (export_webqa_slice.py)")
    if args.dry_run:
        print(f"Mode: --dry-run (placeholder LLM, fast, no GPU)")
    else:
        print(f"Mode: default (HF Gemma in-process)")
    print(f"{'='*60}")

    import torch
    if not args.dry_run:
        if not torch.cuda.is_available():
            print("[ERROR] CUDA required for HF Gemma (default mode). Use --dry-run for CPU-only testing.")
            return 1
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

    gemma_path = (args.gemma_model_path or "").strip()
    if not gemma_path:
        gemma_path = os.environ.get("GEMMA4_E4B_IT_MODEL_PATH", "").strip() or DEFAULT_GEMMA4_E4B_IT_MODEL_PATH

    common_env = {
        "PYTHONPATH": str(base_dir),
        "MMGRAPHRAG_RUN_ID": run_id,
        "WEBQA_RUN_PROFILE": "val_n100",
        "WEBQA_DATA_ROOT": str(_LOCAL_DATA) if _LOCAL_DATA.is_dir() else "/workspace/data/webqa",
        "PATTERN_MAX_SAMPLES": str(n_queries),
        "WEBQA_EXPORT_MAX": str(n_queries),
        "EXTRACTION_MAX_QUESTIONS": str(n_queries),
        "CONSTRUCT_MAX_QUESTIONS": str(n_queries),
        "PATTERN_DRY_RUN": dry_run,
        "EXTRACTION_DRY_RUN": dry_run,
        "PATTERN_CONCURRENCY": "1",
        "EXTRACTION_CONCURRENCY": "1",
        "GEMMA4_E4B_IT_MODEL_PATH": gemma_path,
        "VIDORE_TEXT_LLM_BACKEND": "hf_gemma_4_e4b_it",
        **(
            {"GEMMA4_E4B_IT_TORCH_DTYPE": os.environ["GEMMA4_E4B_IT_TORCH_DTYPE"]}
            if os.environ.get("GEMMA4_E4B_IT_TORCH_DTYPE")
            else {}
        ),
    }

    common_env.setdefault(
        "COLEMBED_MODEL_PATH",
        os.environ.get(
            "COLEMBED_MODEL_PATH",
            "/workspace/models/retriever/llama-nemotron-colembed-vl-3b-v2",
        ),
    )

    results = {}

    # ── Phase 0: Export WebQA slice ──
    print(f"\n{'='*60}\n[PHASE 0] Export WebQA slice to JSONL\n{'='*60}")
    slice_dir = result_dir / "webqa_slice"

    if prebuilt_slice_dir:
        print(f"[INFO] Using pre-built slice: {prebuilt_slice_dir}")
        slice_dir.parent.mkdir(parents=True, exist_ok=True)
        if slice_dir.exists():
            shutil.rmtree(slice_dir)
        shutil.copytree(prebuilt_slice_dir, slice_dir)
        results["phase0_export"] = True
        print(f"[OK] Copied pre-built slice to {slice_dir}")
    else:
        export_env = {**common_env, "WEBQA_SLICE_DIR": str(slice_dir)}
        rc = run_cmd([py, "export_webqa_slice.py"], export_env, base_dir, "Export WebQA slice")
        results["phase0_export"] = rc == 0
        if rc != 0:
            print("[ABORT] Phase 0 failed, cannot continue")
            sys.exit(1)

    q_file = slice_dir / "webqa_questions.jsonl"
    t_file = slice_dir / "webqa_texts.jsonl"
    if not q_file.exists() or not t_file.exists():
        print(f"[ERROR] Export files missing: {q_file.exists()=}, {t_file.exists()=}")
        sys.exit(1)

    with open(q_file, "r") as f:
        n_exported = sum(1 for _ in f)
    print(f"[INFO] Slice contains {n_exported} questions in {q_file}")

    # ── Phase 2: Pattern extraction ──
    print(f"\n{'='*60}\n[PHASE 2] Pattern extraction\n{'='*60}")
    pattern_cache = result_dir / "phase2_pattern_cache"
    _local_pattern = _LOCAL_DATA / "webqa_shard14_toy" / "webqa_slice" / "webqa_questions.jsonl"
    if _local_pattern.is_file():
        pattern_question_file = str(_local_pattern)
    else:
        pattern_question_file = "/workspace/data/webqa/WebQA_data_first_release/WebQA_train_val.json"
    pattern_env = {
        **common_env,
        "PATTERN_JSON_FILE_PATH": pattern_question_file,
        "PATTERN_CACHE_DIR": str(pattern_cache),
    }
    rc = run_cmd([py, "pattern.py"], pattern_env, base_dir, "Pattern extraction")
    results["phase2_pattern"] = rc == 0
    if rc != 0:
        print("[WARNING] Phase 2 failed, continuing with available outputs")
    n_patterns = len(list(pattern_cache.glob("*.json"))) if pattern_cache.exists() else 0
    print(f"[INFO] Pattern cache: {n_patterns} files")

    # ── Phase 3: Knowledge extraction ──
    print(f"\n{'='*60}\n[PHASE 3] Knowledge extraction\n{'='*60}")
    extraction_cache = result_dir / "phase3_extraction_cache"
    extraction_env = {
        **common_env,
        "EXTRACTION_QUESTION_FILE": str(q_file),
        "EXTRACTION_TEXT_FILE": str(t_file),
        "EXTRACTION_PATTERN_CACHE_DIR": str(pattern_cache),
        "EXTRACTION_CACHE_DIR": str(extraction_cache),
    }
    rc = run_cmd([py, "extraction.py"], extraction_env, base_dir, "Knowledge extraction")
    results["phase3_extraction"] = rc == 0
    if rc != 0:
        print("[WARNING] Phase 3 failed, continuing with available outputs")
    n_extractions = len(list(extraction_cache.glob("*.json"))) if extraction_cache.exists() else 0
    print(f"[INFO] Extraction cache: {n_extractions} files")

    # ── Phase 4: Graph construction ──
    print(f"\n{'='*60}\n[PHASE 4] Graph construction\n{'='*60}")
    graph_dir = result_dir / "phase4_graphs_out"
    construct_env = {
        **common_env,
        "CONSTRUCT_QUESTION_FILE": str(q_file),
        "CONSTRUCT_TABLE_FILE": str(slice_dir / "webqa_tables.jsonl"),
        "CONSTRUCT_IMAGE_FILE": str(slice_dir / "webqa_images.jsonl"),
        "CONSTRUCT_TEXT_FILE": str(t_file),
        "CONSTRUCT_EXTRACTION_CACHE": str(extraction_cache),
        "CONSTRUCT_OUTPUT_GRAPH_DIR": str(graph_dir),
        "CONSTRUCT_WEBQA_SLICE_DIR": str(slice_dir),
    }
    rc = run_cmd([py, "construct.py"], construct_env, base_dir, "Graph construction")
    results["phase4_construct"] = rc == 0
    n_graphs = len(list(graph_dir.glob("*.graphml"))) if graph_dir.exists() else 0
    print(f"[INFO] Graphs: {n_graphs} GraphML files")

    # ── Phase 5: Inference ──
    print(f"\n{'='*60}\n[PHASE 5] Inference\n{'='*60}")
    phase5_dir = result_dir / "phase5_inference"
    phase5_dir.mkdir(parents=True, exist_ok=True)
    pred_json = phase5_dir / "predictions.json"
    infer_dry = "1" if args.dry_run else "0"

    inference_env = {
        **common_env,
        "MMGRAPHRAG_RUN_ID": run_id,
        "INFERENCE_GRAPH_DIR": str(graph_dir),
        "INFERENCE_QUESTION_FILE": str(q_file),
        "INFERENCE_OUTPUT_JSON": str(pred_json),
        "INFERENCE_MAX_QUESTIONS": str(n_queries),
        "INFERENCE_DRY_RUN": infer_dry,
        "INFERENCE_COLEMBED_RETRIEVAL": os.environ.get("INFERENCE_COLEMBED_RETRIEVAL", "1"),
        "INFERENCE_WEBQA_SLICE_DIR": str(slice_dir),
        "WEBQA_IMGS_DIR": (
            SHARD14_TOY_IMGS_DIR if args.toy else
            os.environ.get("WEBQA_IMGS_DIR", str(
                _LOCAL_DATA / "WebQA_data_first_release" / "imgs"
                if (_LOCAL_DATA / "WebQA_data_first_release" / "imgs").is_dir()
                else Path("/workspace/data/webqa/WebQA_data_first_release/imgs")
            ))
        ),
    }
    rc = run_cmd([py, "inference.py"], inference_env, base_dir, "Inference")
    results["phase5_inference"] = rc == 0

    skip_eval = rc != 0 or not pred_json.is_file()
    if rc != 0:
        print("[WARNING] Phase 5 failed; skipping evaluation")

    # ── Phase 6: QA evaluation ──
    if not skip_eval:
        print(f"\n{'='*60}\n[PHASE 6] QA evaluation\n{'='*60}")
        report_json = phase5_dir / "evaluation_report.json"
        eval_env = {**common_env, "MMGRAPHRAG_RUN_ID": run_id}
        rc = run_cmd(
            [
                py, "eval/evaluate_webqa_qa.py",
                "--predictions", str(pred_json),
                "--gold_jsonl", str(q_file),
                "--report_json", str(report_json),
                "--split_label", "val",
            ],
            eval_env, base_dir, "QA evaluation",
        )
        results["phase6_eval"] = rc == 0
        if rc != 0:
            print("[WARNING] Phase 6 evaluation failed")
    else:
        results["phase6_eval"] = False

    # ── Summary ──
    print(f"\n{'='*60}")
    print("[SUMMARY] Pipeline Results")
    print(f"{'='*60}")
    all_passed = all(results.values())
    for phase, passed in results.items():
        print(f"  {phase}: {'PASS' if passed else 'FAIL'}")

    print(f"\nOutputs:")
    print(f"  Questions exported: {n_exported}")
    print(f"  Patterns generated: {n_patterns}")
    print(f"  Extractions generated: {n_extractions}")
    print(f"  Graphs constructed: {n_graphs}")
    print(f"\nResult directory: {result_dir}")

    if all_passed and n_graphs >= n_queries:
        print(f"\n[SUCCESS] {n_queries}-query pipeline test completed successfully!")
        return 0
    elif all_passed:
        print(f"\n[WARNING] Pipeline completed but fewer graphs than expected ({n_graphs} < {n_queries})")
        return 0
    else:
        print(f"\n[FAILURE] Some phases failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
