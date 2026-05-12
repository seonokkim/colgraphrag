#!/usr/bin/env python
"""
End-to-end pipeline test: pattern → extraction → construct → inference → (eval).

**Datasets**
- ``--dataset webqa`` (default): toy shard or ``export_webqa_slice``; Phase 6 uses
  ``eval/evaluate_webqa_qa.py``.
- ``--dataset mmqa``: ``export_mmqa_slice`` or ``--mmqa-slice-dir``; Phase 6 uses
  ``eval/evaluate_multimodal_qa.py`` (delegates to ``evaluate_webqa_qa.py``; list EM/F1;
  stratification via ``metadata.modalities``).

**Data roots**
- WebQA: env ``PIPELINE_WEBQA_DATA_DIR`` or ``--webqa-data-dir`` (default ``data/webqa`` under repo).
- MMQA export: env ``MMQA_DATA_DIR`` or ``--mmqa-data-dir`` (default ``data/multimodalqa/dataset``; parent ``data/multimodalqa`` is accepted if it contains ``dataset/`` with MMQA files).

**Run IDs**
- WebQA: ``webqa/<YYYYMMDD_HHMMSS>_…`` under ``result/webqa/``.
- MMQA: ``multimodalqa/<YYYYMMDD_HHMMSS>_…`` under ``result/multimodalqa/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
_repo_root_str = str(_BASE_DIR)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from util.pipeline_session_log import TeeTextStream, new_session_log_path  # noqa: E402
from util.llm_defaults import DEFAULT_GEMMA4_E4B_IT_MODEL_PATH  # noqa: E402
from util.result_layout import (  # noqa: E402
    multimodalqa_stamped_run_id,
    webqa_stamped_run_id,
)


def _resolve_webqa_data_root(base_dir: Path, cli_override: str) -> Path:
    if cli_override.strip():
        p = Path(cli_override).expanduser()
        return p.resolve() if p.is_absolute() else (base_dir / p).resolve()
    env = os.getenv("PIPELINE_WEBQA_DATA_DIR", "").strip()
    if env:
        p = Path(env).expanduser()
        return p.resolve() if p.is_absolute() else (base_dir / p).resolve()
    return (base_dir / "data" / "webqa").resolve()


def _resolve_mmqa_data_dir(base_dir: Path, cli_override: str) -> Path:
    """Directory containing ``MMQA_<split>_n*.jsonl`` (accepts ``data/multimodalqa`` → ``dataset``)."""

    def _has_mmqa_questions(p: Path) -> bool:
        return p.is_dir() and any(p.glob("MMQA_dev_n*.jsonl"))

    if cli_override.strip():
        p = Path(cli_override).expanduser()
        root = p.resolve() if p.is_absolute() else (base_dir / p).resolve()
        if _has_mmqa_questions(root):
            return root
        nested = root / "dataset"
        if _has_mmqa_questions(nested):
            return nested
        return root
    env = os.getenv("MMQA_DATA_DIR", "").strip()
    if env:
        p = Path(env).expanduser()
        root = p.resolve() if p.is_absolute() else (base_dir / p).resolve()
        if _has_mmqa_questions(root):
            return root
        nested = root / "dataset"
        if _has_mmqa_questions(nested):
            return nested
        return root
    return (base_dir / "data" / "multimodalqa" / "dataset").resolve()


def _shard14_toy_slice(webqa_root: Path) -> Path:
    local = webqa_root / "webqa_shard14_toy" / "webqa_slice"
    if local.is_dir():
        return local
    return webqa_root / "WebQA_imgs_7z_chunks" / "webqa_shard14_toy" / "webqa_slice"


def _shard14_imgs(webqa_root: Path) -> Path:
    return webqa_root / "WebQA_imgs_7z_chunks" / "imgs" / "all_png" / "shard_00014"


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GraphRAG pipeline (WebQA or MultiModalQA slice)",
    )
    parser.add_argument(
        "--dataset",
        choices=("webqa", "mmqa"),
        default="webqa",
        help="Corpus profile (sets MMGRAPHRAG_DATASET in subprocesses).",
    )
    parser.add_argument(
        "--n-queries", "-n", type=int, default=5,
        help="Max questions per phase (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Placeholder LLM (sets MMGRAPHRAG_STRICT_REAL=0).",
    )
    parser.add_argument(
        "--run-id", type=str, default="",
        help="MMGRAPHRAG_RUN_ID (default: stamped id under result/<webqa|multimodalqa>/)",
    )
    parser.add_argument(
        "--gemma-model-path", type=str, default="",
        help="Override GEMMA4_E4B_IT_MODEL_PATH.",
    )
    parser.add_argument(
        "--webqa-data-dir", type=str, default="",
        help="WebQA bundle root (overrides env PIPELINE_WEBQA_DATA_DIR; default data/webqa).",
    )
    parser.add_argument(
        "--toy", action="store_true", default=True,
        help="[webqa] Use shard-14 toy under webqa-data-dir (default: True).",
    )
    parser.add_argument(
        "--no-toy", action="store_true",
        help="[webqa] Run export_webqa_slice.py instead of copying toy slice.",
    )
    parser.add_argument(
        "--webqa-slice-dir", type=str, default="",
        help="[webqa] Pre-built webqa_slice/ directory.",
    )
    parser.add_argument(
        "--mmqa-slice-dir", type=str, default="",
        help="[mmqa] Pre-built mmqa_slice/ directory (skip export_mmqa_slice).",
    )
    parser.add_argument(
        "--mmqa-data-dir", type=str, default="",
        help="[mmqa] Override MMQA_DATA_DIR for export_mmqa_slice.py.",
    )
    args = parser.parse_args()

    if args.no_toy:
        args.toy = False

    webqa_root = _resolve_webqa_data_root(_BASE_DIR, args.webqa_data_dir)

    if args.dataset == "mmqa":
        if args.webqa_slice_dir:
            print("[ERROR] Use --mmqa-slice-dir for mmqa, not --webqa-slice-dir.")
            return 1
        args.toy = False
        prebuilt_slice_dir = (
            Path(args.mmqa_slice_dir.strip()) if args.mmqa_slice_dir.strip() else None
        )
        run_id = args.run_id.strip() or multimodalqa_stamped_run_id(f"n{args.n_queries}")
    else:
        if args.mmqa_slice_dir:
            print("[ERROR] --mmqa-slice-dir is only for --dataset mmqa.")
            return 1
        if args.toy and args.webqa_slice_dir:
            print("[ERROR] Choose either --toy or --webqa-slice-dir, not both.")
            return 1
        prebuilt_slice_dir = None
        if args.toy:
            prebuilt_slice_dir = _shard14_toy_slice(webqa_root)
            if not prebuilt_slice_dir.is_dir():
                print(f"[ERROR] Toy dataset not found: {prebuilt_slice_dir}")
                return 1
        elif args.webqa_slice_dir.strip():
            prebuilt_slice_dir = Path(args.webqa_slice_dir)
            if not prebuilt_slice_dir.is_dir():
                print(f"[ERROR] webqa_slice_dir not found: {args.webqa_slice_dir}")
                return 1
        run_id = args.run_id.strip() or webqa_stamped_run_id("pipeline_test")

    base_dir = _BASE_DIR
    result_dir = base_dir / "result" / run_id

    _safe_run_id = "".join(c if c not in '\\/:' else "_" for c in run_id.replace(os.sep, "_"))
    session_log_path = new_session_log_path(base_dir, f"test_pipeline_{_safe_run_id}")
    session_log_file = session_log_path.open("w", encoding="utf-8")
    _old_stdout, _old_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeTextStream(_old_stdout, session_log_file)
    sys.stderr = TeeTextStream(_old_stderr, session_log_file)

    try:
        return _main_run(
            args=args,
            base_dir=base_dir,
            result_dir=result_dir,
            run_id=run_id,
            session_log_path=session_log_path,
            webqa_root=webqa_root,
            mmqa_prebuilt=prebuilt_slice_dir if args.dataset == "mmqa" else None,
            webqa_prebuilt=prebuilt_slice_dir if args.dataset == "webqa" else None,
        )
    finally:
        sys.stdout = _old_stdout
        sys.stderr = _old_stderr
        session_log_file.close()


def _main_run(
    *,
    args: argparse.Namespace,
    base_dir: Path,
    result_dir: Path,
    run_id: str,
    session_log_path: Path,
    webqa_root: Path,
    mmqa_prebuilt: Path | None,
    webqa_prebuilt: Path | None,
) -> int:

    py = sys.executable
    n_queries = args.n_queries
    dry_run = "1" if args.dry_run else "0"
    is_mmqa = args.dataset == "mmqa"
    slice_leaf = "mmqa_slice" if is_mmqa else "webqa_slice"
    ds_prefix = "mmqa" if is_mmqa else "webqa"

    print(f"{'='*60}")
    print("GraphRAG pipeline test")
    print(f"{'='*60}")
    print(f"Session log: {session_log_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Run ID: {run_id}")
    print(f"Result dir: {result_dir}")
    print(f"Queries (cap): {n_queries}")
    if is_mmqa:
        print(f"MMQA data dir: {_resolve_mmqa_data_dir(base_dir, args.mmqa_data_dir)}")
        print(f"MMQA: prebuilt slice={mmqa_prebuilt or '(export_mmqa_slice)'}")
    elif args.toy:
        print(f"WebQA data root: {webqa_root}")
        print("Data: shard-14 toy slice")
    elif webqa_prebuilt:
        print(f"WebQA: pre-built {webqa_prebuilt}")
    else:
        print("WebQA: export_webqa_slice.py")
    if args.dry_run:
        print("Mode: dry-run (MMGRAPHRAG_STRICT_REAL=0 in children)")
    else:
        print("Mode: real Gemma + ColEmbed (MMGRAPHRAG_STRICT_REAL=1)")
    print(f"{'='*60}")

    import torch
    if not args.dry_run:
        if not torch.cuda.is_available():
            print(
                "[ERROR] CUDA required for real pipeline. "
                "Use --dry-run only for wiring checks.",
            )
            return 1
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

    gemma_path = (args.gemma_model_path or "").strip()
    if not gemma_path:
        gemma_path = os.environ.get("GEMMA4_E4B_IT_MODEL_PATH", "").strip() or DEFAULT_GEMMA4_E4B_IT_MODEL_PATH

    webqa_data_root_str = str(webqa_root) if webqa_root.is_dir() else str(base_dir / "data" / "webqa")

    common_env: dict[str, str] = {
        "PYTHONPATH": str(base_dir),
        "MMGRAPHRAG_RUN_ID": run_id,
        "MMGRAPHRAG_DATASET": "mmqa" if is_mmqa else "webqa",
        "WEBQA_RUN_PROFILE": "val_n100",
        "WEBQA_DATA_ROOT": webqa_data_root_str,
        "PATTERN_MAX_SAMPLES": str(n_queries),
        "WEBQA_EXPORT_MAX": str(n_queries),
        "EXTRACTION_MAX_QUESTIONS": str(n_queries),
        "CONSTRUCT_MAX_QUESTIONS": str(n_queries),
        "INFERENCE_MAX_QUESTIONS": str(n_queries),
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
            str(base_dir / "models" / "retriever" / "llama-nemotron-colembed-vl-3b-v2"),
        ),
    )
    if args.dry_run:
        common_env["MMGRAPHRAG_STRICT_REAL"] = "0"

    if is_mmqa:
        common_env["MMQA_DATA_DIR"] = str(_resolve_mmqa_data_dir(base_dir, args.mmqa_data_dir))

    results: dict[str, bool] = {}

    # ── Phase 0 ──
    slice_dir = result_dir / slice_leaf
    if is_mmqa:
        print(f"\n{'='*60}\n[PHASE 0] MMQA slice → {slice_dir.name}\n{'='*60}")
        if mmqa_prebuilt and mmqa_prebuilt.is_dir():
            slice_dir.parent.mkdir(parents=True, exist_ok=True)
            if slice_dir.exists():
                shutil.rmtree(slice_dir)
            shutil.copytree(mmqa_prebuilt, slice_dir)
            results["phase0_export"] = True
            print(f"[OK] Copied {mmqa_prebuilt} → {slice_dir}")
        else:
            export_env = {**common_env, "MMQA_SLICE_DIR": str(slice_dir)}
            rc = run_cmd([py, "export_mmqa_slice.py"], export_env, base_dir, "Export MMQA slice")
            results["phase0_export"] = rc == 0
            if rc != 0:
                print("[ABORT] Phase 0 failed")
                return 1
    else:
        print(f"\n{'='*60}\n[PHASE 0] WebQA slice → {slice_dir.name}\n{'='*60}")
        if webqa_prebuilt and webqa_prebuilt.is_dir():
            slice_dir.parent.mkdir(parents=True, exist_ok=True)
            if slice_dir.exists():
                shutil.rmtree(slice_dir)
            shutil.copytree(webqa_prebuilt, slice_dir)
            results["phase0_export"] = True
            print(f"[OK] Copied pre-built slice to {slice_dir}")
        else:
            export_env = {**common_env, "WEBQA_SLICE_DIR": str(slice_dir)}
            rc = run_cmd([py, "export_webqa_slice.py"], export_env, base_dir, "Export WebQA slice")
            results["phase0_export"] = rc == 0
            if rc != 0:
                print("[ABORT] Phase 0 failed")
                return 1

    q_file = slice_dir / f"{ds_prefix}_questions.jsonl"
    t_file = slice_dir / f"{ds_prefix}_texts.jsonl"
    if not q_file.exists() or not t_file.exists():
        print(f"[ERROR] Missing slice files: {q_file.exists()=}, {t_file.exists()=}")
        return 1

    with open(q_file, "r") as f:
        n_exported = sum(1 for _ in f)
    print(f"[INFO] Slice: {n_exported} questions in {q_file}")

    # ── Phase 2 ──
    print(f"\n{'='*60}\n[PHASE 2] Pattern extraction\n{'='*60}")
    pattern_cache = result_dir / "phase2_pattern_cache"
    # Same questions as extraction/construct/inference so pattern count matches the slice.
    pattern_question_file = str(q_file)
    pattern_env = {
        **common_env,
        "PATTERN_JSON_FILE_PATH": pattern_question_file,
        "PATTERN_CACHE_DIR": str(pattern_cache),
    }
    rc = run_cmd([py, "pattern.py"], pattern_env, base_dir, "Pattern extraction")
    results["phase2_pattern"] = rc == 0
    if rc != 0:
        print("[WARNING] Phase 2 failed, continuing")
    n_patterns = len(list(pattern_cache.glob("*.json"))) if pattern_cache.exists() else 0
    print(f"[INFO] Pattern cache: {n_patterns} files")

    # ── Phase 3 ──
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
        print("[WARNING] Phase 3 failed, continuing")
    n_extractions = len(list(extraction_cache.glob("*.json"))) if extraction_cache.exists() else 0
    print(f"[INFO] Extraction cache: {n_extractions} files")

    # ── Phase 4 ──
    print(f"\n{'='*60}\n[PHASE 4] Graph construction\n{'='*60}")
    graph_dir = result_dir / "phase4_graphs_out"
    construct_env = {
        **common_env,
        "CONSTRUCT_SLICE_DIR": str(slice_dir),
        "CONSTRUCT_QUESTION_FILE": str(q_file),
        "CONSTRUCT_TABLE_FILE": str(slice_dir / f"{ds_prefix}_tables.jsonl"),
        "CONSTRUCT_IMAGE_FILE": str(slice_dir / f"{ds_prefix}_images.jsonl"),
        "CONSTRUCT_TEXT_FILE": str(t_file),
        "CONSTRUCT_EXTRACTION_CACHE": str(extraction_cache),
        "CONSTRUCT_OUTPUT_GRAPH_DIR": str(graph_dir),
        "CONSTRUCT_WEBQA_SLICE_DIR": str(slice_dir),
    }
    rc = run_cmd([py, "construct.py"], construct_env, base_dir, "Graph construction")
    results["phase4_construct"] = rc == 0
    n_graphs = len(list(graph_dir.glob("*.graphml"))) if graph_dir.exists() else 0
    print(f"[INFO] Graphs: {n_graphs} GraphML files")

    # ── Phase 5 ──
    print(f"\n{'='*60}\n[PHASE 5] Inference\n{'='*60}")
    phase5_dir = result_dir / "phase5_inference"
    phase5_dir.mkdir(parents=True, exist_ok=True)
    pred_json = phase5_dir / "predictions.json"
    infer_dry = "1" if args.dry_run else "0"

    imgs_root = _shard14_imgs(webqa_root)
    default_webqa_imgs = str(
        webqa_root / "WebQA_data_first_release" / "imgs"
        if (webqa_root / "WebQA_data_first_release" / "imgs").is_dir()
        else base_dir / "data" / "webqa" / "WebQA_data_first_release" / "imgs"
    )
    mmqa_imgs = base_dir / "data" / "multimodalqa" / "final_dataset_images"

    inference_env = {
        **common_env,
        "INFERENCE_GRAPH_DIR": str(graph_dir),
        "INFERENCE_QUESTION_FILE": str(q_file),
        "INFERENCE_OUTPUT_JSON": str(pred_json),
        "INFERENCE_MAX_QUESTIONS": str(n_queries),
        "INFERENCE_DRY_RUN": infer_dry,
        "INFERENCE_COLEMBED_RETRIEVAL": os.environ.get("INFERENCE_COLEMBED_RETRIEVAL", "1"),
        "INFERENCE_SLICE_DIR": str(slice_dir),
        "INFERENCE_WEBQA_SLICE_DIR": str(slice_dir),
    }
    if is_mmqa:
        inference_env["MMQA_IMAGES_DIR"] = str(
            mmqa_imgs if mmqa_imgs.is_dir() else Path(os.getenv("MMQA_IMAGES_DIR", str(mmqa_imgs)))
        )
    else:
        inference_env["WEBQA_IMGS_DIR"] = (
            str(imgs_root)
            if imgs_root.is_dir() and args.toy
            else os.environ.get("WEBQA_IMGS_DIR", default_webqa_imgs)
        )

    rc = run_cmd([py, "inference.py"], inference_env, base_dir, "Inference")
    results["phase5_inference"] = rc == 0

    skip_eval = rc != 0 or not pred_json.is_file()
    if rc != 0:
        print("[WARNING] Phase 5 failed; skipping evaluation")

    # ── Phase 6 ──
    if not skip_eval:
        eval_label = (
            "WebQA (evaluate_webqa_qa)"
            if not is_mmqa
            else "MultiModalQA (evaluate_multimodal_qa → evaluate_webqa_qa)"
        )
        print(f"\n{'='*60}\n[PHASE 6] QA evaluation ({eval_label})\n{'='*60}")
        report_json = phase5_dir / "evaluation_report.json"
        eval_env = {**common_env, "MMGRAPHRAG_RUN_ID": run_id}
        split_label = "val" if not is_mmqa else f"mmqa_dev_n{n_queries}"
        eval_script = (
            "eval/evaluate_multimodal_qa.py" if is_mmqa else "eval/evaluate_webqa_qa.py"
        )
        rc = run_cmd(
            [
                py,
                eval_script,
                "--predictions",
                str(pred_json),
                "--gold_jsonl",
                str(q_file),
                "--report_json",
                str(report_json),
                "--split_label",
                split_label,
            ],
            eval_env,
            base_dir,
            "QA evaluation",
        )
        results["phase6_eval"] = rc == 0
        if rc != 0:
            print("[WARNING] Phase 6 evaluation failed")
    else:
        results["phase6_eval"] = False

    print(f"\n{'='*60}\n[SUMMARY]\n{'='*60}")
    all_passed = all(results.values())
    for phase, passed in results.items():
        print(f"  {phase}: {'PASS' if passed else 'FAIL'}")

    print(f"\nOutputs:")
    print(f"  Questions in slice: {n_exported}")
    print(f"  Patterns: {n_patterns} | Extractions: {n_extractions} | Graphs: {n_graphs}")
    print(f"  Result directory: {result_dir}")

    if all_passed and n_graphs >= n_queries:
        print(f"\n[SUCCESS] {n_queries}-query pipeline completed.")
        return 0
    if all_passed:
        print(f"\n[WARNING] Fewer graphs than query cap ({n_graphs} vs {n_queries})")
        return 0
    print(f"\n[FAILURE] Some phases failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
