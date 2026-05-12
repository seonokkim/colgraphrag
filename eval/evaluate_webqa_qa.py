"""
WebQA QA evaluation: list EM / list F1 for Unimodal, Multimodal, and All.

Scoring uses `eval/webqa_qa_scoring.py` (self-contained).
Gold lines come from `export_webqa_slice.py` (`webqa_questions.jsonl` with `answers` + metadata.webqa).

MMQA pipelines call the thin wrapper `evaluate_multimodal_qa.py`, which delegates here unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from util.run_id import default_stamp
from util.webqa_fluency import (
    active_backend as _fl_active_backend,
    active_model_name as _fl_active_model,
    fluency_score,
)
from util.webqa_gold_normalize import normalize_webqa_answer_strings
from util.webqa_metrics_approx import qcate_keyword_tokens, webqa_metrics_approx

from eval.evaluate_retrieval import (
    _load_predictions as _load_retrieval_predictions,
    evaluate_retrieval_stratified,
)


def _load_evaluate_predictions():
    scoring_path = Path(__file__).resolve().parent / "webqa_qa_scoring.py"
    if not scoring_path.is_file():
        raise ImportError(f"webqa_qa_scoring.py not found at {scoring_path}")
    spec = importlib.util.spec_from_file_location("webqa_qa_scoring", scoring_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {scoring_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate_predictions


evaluate_predictions = _load_evaluate_predictions()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stratum_for_example(ex: dict[str, Any]) -> str:
    md = ex.get("metadata") or {}
    w = md.get("webqa") or {}
    if isinstance(w.get("webqa_stratum"), str) and w["webqa_stratum"] in (
        "Unimodal",
        "Multimodal",
    ):
        return w["webqa_stratum"]
    if isinstance(w.get("multimodal"), bool):
        return "Multimodal" if w["multimodal"] else "Unimodal"
    # MultiModalQA export: metadata.modalities is a list of involved modalities.
    mods = md.get("modalities")
    if isinstance(mods, list) and mods:
        return "Multimodal" if len(mods) > 1 else "Unimodal"
    return "Unimodal"


def _qcate_for_example(ex: dict[str, Any]) -> str:
    md = ex.get("metadata") or {}
    w = md.get("webqa") or {}
    q = w.get("Qcate") or ex.get("Qcate")
    if isinstance(q, str) and q:
        return str(q)
    mt = md.get("type")
    if isinstance(mt, str) and mt:
        return mt
    return "text"


def _build_gold_and_types(
    examples: list[dict[str, Any]],
) -> tuple[
    dict[str, list[list[str]]],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    gold_answers: dict[str, list[list[str]]] = {}
    example_types: dict[str, str] = {}
    qcates: dict[str, str] = {}
    gold_text: dict[str, str] = {}
    keywords_a: dict[str, str] = {}
    for example in examples:
        qid = str(example.get("qid") or example.get("Guid") or "")
        ans_list = example.get("answers") or []
        if not ans_list:
            raw_a = example.get("A")
            if raw_a:
                if isinstance(raw_a, list):
                    ans_list = [{"answer": a} for a in raw_a]
                elif isinstance(raw_a, str):
                    ans_list = [{"answer": raw_a}]
        if not ans_list:
            raise ValueError(
                f"Gold line for qid={qid} has no `answers` or `A`; re-run export_webqa_slice.py"
            )
        gold_answer: list[str] = []
        for item in ans_list:
            gold_answer.extend(
                normalize_webqa_answer_strings(item.get("answer", ""))
            )
        if not gold_answer:
            gold_answer = [""]
        gold_answers[qid] = [gold_answer]
        example_types[qid] = _stratum_for_example(example)
        qcates[qid] = _qcate_for_example(example)
        gold_text[qid] = " ".join(s for s in gold_answer if s)
        kw_a = example.get("Keywords_A") or ""
        if kw_a:
            keywords_a[qid] = str(kw_a)
    return gold_answers, example_types, qcates, gold_text, keywords_a


def _load_predictions(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "predictions" in raw and isinstance(
        raw["predictions"], dict
    ):
        raw = raw["predictions"]
    return {
        str(k): str(v) if not isinstance(v, (dict, list)) else json.dumps(v)
        for k, v in raw.items()
    }


def _prediction_diagnostics(predictions: dict[str, str]) -> dict[str, Any]:
    """Surface degenerate prediction files (dry-run placeholders) in eval JSON + stderr."""
    n = len(predictions)
    vals = [str(v).strip() for v in predictions.values() if str(v).strip()]
    if n == 0:
        return {
            "num_prediction_keys": 0,
            "warning": "empty_predictions",
            "hint": "phase5_predictions_real.json has no keys; re-run inference.py.",
        }
    unique = len(set(vals))
    lengths = [len(v) for v in vals]
    ph = sum(
        1
        for v in vals
        if "placeholder" in v.lower() or v.lower() in {"unknown", "n/a", "na"}
    )
    few_unique_short = unique <= max(3, n // 25) and (max(lengths) if lengths else 0) < 150
    hint: str | None = None
    if ph > n * 0.4 or few_unique_short:
        hint = (
            "Predictions look like placeholders or very few repeated short strings. "
            "For meaningful list_em / list_f1, run inference with INFERENCE_DRY_RUN=0 "
            "and a working INFERENCE_API_URL / LLM_* so answers match WebQA-style gold text."
        )
    return {
        "num_prediction_keys": n,
        "non_empty_string_values": len(vals),
        "unique_answer_strings": unique,
        "min_answer_len": min(lengths) if lengths else 0,
        "max_answer_len": max(lengths) if lengths else 0,
        "values_placeholder_or_unknown_like": ph,
        "suspicious_few_unique_short_strings": few_unique_short,
        "hint": hint,
    }


def _score_block(list_f1: float, list_em: float) -> dict[str, float]:
    return {
        "list_f1": float(list_f1),
        "list_em": float(list_em),
        "f1": float(list_f1),
        "em": float(list_em),
    }


def _build_qcate_keyword_inputs(
    raw_gold: dict[str, list[list[str]]],
    raw_preds: dict[str, str],
    qcates: dict[str, str],
    keywords_a: dict[str, str] | None = None,
) -> tuple[dict[str, list[list[str]]], dict[str, str]]:
    """Project both gold and prediction onto the Qcate keyword bag.

    WebQA gold answers are full natural-language sentences, so MMQA-style
    ``list_em`` / ``list_f1`` can never fire against them. Filtering both sides
    with :func:`qcate_keyword_tokens` reduces e.g.
    ``'"No, a Minnetonka Rhododendron flower does not ..."'`` + ``Qcate=YesNo``
    to ``"no"`` on the gold side and the model's ``"No"`` to ``"no"`` on the
    prediction side, so EM/F1 report the overlap the WebQA leaderboard scores
    against its hidden ``Keywords_answer`` (see reference/code/WebQA-main/
    WebQA-main/eval_webqa.md).

    When ``keywords_a`` is available (from TSV ``Keywords_A`` field), it is used
    directly as the gold keyword reference for keyword-filtered categories
    (YesNo, color, shape, number).
    """
    _keyword_categories = {"YesNo", "color", "shape", "number"}
    keywords_a = keywords_a or {}
    gold_kw: dict[str, list[list[str]]] = {}
    for qid, ref_wrap in raw_gold.items():
        cat = qcates.get(qid, "text")
        if cat in _keyword_categories and qid in keywords_a:
            tokens = qcate_keyword_tokens(keywords_a[qid], Qcate=cat)
        else:
            tokens = []
            for ref in ref_wrap:
                for ans in ref:
                    tokens.extend(qcate_keyword_tokens(ans, Qcate=cat))
        gold_kw[qid] = [[" ".join(tokens) if tokens else ""]]
    pred_kw: dict[str, str] = {}
    for qid, pred in raw_preds.items():
        cat = qcates.get(qid, "text")
        tokens = qcate_keyword_tokens(pred, Qcate=cat)
        pred_kw[qid] = " ".join(tokens) if tokens else ""
    return gold_kw, pred_kw


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(
        suffix=".json.tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _resolve_retrieval_rankings(rankings_json: Path | None, run_id: str) -> Path | None:
    """Honor --retrieval_rankings_json; else look under result/<run_id>/.

    Checks both the canonical name ``retrieval_rankings.json`` and the legacy
    pipeline name ``phase5_retrieval_rankings.json`` so older runs from
    ``build_retrieval_rankings_from_graphs.py`` resolve without a copy.
    """
    if rankings_json is not None:
        return rankings_json if rankings_json.is_file() else None
    if not run_id:
        return None
    base = _REPO_ROOT / "result" / run_id
    for name in (
        "phase5_inference/predictions_retrieval.json",
        "retrieval_rankings.json",
        "phase5_retrieval_rankings.json",
    ):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _build_retrieval_summary(
    rankings_json: Path | None,
    gold_jsonl: Path,
    ks_csv: str,
    k_gen: int,
    run_id: str,
) -> dict[str, Any]:
    """Compute retrieval IR metrics (hit/recall/MRR/nDCG/MAP/retrieval_f1 @ k).

    Emits a structured block with overall / stratum / by_Qcate / by_modality
    sub-blocks plus a ``k_gen`` scalar for the generator operating point.
    Returns ``{"status": "..."}`` when inputs are missing instead of raising,
    so schema 1.6 is forward-compatible with runs that skip retrieval eval.
    """
    resolved = _resolve_retrieval_rankings(rankings_json, run_id)
    if resolved is None:
        return {
            "status": "missing_rankings_json",
            "hint": (
                "Provide --retrieval_rankings_json or place retrieval_rankings.json under "
                "result/<MMGRAPHRAG_RUN_ID>/ to populate this block."
            ),
        }
    try:
        ks = sorted({int(x.strip()) for x in ks_csv.split(",") if x.strip() and int(x.strip()) > 0})
    except ValueError:
        return {"status": "invalid_ks", "hint": f"Could not parse --retrieval_ks='{ks_csv}'."}
    if not ks:
        return {"status": "invalid_ks", "hint": "No positive integers in --retrieval_ks."}
    if k_gen not in ks:
        ks = sorted(set(ks + [k_gen]))

    predictions = _load_retrieval_predictions(resolved)
    result = evaluate_retrieval_stratified(predictions, gold_jsonl, None, ks)
    result["k_gen"] = k_gen
    result["inputs"] = {
        "retrieval_rankings_json": str(resolved.resolve()),
        "gold_jsonl": str(gold_jsonl.resolve()),
        "ks": ks,
    }
    result["metric_definitions"] = {
        "hit@k": "At least one gold source in top-k (macro-averaged).",
        "recall@k": "#hits_in_topk / #relevant.",
        "mrr@k": "1 / rank of first hit in top-k.",
        "ndcg@k": "DCG(binary hits) / IDCG over min(|R|, k).",
        "map@k": "Truncated AP at k; denominator is min(|R|, k).",
        "retrieval_f1": (
            "Per-query set F1 over top-k vs gold (WebQA-main baseline README wire name)."
        ),
    }
    return result


def _print_retrieval_banner(retrieval: dict[str, Any]) -> None:
    if not retrieval or retrieval.get("status"):
        return
    k = retrieval.get("k_gen", 10)
    overall = retrieval.get("overall") or {}
    overall_m = (overall.get("metrics") or {}).get(f"k={k}")
    if overall_m:
        print(
            f"Retriever@{k} overall  "
            f"hit={overall_m['hit@k']:.4f} recall={overall_m['recall@k']:.4f} "
            f"mrr={overall_m['mrr@k']:.4f} ndcg={overall_m['ndcg@k']:.4f} "
            f"map={overall_m['map@k']:.4f} retF1={overall_m['retrieval_f1']:.4f}"
        )
    by_qcate = retrieval.get("by_Qcate") or {}
    if by_qcate:
        parts = []
        for cat, block in sorted(by_qcate.items()):
            m = (block.get("metrics") or {}).get(f"k={k}")
            if m:
                parts.append(f"{cat}:rec={m['recall@k']:.3f}/retF1={m['retrieval_f1']:.3f}")
        if parts:
            print(f"Retriever by_Qcate@{k}: " + " | ".join(parts))
    by_mod = retrieval.get("by_modality") or {}
    if by_mod:
        parts = []
        for mod in ("text", "image", "table"):
            block = by_mod.get(mod) or {}
            if not block.get("num_queries"):
                continue
            m = (block.get("metrics") or {}).get(f"k={k}")
            if m:
                parts.append(f"{mod}:rec={m['recall@k']:.3f}/ndcg={m['ndcg@k']:.3f}")
        if parts:
            print(f"Retriever by_modality@{k}: " + " | ".join(parts))


def _git_commit() -> str | None:
    try:
        root = _REPO_ROOT.parent
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _metric_log_prefix(args: argparse.Namespace) -> str:
    """Banner label for stdout lines (MMQA runs reuse this script with WebQA-style metrics)."""
    manual = getattr(args, "metric_log_prefix", None)
    if manual and str(manual).strip():
        return str(manual).strip()
    ds = os.getenv("MMGRAPHRAG_DATASET", "").strip().lower()
    if ds == "mmqa":
        return "MMQA"
    gold = str(args.gold_jsonl).lower()
    split_l = (args.split_label or "").lower()
    if "mmqa" in gold or "multimodalqa" in gold or split_l.startswith("mmqa"):
        return "MMQA"
    return "WebQA"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WebQA list EM/F1 with Unimodal / Multimodal / All buckets."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold_jsonl", type=Path, required=True)
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Default: result/json/<run_id>_webqa_qa_eval_report.json if MMGRAPHRAG_RUN_ID set",
    )
    parser.add_argument("--split_label", type=str, default="test")
    parser.add_argument(
        "--metric-log-prefix",
        type=str,
        default=None,
        dest="metric_log_prefix",
        metavar="LABEL",
        help=(
            "Prefix for acc_approx / leaderboard stdout lines (default: MMQA if gold path or "
            "--split_label looks like MMQA, else WebQA)."
        ),
    )
    parser.add_argument(
        "--retrieval_rankings_json",
        type=Path,
        default=None,
        help=(
            "Optional ranked-source JSON (from build_retrieval_rankings_from_graphs.py). "
            "When provided, the QA report embeds a retrieval_summary block "
            "(hit/recall/MRR/nDCG/MAP/retrieval_f1 @ k, by_Qcate, by_modality). "
            "Defaults to result/<MMGRAPHRAG_RUN_ID>/retrieval_rankings.json when that file exists."
        ),
    )
    parser.add_argument(
        "--retrieval_ks",
        type=str,
        default="1,3,5,10",
        help="Comma-separated k values for the retrieval sweep.",
    )
    parser.add_argument(
        "--retrieval_k_gen",
        type=int,
        default=10,
        help="k the answer generator actually consumes; echoed in retrieval_summary.",
    )
    args = parser.parse_args()
    log_pfx = _metric_log_prefix(args)

    if not args.predictions.is_file():
        raise SystemExit(f"Missing predictions: {args.predictions}")
    if not args.gold_jsonl.is_file():
        raise SystemExit(f"Missing gold jsonl: {args.gold_jsonl}")

    examples = read_jsonl(args.gold_jsonl)
    gold_answers, example_types, qcates, gold_text, keywords_a = _build_gold_and_types(examples)
    predictions = _load_predictions(args.predictions)
    diagnostics = _prediction_diagnostics(predictions)
    diagnostics["webqa_fluency_backend"] = _fl_active_backend()
    diagnostics["webqa_fluency_model"] = _fl_active_model()
    if diagnostics.get("hint"):
        print(diagnostics["hint"], file=sys.stderr)

    eval_scores, instance_eval_results, eval_scores_by_types = evaluate_predictions(
        predictions, gold_answers, example_types
    )

    gold_kw, pred_kw = _build_qcate_keyword_inputs(gold_answers, predictions, qcates, keywords_a)
    (
        eval_scores_kw,
        instance_eval_results_kw,
        eval_scores_by_types_kw,
    ) = evaluate_predictions(pred_kw, gold_kw, example_types)
    _, _, eval_scores_by_qcate_kw = evaluate_predictions(pred_kw, gold_kw, qcates)

    def bucket_metric(bucket: str, metric: str) -> float:
        sub = eval_scores_by_types.get(bucket)
        if not sub:
            return 0.0
        return float(sub[metric])

    def bucket_metric_kw(bucket: str, metric: str) -> float:
        sub = eval_scores_by_types_kw.get(bucket)
        if not sub:
            return 0.0
        return float(sub[metric])

    webqa_per_qid: dict[str, float] = {}
    webqa_fl_per_qid: dict[str, float] = {}
    webqa_qa_per_qid: dict[str, float] = {}
    scored_qids = [qid for qid in gold_text if qid in predictions]
    if len(scored_qids) < len(gold_text):
        print(
            f"[evaluate] scoring {len(scored_qids)} questions with predictions "
            f"(out of {len(gold_text)} gold); missing predictions are excluded "
            f"from QA-FL / QA-Acc / QA averages.",
            file=sys.stderr,
        )
    _keyword_categories = {"YesNo", "color", "shape", "number"}
    for qid in scored_qids:
        g_text_val = gold_text[qid]
        pred = predictions[qid]
        cat = qcates.get(qid, "text")
        acc_gold = keywords_a.get(qid, g_text_val) if cat in _keyword_categories else g_text_val
        acc01 = float(
            webqa_metrics_approx(pred, acc_gold, Qcate=cat)["acc_approx"]
        )
        webqa_per_qid[qid] = acc01
        fl01 = float(fluency_score(pred, g_text_val))
        webqa_fl_per_qid[qid] = fl01
        webqa_qa_per_qid[qid] = fl01 * acc01

    last_fl_error = os.environ.get("WEBQA_FLUENCY_LAST_ERROR", "")
    if last_fl_error:
        diagnostics["webqa_fluency_last_error"] = last_fl_error
        diagnostics["webqa_fluency_effective_backend"] = "rouge"
    else:
        diagnostics["webqa_fluency_effective_backend"] = _fl_active_backend()

    def mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) * 100.0 if xs else 0.0

    stratum_members: dict[str, list[str]] = {"Unimodal": [], "Multimodal": []}
    for qid, st in example_types.items():
        if qid in webqa_per_qid:
            stratum_members.setdefault(st, []).append(qid)
    qcate_members: dict[str, list[str]] = {}
    for qid, c in qcates.items():
        if qid in webqa_per_qid:
            qcate_members.setdefault(c, []).append(qid)

    all_qids = list(webqa_per_qid.keys())
    webqa_acc = {
        "All": mean([webqa_per_qid[q] for q in all_qids]),
        "Unimodal": mean([webqa_per_qid[q] for q in stratum_members.get("Unimodal", [])]),
        "Multimodal": mean([webqa_per_qid[q] for q in stratum_members.get("Multimodal", [])]),
        "by_Qcate": {
            cat: mean([webqa_per_qid[q] for q in qids])
            for cat, qids in sorted(qcate_members.items())
        },
    }
    webqa_fl = {
        "All": mean([webqa_fl_per_qid[q] for q in all_qids]),
        "Unimodal": mean(
            [webqa_fl_per_qid[q] for q in stratum_members.get("Unimodal", [])]
        ),
        "Multimodal": mean(
            [webqa_fl_per_qid[q] for q in stratum_members.get("Multimodal", [])]
        ),
        "by_Qcate": {
            cat: mean([webqa_fl_per_qid[q] for q in qids])
            for cat, qids in sorted(qcate_members.items())
        },
    }
    webqa_qa = {
        "All": mean([webqa_qa_per_qid[q] for q in all_qids]),
        "Unimodal": mean(
            [webqa_qa_per_qid[q] for q in stratum_members.get("Unimodal", [])]
        ),
        "Multimodal": mean(
            [webqa_qa_per_qid[q] for q in stratum_members.get("Multimodal", [])]
        ),
        "by_Qcate": {
            cat: mean([webqa_qa_per_qid[q] for q in qids])
            for cat, qids in sorted(qcate_members.items())
        },
    }

    scores = {
        "All": _score_block(float(eval_scores["list_f1"]), float(eval_scores["list_em"])),
        "Unimodal": _score_block(
            bucket_metric("Unimodal", "list_f1"),
            bucket_metric("Unimodal", "list_em"),
        ),
        "Multimodal": _score_block(
            bucket_metric("Multimodal", "list_f1"),
            bucket_metric("Multimodal", "list_em"),
        ),
    }
    for bucket in ("All", "Unimodal", "Multimodal"):
        # Backward-compat keys (preserve older schema 1.4 output so
        # plot scripts / post-mortem docs do not break).
        scores[bucket]["webqa_acc_approx"] = webqa_acc[bucket]
        scores[bucket]["webqa_qa_fl"] = webqa_fl[bucket]
        scores[bucket]["webqa_qa"] = webqa_qa[bucket]
        # ACL-25 Query-Driven Multimodal GraphRAG (§5.1, Table 2) + WebQA
        # EvalAI server wire names (Call_WebQA_eval_server_locally.py).
        # These are the primary leaderboard metrics the paper reports.
        scores[bucket]["qa_fl"] = webqa_fl[bucket]
        scores[bucket]["qa_acc"] = webqa_acc[bucket]
        scores[bucket]["qa"] = webqa_qa[bucket]
    scores["by_Qcate_webqa_acc_approx"] = webqa_acc["by_Qcate"]
    scores["by_Qcate_webqa_qa_fl"] = webqa_fl["by_Qcate"]
    scores["by_Qcate_webqa_qa"] = webqa_qa["by_Qcate"]
    scores["by_Qcate_qa_fl"] = webqa_fl["by_Qcate"]
    scores["by_Qcate_qa_acc"] = webqa_acc["by_Qcate"]
    scores["by_Qcate_qa"] = webqa_qa["by_Qcate"]

    scores["All"]["list_f1_keyword"] = float(eval_scores_kw["list_f1"])
    scores["All"]["list_em_keyword"] = float(eval_scores_kw["list_em"])
    scores["Unimodal"]["list_f1_keyword"] = bucket_metric_kw("Unimodal", "list_f1")
    scores["Unimodal"]["list_em_keyword"] = bucket_metric_kw("Unimodal", "list_em")
    scores["Multimodal"]["list_f1_keyword"] = bucket_metric_kw("Multimodal", "list_f1")
    scores["Multimodal"]["list_em_keyword"] = bucket_metric_kw("Multimodal", "list_em")
    scores["by_Qcate_list_em_keyword"] = {
        cat: float(v["list_em"]) for cat, v in sorted(eval_scores_by_qcate_kw.items())
    }
    scores["by_Qcate_list_f1_keyword"] = {
        cat: float(v["list_f1"]) for cat, v in sorted(eval_scores_by_qcate_kw.items())
    }

    counts: dict[str, int] = {"All": len(gold_answers), "scored": len(scored_qids), "Unimodal": 0, "Multimodal": 0}
    for qid, bucket in example_types.items():
        if qid in gold_answers:
            counts[bucket] = counts.get(bucket, 0) + 1
    counts["by_Qcate"] = {cat: len(qids) for cat, qids in sorted(qcate_members.items())}

    run_id = os.getenv("MMGRAPHRAG_RUN_ID", "").strip()

    retrieval_summary = _build_retrieval_summary(
        rankings_json=args.retrieval_rankings_json,
        gold_jsonl=args.gold_jsonl,
        ks_csv=args.retrieval_ks,
        k_gen=args.retrieval_k_gen,
        run_id=run_id,
    )

    leaderboard_summary = {
        "All": {
            "QA-FL": scores["All"]["qa_fl"],
            "QA-Acc": scores["All"]["qa_acc"],
            "QA": scores["All"]["qa"],
        },
        "Unimodal": {
            "QA-FL": scores["Unimodal"]["qa_fl"],
            "QA-Acc": scores["Unimodal"]["qa_acc"],
            "QA": scores["Unimodal"]["qa"],
        },
        "Multimodal": {
            "QA-FL": scores["Multimodal"]["qa_fl"],
            "QA-Acc": scores["Multimodal"]["qa_acc"],
            "QA": scores["Multimodal"]["qa"],
        },
        "source": (
            "ACL-25 Query-Driven-Multimodal-GraphRAG Table 2 + WebQA EvalAI "
            "wire names (Fluency/Accuracy/mul -> QA-FL/QA-Acc/QA)."
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "1.6",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "predictions_path": str(args.predictions.resolve()),
            "gold_jsonl_path": str(args.gold_jsonl.resolve()),
            "split_label": args.split_label,
            "mmgraphrag_run_id": run_id or None,
            "git_commit": _git_commit(),
        },
        "counts": counts,
        "leaderboard_summary": leaderboard_summary,
        "retrieval_summary": retrieval_summary,
        "scores": scores,
        "diagnostics": diagnostics,
        "definitions": {
            "Unimodal": "metadata.webqa.multimodal == false (val: no img_posFacts; test: no img_Facts)",
            "Multimodal": "metadata.webqa.multimodal == true",
            "list_em_list_f1": (
                "list EM / list F1 from webqa_qa_scoring.evaluate_predictions; "
                "reference Query-Driven-Multimodal-GraphRAG eval/evaluate.py."
            ),
            "webqa_acc_approx": (
                "Category-aware approximation of WebQA QA-Acc (Qibin Chen / WebQA-main/eval_webqa.md); "
                "color/shape/YesNo -> keyword-set F1, number -> numeric-token F1, "
                "text/Others/choose -> raw bag-of-words recall. Matches the metric WebQA's leaderboard "
                "uses on val when keyword_answer is not available."
            ),
            "list_em_keyword_list_f1_keyword": (
                "Same MMQA list EM / list F1 evaluator as above, but both gold and prediction are "
                "first reduced with qcate_keyword_tokens(). For WebQA the raw gold is always a full "
                "natural-language sentence, so the unfiltered list_em collapses to 0; the keyword "
                "variant projects both sides onto the same Qcate-filtered bag the WebQA leaderboard "
                "uses against its hidden Keywords_answer, so EM/F1 can actually fire."
            ),
            "webqa_qa_fl": (
                "WebQA QA-FL (Fluency): BARTScore (default facebook/bart-large-cnn) "
                "normalized to [0,1] via exp(avg_log_p / WEBQA_FLUENCY_SCALE); see "
                "util/webqa_fluency.py. Falls back to ROUGE-L F1 when BART weights "
                "are unavailable (diagnostics.webqa_fluency_effective_backend=='rouge'). "
                "Aggregated to the leaderboard [0,100] scale."
            ),
            "webqa_qa": (
                "WebQA QA leaderboard score: per-sample FL_i * Acc_i averaged over the "
                "split. Matches the 'mul' field returned by the WebQA EvalAI server "
                "(reference/code/WebQA-main/WebQA-main/demo/Call_WebQA_eval_server_locally.py). "
                "Per-sample product then macro mean -- not product of macro means."
            ),
            "qa_fl_qa_acc_qa": (
                "Paper-consistent aliases introduced in schema 1.5. ``qa_fl`` / "
                "``qa_acc`` / ``qa`` are identical values to ``webqa_qa_fl`` / "
                "``webqa_acc_approx`` / ``webqa_qa`` above, but named after the "
                "ACL-25 Query-Driven-Multimodal-GraphRAG paper (Table 2) and the "
                "WebQA EvalAI server wire format "
                "(Fluency/Accuracy/mul -> QA-FL/QA-Acc/QA). Use these for any "
                "leaderboard-facing report / README table."
            ),
            "retrieval_summary": (
                "IR metrics for the ranked-source list fed to the generator (schema 1.6+). "
                "Filled from --retrieval_rankings_json (or result/<RUN_ID>/retrieval_rankings.json) "
                "using eval/evaluate_retrieval.py: overall / multimodal / unimodal / by_Qcate / "
                "by_modality buckets, each with hit@k / recall@k / mrr@k / ndcg@k / map@k / "
                "retrieval_f1 across the --retrieval_ks sweep. ``k_gen`` is the operating point "
                "the answer generator actually sees (default 10)."
            ),
        },
    }

    out = args.report_json
    if out is None:
        rid = run_id or default_stamp()
        out = _REPO_ROOT / "result" / "json" / f"{rid}_webqa_qa_eval_report.json"

    _atomic_write_json(out, report)

    qa_lb_path = out.parent / "qa_leaderboard.json"
    _atomic_write_json(
        qa_lb_path,
        {
            "schema_version": "1.0",
            "generated_at": report["generated_at"],
            "leaderboard_summary": leaderboard_summary,
            "inputs": report["inputs"],
        },
    )

    print(
        f"Unimodal F1/EM: {scores['Unimodal']['f1']:.4f} / {scores['Unimodal']['em']:.4f} | "
        f"Multimodal F1/EM: {scores['Multimodal']['f1']:.4f} / {scores['Multimodal']['em']:.4f} | "
        f"All F1/EM: {scores['All']['f1']:.4f} / {scores['All']['em']:.4f}"
    )
    print(
        "Keyword list F1/EM  "
        f"All: {scores['All']['list_f1_keyword']:.4f} / {scores['All']['list_em_keyword']:.4f} | "
        f"Unimodal: {scores['Unimodal']['list_f1_keyword']:.4f} / "
        f"{scores['Unimodal']['list_em_keyword']:.4f} | "
        f"Multimodal: {scores['Multimodal']['list_f1_keyword']:.4f} / "
        f"{scores['Multimodal']['list_em_keyword']:.4f}"
    )
    print(
        f"{log_pfx} acc_approx  "
        f"All: {scores['All']['webqa_acc_approx']:.4f} | "
        f"Unimodal: {scores['Unimodal']['webqa_acc_approx']:.4f} | "
        f"Multimodal: {scores['Multimodal']['webqa_acc_approx']:.4f}"
    )
    print(
        f"{log_pfx} leaderboard  QA-FL / QA-Acc / QA  (ACL-25 paper Table 2)  "
        f"All: {scores['All']['qa_fl']:.4f} / {scores['All']['qa_acc']:.4f} / {scores['All']['qa']:.4f} | "
        f"Unimodal: {scores['Unimodal']['qa_fl']:.4f} / {scores['Unimodal']['qa_acc']:.4f} / {scores['Unimodal']['qa']:.4f} | "
        f"Multimodal: {scores['Multimodal']['qa_fl']:.4f} / {scores['Multimodal']['qa_acc']:.4f} / {scores['Multimodal']['qa']:.4f}"
    )
    print(
        f"  fluency backend={diagnostics.get('webqa_fluency_effective_backend', 'bart')} "
        f"model={diagnostics.get('webqa_fluency_model', '')}"
    )
    _print_retrieval_banner(retrieval_summary)
    if scores["by_Qcate_webqa_acc_approx"]:
        per_cat = " | ".join(
            f"{c}={v:.2f}" for c, v in scores["by_Qcate_webqa_acc_approx"].items()
        )
        print(f"{log_pfx} acc_approx by Qcate: {per_cat}")
    if scores.get("by_Qcate_webqa_qa"):
        per_cat_qa = " | ".join(
            f"{c}={v:.2f}" for c, v in scores["by_Qcate_webqa_qa"].items()
        )
        print(f"{log_pfx} QA by Qcate: {per_cat_qa}")
    if scores.get("by_Qcate_list_em_keyword"):
        per_cat_em = " | ".join(
            f"{c}={v:.2f}" for c, v in scores["by_Qcate_list_em_keyword"].items()
        )
        print(f"Keyword list EM by Qcate: {per_cat_em}")
    print(f"Wrote {out.resolve()}")
    print(f"Wrote {qa_lb_path.resolve()}")


if __name__ == "__main__":
    main()
