"""Retrieval IR metrics for the WebQA pipeline (final phase).

Metrics per query at top-k (macro-averaged across the slice):

* ``hit@k``          : at least one gold source inside the top-k.
* ``recall@k``       : ``#hits@k / #relevant``.
* ``mrr@k``          : ``1 / rank`` of the first hit inside top-k, else ``0``.
* ``ndcg@k``         : DCG of binary hits / ideal DCG over ``min(|R|, k)``.
* ``map@k``          : average precision at relevant ranks within top-k
  (``sum(P@r for r in hit_ranks) / min(#relevant, k)``). See Manning et al.
  IR textbook; standard "TREC-style" AP truncated at ``k``.
* ``retrieval_f1``   : per-query F1 over the **set overlap** of the top-k
  predicted IDs vs gold relevant IDs (precision=TP/|topk|, recall=TP/|R|).
  This matches the WebQA-main baseline README ``Retrieval F1`` column
  (``reference/code/WebQA-main/.../Baseline_prediction_files_on_Val/README.md``).

Stratification (WebQA-specific, enabled by ``--stratify_webqa`` and always on
in the stratified JSON block):

* ``overall``        : every gold row.
* ``multimodal`` /
  ``unimodal``       : from ``metadata.webqa.multimodal`` / presence of
  ``image_doc_ids``.
* ``by_Qcate``       : one sub-block per WebQA ``Qcate`` (``color`` /
  ``shape`` / ``YesNo`` / ``number`` / ``text`` / ``Others`` / ``choose``).
* ``by_modality``    : restrict the gold set to a single modality ("text" =
  ``text_doc_ids``, "image" = ``image_doc_ids``, "table" = ``table_id``),
  keeping the predicted ranking intact. This isolates whether the retriever
  fails on image nodes vs text/table nodes.

The consumer-facing ``k_gen`` (the k the answer generator actually sees) is
emitted separately as a top-level integer so reviewers can look up the exact
operating point without parsing the ``ks`` sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


REPO_MM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_OUT_DIR = REPO_MM_ROOT / "result" / "json"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _parse_ranked_ids(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        ranked: List[str] = []
        for item in value:
            if isinstance(item, str):
                ranked.append(item)
            elif isinstance(item, dict):
                if "id" in item:
                    ranked.append(str(item["id"]))
                elif "doc_id" in item:
                    ranked.append(str(item["doc_id"]))
                elif "corpus_id" in item:
                    ranked.append(str(item["corpus_id"]))
        return ranked
    return []


def _extract_mmq_relevant_ids(example: dict) -> Set[str]:
    md = example.get("metadata", {})
    relevant: Set[str] = set()
    for x in (md.get("text_doc_ids", []) or []):
        relevant.add(str(x))
    for x in (md.get("image_doc_ids", []) or []):
        relevant.add(str(x))
    table_id = md.get("table_id", None)
    if table_id is not None and str(table_id).strip():
        relevant.add(str(table_id))
    return relevant


def _extract_relevant_by_modality(example: dict) -> Dict[str, Set[str]]:
    """Return gold IDs partitioned by modality: text / image / table."""
    md = example.get("metadata", {})
    out: Dict[str, Set[str]] = {"text": set(), "image": set(), "table": set()}
    for x in (md.get("text_doc_ids", []) or []):
        out["text"].add(str(x))
    for x in (md.get("image_doc_ids", []) or []):
        out["image"].add(str(x))
    table_id = md.get("table_id", None)
    if table_id is not None and str(table_id).strip():
        out["table"].add(str(table_id))
    return out


def _stratum_for_example(ex: dict) -> str:
    md = ex.get("metadata") or {}
    w = md.get("webqa") or {}
    if isinstance(w.get("multimodal"), bool):
        return "multimodal" if w["multimodal"] else "unimodal"
    if md.get("image_doc_ids"):
        return "multimodal"
    return "unimodal"


def _qcate_for_example(ex: dict) -> str:
    md = ex.get("metadata") or {}
    w = md.get("webqa") or {}
    q = w.get("Qcate")
    return str(q) if isinstance(q, str) and q else "text"


def _load_gold_relevance(gold_jsonl_path: Path, qrels_json_path: Path | None) -> Dict[str, Set[str]]:
    if qrels_json_path is not None:
        raw = json.loads(qrels_json_path.read_text(encoding="utf-8"))
        return {str(qid): set(_parse_ranked_ids(v)) for qid, v in raw.items()}
    examples = read_jsonl(gold_jsonl_path)
    return {str(ex["qid"]): _extract_mmq_relevant_ids(ex) for ex in examples}


def _dcg_at_k(binary_rels: List[int], k: int) -> float:
    return sum((1.0 / math.log2(i + 1.0)) for i, rel in enumerate(binary_rels[:k], start=1) if rel)


def _ap_at_k(hits: List[int], num_relevant: int, k: int) -> float:
    """Truncated average precision: mean of precision@rank at each hit in top-k.

    Denominator is ``min(num_relevant, k)`` — the standard AP@k convention that
    TREC and most IR toolkits use. Returns 0.0 when there is no gold relevance.
    """
    if num_relevant <= 0 or k <= 0:
        return 0.0
    num_hits = 0
    ap_sum = 0.0
    for rank, h in enumerate(hits[:k], start=1):
        if h:
            num_hits += 1
            ap_sum += num_hits / rank
    denom = min(num_relevant, k)
    return ap_sum / denom if denom > 0 else 0.0


def _retrieval_f1_at_k(topk: List[str], relevant: Set[str]) -> float:
    """Per-query F1 over the *set overlap* of top-k predictions vs gold.

    Mirrors the WebQA-main baseline README "Retrieval F1" column (per-query
    precision & recall over predicted source IDs, F1 then macro-averaged
    outside this function). Order is ignored by construction.
    """
    if not topk or not relevant:
        return 0.0
    pred_set = set(topk)
    tp = len(pred_set & relevant)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(relevant)
    if precision + recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _metrics_for_one_query(
    ranked_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> Dict[str, float]:
    if k <= 0:
        return {"hit@k": 0.0, "recall@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0,
                "map@k": 0.0, "retrieval_f1": 0.0}
    topk = ranked_ids[:k]
    hits = [1 if rid in relevant_ids else 0 for rid in topk]
    num_hits = sum(hits)
    hit_k = 1.0 if num_hits > 0 else 0.0
    recall_k = (num_hits / len(relevant_ids)) if relevant_ids else 0.0
    mrr_k = 0.0
    for rank, h in enumerate(hits, start=1):
        if h:
            mrr_k = 1.0 / rank
            break
    dcg = _dcg_at_k(hits, k)
    idcg = _dcg_at_k([1] * min(len(relevant_ids), k), k)
    ndcg_k = (dcg / idcg) if idcg > 0 else 0.0
    map_k = _ap_at_k(hits, len(relevant_ids), k)
    f1 = _retrieval_f1_at_k(topk, relevant_ids)
    return {
        "hit@k": hit_k,
        "recall@k": recall_k,
        "mrr@k": mrr_k,
        "ndcg@k": ndcg_k,
        "map@k": map_k,
        "retrieval_f1": f1,
    }


def evaluate_retrieval(
    predictions: Dict[str, List[str]],
    gold_relevance: Dict[str, Set[str]],
    ks: Iterable[int],
) -> dict:
    qids = sorted(gold_relevance.keys())
    total = len(qids)
    missing_preds = 0
    no_gold_relevance = 0
    metrics: Dict[str, Dict[str, float]] = {}
    for k in ks:
        accum = {"hit@k": 0.0, "recall@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0,
                 "map@k": 0.0, "retrieval_f1": 0.0}
        for qid in qids:
            rel = gold_relevance.get(qid, set())
            ranked = predictions.get(qid, [])
            if qid not in predictions:
                missing_preds += 1
            if not rel:
                no_gold_relevance += 1
            per_q = _metrics_for_one_query(ranked, rel, k)
            for name, val in per_q.items():
                accum[name] += val
        denom = max(total, 1)
        metrics[f"k={k}"] = {name: round(val / denom, 6) for name, val in accum.items()}
    return {
        "num_queries": total,
        "num_queries_missing_predictions": missing_preds,
        "num_queries_with_empty_gold_relevance": no_gold_relevance,
        "metrics": metrics,
    }


def _evaluate_retrieval_by_modality(
    predictions: Dict[str, List[str]],
    modality_gold: Dict[str, Dict[str, Set[str]]],
    ks: Iterable[int],
) -> Dict[str, dict]:
    """Restrict gold to a single modality and evaluate; skip empty slices."""
    out: Dict[str, dict] = {}
    for modality in ("text", "image", "table"):
        slice_gold = {
            qid: modality_gold[qid][modality]
            for qid in modality_gold
            if modality_gold[qid][modality]
        }
        if not slice_gold:
            out[modality] = {"num_queries": 0, "metrics": {}}
            continue
        out[modality] = evaluate_retrieval(predictions, slice_gold, ks)
        out[modality]["num_queries_in_modality"] = len(slice_gold)
    return out


def evaluate_retrieval_stratified(
    predictions: Dict[str, List[str]],
    gold_jsonl_path: Path,
    qrels_json_path: Path | None,
    ks: Iterable[int],
) -> dict:
    examples = read_jsonl(gold_jsonl_path)
    by_stratum: Dict[str, List[dict]] = {"multimodal": [], "unimodal": []}
    by_qcate: Dict[str, List[dict]] = {}
    for ex in examples:
        by_stratum[_stratum_for_example(ex)].append(ex)
        by_qcate.setdefault(_qcate_for_example(ex), []).append(ex)

    def gold_subset(rows: List[dict]) -> Dict[str, Set[str]]:
        if qrels_json_path is not None:
            all_q = _load_gold_relevance(gold_jsonl_path, qrels_json_path)
            qids = {str(ex["qid"]) for ex in rows}
            return {q: all_q[q] for q in qids if q in all_q}
        return {str(ex["qid"]): _extract_mmq_relevant_ids(ex) for ex in rows}

    out: dict = {}
    for name, rows in by_stratum.items():
        rel = gold_subset(rows)
        if not rel:
            out[name] = {"num_queries": 0, "metrics": {}}
            continue
        out[name] = evaluate_retrieval(predictions, rel, ks)
        out[name]["num_queries_in_stratum"] = len(rel)

    by_qcate_out: Dict[str, dict] = {}
    for cat, rows in sorted(by_qcate.items()):
        rel = gold_subset(rows)
        if not rel:
            by_qcate_out[cat] = {"num_queries": 0, "metrics": {}}
            continue
        block = evaluate_retrieval(predictions, rel, ks)
        block["num_queries_in_qcate"] = len(rel)
        by_qcate_out[cat] = block
    out["by_Qcate"] = by_qcate_out

    if qrels_json_path is None:
        modality_gold = {str(ex["qid"]): _extract_relevant_by_modality(ex) for ex in examples}
        out["by_modality"] = _evaluate_retrieval_by_modality(predictions, modality_gold, ks)

    full = _load_gold_relevance(gold_jsonl_path, qrels_json_path)
    out["overall"] = evaluate_retrieval(predictions, full, ks)
    out["overall"]["num_queries_in_stratum"] = len(full)
    return out


def _default_gold_jsonl() -> Path:
    rid = os.getenv("MMGRAPHRAG_RUN_ID", "").strip()
    base = Path(__file__).resolve().parents[1]
    if rid:
        p = base / "result" / rid / "webqa_slice" / "webqa_questions.jsonl"
        if p.is_file():
            return p
    env = os.getenv("EVAL_GOLD_JSONL", "").strip()
    if env:
        return Path(env)
    return base / "result" / (rid or "webqa_slice") / "webqa_questions.jsonl"


def _load_predictions(prediction_json_path: Path) -> Dict[str, List[str]]:
    raw = json.loads(prediction_json_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "predictions" in raw and isinstance(raw["predictions"], dict):
        raw = raw["predictions"]
    return {str(qid): _parse_ranked_ids(v) for qid, v in raw.items()}


def _print_banner(result: dict, k_gen: int) -> None:
    """Human-friendly stdout banner emphasising the generator-visible k."""
    def _metric_block(block: dict, label: str) -> str:
        m = (block or {}).get("metrics", {}).get(f"k={k_gen}")
        if not m:
            return f"{label}@{k_gen}: (empty)"
        return (
            f"{label}@{k_gen}: "
            f"hit={m['hit@k']:.4f} recall={m['recall@k']:.4f} "
            f"mrr={m['mrr@k']:.4f} ndcg={m['ndcg@k']:.4f} "
            f"map={m['map@k']:.4f} retF1={m['retrieval_f1']:.4f}"
        )

    if "overall" in result:
        print(_metric_block(result.get("overall", {}), "Retriever overall"))
        if result.get("multimodal", {}).get("num_queries"):
            print(_metric_block(result.get("multimodal", {}), "Retriever multimodal"))
        if result.get("unimodal", {}).get("num_queries"):
            print(_metric_block(result.get("unimodal", {}), "Retriever unimodal"))
        by_qcate = result.get("by_Qcate") or {}
        if by_qcate:
            per_cat_parts = []
            for cat, block in sorted(by_qcate.items()):
                m = (block or {}).get("metrics", {}).get(f"k={k_gen}")
                if m:
                    per_cat_parts.append(f"{cat}:rec={m['recall@k']:.3f}/ndcg={m['ndcg@k']:.3f}")
            if per_cat_parts:
                print(f"Retriever by_Qcate@{k_gen}: " + " | ".join(per_cat_parts))
        by_mod = result.get("by_modality") or {}
        if by_mod:
            per_mod_parts = []
            for mod, block in (("text", by_mod.get("text")), ("image", by_mod.get("image")), ("table", by_mod.get("table"))):
                if not block or not block.get("num_queries"):
                    continue
                m = block.get("metrics", {}).get(f"k={k_gen}")
                if m:
                    per_mod_parts.append(f"{mod}:rec={m['recall@k']:.3f}/retF1={m['retrieval_f1']:.3f}")
            if per_mod_parts:
                print(f"Retriever by_modality@{k_gen}: " + " | ".join(per_mod_parts))
    else:
        print(_metric_block(result, "Retriever"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval-only metrics from ranked IDs.")
    parser.add_argument("--prediction_json", type=Path, required=True)
    parser.add_argument(
        "--gold_jsonl",
        type=Path,
        default=None,
        help="Defaults to result/<MMGRAPHRAG_RUN_ID>/webqa_slice/webqa_questions.jsonl when present.",
    )
    parser.add_argument("--qrels_json", type=Path, default=None)
    parser.add_argument("--ks", type=str, default="1,3,5,10")
    parser.add_argument(
        "--k_gen",
        type=int,
        default=10,
        help="k the answer generator actually consumes; echoed in stdout and JSON.",
    )
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument(
        "--stratify_webqa",
        action="store_true",
        help="If set, also report multimodal / unimodal / overall / by_Qcate / by_modality.",
    )
    args = parser.parse_args()
    if args.gold_jsonl is None:
        args.gold_jsonl = _default_gold_jsonl()

    if not args.prediction_json.is_file():
        raise SystemExit(f"Missing predictions: {args.prediction_json}")
    if args.qrels_json is None and not args.gold_jsonl.is_file():
        raise SystemExit(f"Missing gold jsonl: {args.gold_jsonl}")
    if args.qrels_json is not None and not args.qrels_json.is_file():
        raise SystemExit(f"Missing qrels json: {args.qrels_json}")

    ks = sorted(set(int(x.strip()) for x in args.ks.split(",") if x.strip() and int(x.strip()) > 0))
    if not ks:
        raise SystemExit("No valid k values in --ks")
    if args.k_gen not in ks:
        ks = sorted(set(ks + [args.k_gen]))

    predictions = _load_predictions(args.prediction_json)
    if args.stratify_webqa and args.qrels_json is None:
        result = evaluate_retrieval_stratified(predictions, args.gold_jsonl, args.qrels_json, ks)
    else:
        gold_relevance = _load_gold_relevance(args.gold_jsonl, args.qrels_json)
        result = evaluate_retrieval(predictions, gold_relevance, ks)
    result["k_gen"] = args.k_gen
    result["inputs"] = {
        "prediction_json": str(args.prediction_json.resolve()),
        "gold_jsonl": str(args.gold_jsonl.resolve()) if args.gold_jsonl else None,
        "qrels_json": str(args.qrels_json.resolve()) if args.qrels_json else None,
        "ks": ks,
        "k_gen": args.k_gen,
    }
    result["metric_definitions"] = {
        "hit@k": "At least one gold source inside top-k (0 or 1), macro-averaged.",
        "recall@k": "#hits_in_topk / #relevant, macro-averaged.",
        "mrr@k": "1 / rank of first hit in top-k, else 0; macro-averaged.",
        "ndcg@k": "DCG(binary hits) / IDCG over min(|relevant|, k); macro-averaged.",
        "map@k": "Truncated average precision at k (sum P@r over hits / min(|R|, k)); macro-averaged.",
        "retrieval_f1": (
            "Per-query set F1 over top-k vs gold, macro-averaged. Matches the WebQA-main "
            "baseline README 'Retrieval F1' column "
            "(reference/code/WebQA-main/.../Baseline_prediction_files_on_Val/README.md)."
        ),
    }

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    _print_banner(result, args.k_gen)
    if args.out_json is not None:
        out_path = args.out_json
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = DEFAULT_JSON_OUT_DIR / f"{stamp}_retrieval_eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
