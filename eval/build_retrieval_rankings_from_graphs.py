import argparse
import json
import os
from pathlib import Path

import networkx as nx


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_ranked_source_ids_from_graph(graph: nx.Graph, top_k: int = 10):
    ranked = []
    seen = set()
    for node_name, node_data in graph.nodes(data=True):
        source_id = node_data.get("source_id")
        if not source_id:
            continue
        sid = str(source_id)
        if sid in seen:
            continue
        seen.add(sid)
        ranked.append({"id": sid, "score": float(graph.degree[node_name])})
    ranked.sort(key=lambda x: (-x["score"], x["id"]))
    return ranked[:top_k]


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
    raise SystemExit("Set --gold_jsonl or MMGRAPHRAG_RUN_ID with an exported webqa_slice.")


def main():
    parser = argparse.ArgumentParser(description="Build retrieval ranking JSON from graphml files.")
    parser.add_argument("--graph_dir", type=Path, required=True)
    parser.add_argument("--gold_jsonl", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()
    if args.gold_jsonl is None:
        args.gold_jsonl = _default_gold_jsonl()

    if not args.graph_dir.is_dir():
        raise SystemExit(f"Missing graph dir: {args.graph_dir}")
    if not args.gold_jsonl.is_file():
        raise SystemExit(f"Missing gold jsonl: {args.gold_jsonl}")

    examples = read_jsonl(args.gold_jsonl)
    out = {}
    for ex in examples:
        qid = str(ex["qid"])
        graph_path = args.graph_dir / f"{qid}_graph.graphml"
        if not graph_path.exists():
            out[qid] = []
            continue
        g = nx.read_graphml(graph_path)
        out[qid] = extract_ranked_source_ids_from_graph(g, top_k=args.top_k)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
