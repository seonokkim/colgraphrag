#!/usr/bin/env python3
"""
Validate a MultiModalQA toy bundle produced by build_multimodalqa_toy.py.

Checks question count, duplicate qids, referenced doc IDs vs subset jsonl rows,
optional image file presence under final_dataset_images/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


def _load_collect_doc_ids(repo_root: Path):
    path = repo_root / "scripts" / "build_multimodalqa_toy.py"
    spec = importlib.util.spec_from_file_location("build_multimodalqa_toy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.collect_doc_ids


def _read_jsonl_ids(path: Path, key: str = "id") -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            oid = str(row.get(key, ""))
            out[oid] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify MMQA toy bundle under data/multimodalqa.")
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--bundle",
        type=Path,
        default=repo / "data" / "multimodalqa",
        help="Root directory containing dataset/ and manifest.json",
    )
    ap.add_argument(
        "--require-images",
        action="store_true",
        help="Fail if referenced image pixels are missing under final_dataset_images/.",
    )
    args = ap.parse_args()

    root = args.bundle.resolve()
    ds = root / "dataset"
    manifest_path = root / "manifest.json"
    errs: list[str] = []

    if not manifest_path.is_file():
        print(f"ERROR: missing {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_manifest = int(manifest.get("n_questions_written", 0))
    split = str(manifest.get("split", "dev"))

    q_path = ds / f"MMQA_{split}_n{n_manifest}.jsonl"
    if not q_path.is_file():
        # Fall back to any MMQA_*_n*.jsonl
        candidates = sorted(ds.glob("MMQA_*_n*.jsonl"))
        candidates = [p for p in candidates if "tables" not in p.name and "texts" not in p.name and "images" not in p.name]
        if candidates:
            q_path = candidates[0]
            print(f"Note: using {q_path.name} (manifest split/count mismatch)", file=sys.stderr)
        else:
            print(f"ERROR: no questions jsonl under {ds}", file=sys.stderr)
            return 2

    collect_doc_ids = _load_collect_doc_ids(repo)

    questions: list[dict] = []
    qids: list[str] = []
    with q_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append(obj)
            qids.append(str(obj.get("qid", "")))

    n_q = len(questions)
    if n_manifest and n_q != n_manifest:
        errs.append(f"manifest n_questions_written={n_manifest} but {q_path.name} has {n_q} lines")

    dup_qids = [k for k, v in Counter(qids).items() if v > 1]
    if dup_qids:
        errs.append(f"duplicate qids (sample): {dup_qids[:5]}")

    for i, q in enumerate(questions):
        if not str(q.get("qid", "")).strip():
            errs.append(f"question line {i + 1}: empty qid")
        if not str(q.get("question", "")).strip():
            errs.append(f"question line {i + 1}: empty question text")

    all_t: set[str] = set()
    all_x: set[str] = set()
    all_i: set[str] = set()
    for q in questions:
        t, tx, im = collect_doc_ids(q)
        all_t |= t
        all_x |= tx
        all_i |= im

    tab_path = ds / f"MMQA_tables_n{n_q}.jsonl"
    txt_path = ds / f"MMQA_texts_n{n_q}.jsonl"
    img_path = ds / f"MMQA_images_n{n_q}.jsonl"
    for p in (tab_path, txt_path, img_path):
        if not p.is_file():
            errs.append(f"missing subset file: {p}")
    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    tabs = _read_jsonl_ids(tab_path)
    texts = _read_jsonl_ids(txt_path)
    images = _read_jsonl_ids(img_path)

    def check_subset(name: str, needed: set[str], have: dict[str, dict]) -> None:
        have_ids = set(have.keys())
        if "" in have_ids:
            errs.append(f"{name}: row with empty id")
        miss = needed - have_ids
        if miss:
            errs.append(f"{name}: {len(miss)} referenced doc_ids missing from subset (e.g. {sorted(miss)[:3]})")
        extra = have_ids - needed
        if extra:
            errs.append(f"{name}: {len(extra)} unexpected rows not referenced by any question (e.g. {sorted(extra)[:3]})")

    check_subset("tables", all_t, tabs)
    check_subset("texts", all_x, texts)
    check_subset("images", all_i, images)

    img_root = root / "final_dataset_images"
    img_missing_paths: list[str] = []
    for oid, row in images.items():
        rel = str(row.get("path", "")).strip()
        if not rel:
            img_missing_paths.append(f"(id={oid}) empty path")
            continue
        cand = img_root / rel
        if not cand.is_file():
            cand = img_root / Path(rel).name
        if not cand.is_file():
            img_missing_paths.append(rel)

    if img_missing_paths and args.require_images:
        errs.append(f"images: {len(img_missing_paths)} files missing under {img_root}")
    elif img_missing_paths:
        print(f"WARN: {len(img_missing_paths)} image files not on disk (expected without --images-root build). OK for text/table-only checks.")

    if errs:
        for e in errs:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("OK: multimodalqa toy bundle")
    print(f"  bundle: {root}")
    print(f"  questions: {n_q}  refs — tables:{len(all_t)} texts:{len(all_x)} images:{len(all_i)}")
    print(f"  subset rows: tables:{len(tabs)} texts:{len(texts)} images:{len(images)}")
    if not img_missing_paths:
        print(f"  image pixels: all {len(images)} paths resolve under final_dataset_images/")

    meta_counts_table = sum(
        1
        for q in questions
        if str((q.get("metadata") or {}).get("table_id") or "").strip()
    )
    meta_counts_text = sum(
        1 for q in questions if (q.get("metadata") or {}).get("text_doc_ids")
    )
    meta_counts_image = sum(
        1 for q in questions if (q.get("metadata") or {}).get("image_doc_ids")
    )
    mod_hist = Counter()
    for q in questions:
        mds = (q.get("metadata") or {}).get("modalities")
        if isinstance(mds, list) and mds:
            key = tuple(sorted(str(x).lower() for x in mds))
            mod_hist[key] += 1
        else:
            mod_hist[()] += 1
    print(
        "  metadata coverage: "
        f"questions with table_id={meta_counts_table}, "
        f"with text_doc_ids={meta_counts_text}, "
        f"with image_doc_ids={meta_counts_image}"
    )
    top_mod = ", ".join(f"{k or 'unset'}:{v}" for k, v in mod_hist.most_common(5))
    print(f"  modalities (top 5 tuple:histogram): {top_mod}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
