#!/usr/bin/env python3
"""
Build a small MultiModalQA (Allen AI) toy bundle: N questions plus only the
referenced tables / texts / images rows and optional image file copies.

Official layout reference: https://github.com/allenai/multimodalqa

Typical full-data layout (mirrors ``C:\\workspace\\data\\multimodalqa``):

  MMQA_dev.jsonl.gz | MMQA_train.jsonl.gz | MMQA_test.jsonl.gz
  MMQA_tables.jsonl.gz
  MMQA_texts.jsonl.gz
  MMQA_images.jsonl.gz
  final_dataset_images/   # extracted image bundle (paths match ``path`` in images jsonl)

This script can stream context files from GitHub when local paths are omitted.

Env:
  MMQA_TOY_SOURCE_ROOT   Optional directory containing the gz files above.
  MMQA_TOY_IMAGES_ROOT     Directory containing image files (often ``final_dataset_images``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

GITHUB_DATASET = (
    "https://raw.githubusercontent.com/allenai/multimodalqa/master/dataset"
)


def _open_gz_path(path: Path):
    return gzip.open(path, "rb")


def _open_gz_url(url: str):
    req = Request(url, headers={"User-Agent": "build_multimodalqa_toy/1.0"})
    resp = urlopen(req, timeout=120)
    return gzip.GzipFile(fileobj=resp)


def _iter_jsonl_gz(open_fn) -> Any:
    with open_fn as gz:
        for raw in gz:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            yield json.loads(line)


def _walk_answer_obj(obj: Any, tables: set[str], texts: set[str], images: set[str]) -> None:
    if isinstance(obj, dict):
        modality = str(obj.get("modality") or "").lower()
        for ti in obj.get("text_instances") or []:
            if isinstance(ti, dict) and ti.get("doc_id"):
                texts.add(str(ti["doc_id"]))
        for ii in obj.get("image_instances") or []:
            if isinstance(ii, dict) and ii.get("doc_id"):
                images.add(str(ii["doc_id"]))
        if modality == "table" and obj.get("table_indices"):
            pass  # table doc id comes from question metadata
        for v in obj.values():
            _walk_answer_obj(v, tables, texts, images)
    elif isinstance(obj, list):
        for x in obj:
            _walk_answer_obj(x, tables, texts, images)


def collect_doc_ids(question: dict) -> tuple[set[str], set[str], set[str]]:
    tables: set[str] = set()
    texts: set[str] = set()
    images: set[str] = set()

    meta = question.get("metadata") or {}
    if isinstance(meta, dict):
        for x in meta.get("image_doc_ids") or []:
            images.add(str(x))
        for x in meta.get("text_doc_ids") or []:
            texts.add(str(x))
        tid = meta.get("table_id")
        if tid:
            tables.add(str(tid))
        for ia in meta.get("intermediate_answers") or []:
            _walk_answer_obj(ia, tables, texts, images)

    for sc in question.get("supporting_context") or []:
        if not isinstance(sc, dict):
            continue
        did = str(sc.get("doc_id") or "")
        part = str(sc.get("doc_part") or "").lower()
        if part == "table":
            tables.add(did)
        elif part == "text":
            texts.add(did)
        elif part == "image":
            images.add(did)

    for ans in question.get("answers") or []:
        _walk_answer_obj(ans, tables, texts, images)

    return tables, texts, images


def load_first_n_questions(open_fn, n: int) -> list[dict]:
    rows: list[dict] = []
    for obj in _iter_jsonl_gz(open_fn):
        rows.append(obj)
        if len(rows) >= n:
            break
    return rows


def filter_context_stream(open_fn, keep_ids: set[str], out_path: Path, label: str) -> int:
    found: set[str] = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for obj in _iter_jsonl_gz(open_fn):
            oid = str(obj.get("id", ""))
            if oid in keep_ids:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                found.add(oid)
    missing = keep_ids - found
    if missing:
        sample = list(sorted(missing))[:5]
        print(
            f"  [{label}] wrote {len(found)} / {len(keep_ids)} ids; "
            f"missing {len(missing)} (e.g. {sample})",
            file=sys.stderr,
        )
    return len(found)


def gzip_file(src: Path, dst: Path) -> None:
    with src.open("rb") as f_in:
        data = f_in.read()
    with gzip.open(dst, "wb") as f_out:
        f_out.write(data)


def copy_image_files(
    images_subset_path: Path,
    images_root: Path | None,
    dest_images: Path,
) -> tuple[int, int]:
    """Return (copied, missing)."""
    copied = 0
    missing = 0
    dest_images.mkdir(parents=True, exist_ok=True)
    if images_root is None or not images_root.is_dir():
        return copied, missing

    with images_subset_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rel = str(row.get("path") or "").strip()
            if not rel:
                missing += 1
                continue
            # Paths in MMQA are typically flat filenames (e.g. Taipei.jpg)
            src = images_root / rel
            if not src.is_file():
                # try basename only
                src = images_root / Path(rel).name
            dst = dest_images / Path(rel).name
            if src.is_file():
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1
    return copied, missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Build MultiModalQA toy subset (N questions).")
    ap.add_argument(
        "--split",
        choices=("dev", "train", "test"),
        default="dev",
        help="Which QA split to draw the first N lines from (default: dev).",
    )
    ap.add_argument("--n", type=int, default=100, help="Number of questions (default: 100).")
    repo = Path(__file__).resolve().parents[1]
    default_out = repo / "data" / "multimodalqa"
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Output directory (default: {default_out})",
    )
    ap.add_argument(
        "--mmqa-root",
        type=Path,
        default=None,
        help="Directory with MMQA_*.jsonl.gz. If omitted, use MMQA_TOY_SOURCE_ROOT or download QA gz from GitHub.",
    )
    ap.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Folder with image files (e.g. final_dataset_images). "
        "Also reads MMQA_TOY_IMAGES_ROOT when unset.",
    )
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="Do not fetch gz files from GitHub; require local --mmqa-root for context files.",
    )
    args = ap.parse_args()

    mmqa_root = args.mmqa_root or (
        Path(os.getenv("MMQA_TOY_SOURCE_ROOT", "").strip())
        if os.getenv("MMQA_TOY_SOURCE_ROOT", "").strip()
        else None
    )
    images_root = args.images_root or (
        Path(os.getenv("MMQA_TOY_IMAGES_ROOT", "").strip())
        if os.getenv("MMQA_TOY_IMAGES_ROOT", "").strip()
        else None
    )

    split_file = f"MMQA_{args.split}.jsonl.gz"
    qa_name = split_file

    def qa_local() -> Path | None:
        if mmqa_root and (mmqa_root / qa_name).is_file():
            return mmqa_root / qa_name
        return None

    def ctx_local(name: str) -> Path | None:
        if mmqa_root and (mmqa_root / name).is_file():
            return mmqa_root / name
        return None

    qa_path = qa_local()
    if qa_path is not None:
        qa_open = lambda: _open_gz_path(qa_path)
    else:
        if args.no_download:
            print("QA file not found and --no-download set.", file=sys.stderr)
            return 2
        qa_url = f"{GITHUB_DATASET}/{qa_name}"
        print(f"Streaming QA split from {qa_url}")
        qa_open = lambda: _open_gz_url(qa_url)

    questions = load_first_n_questions(qa_open(), args.n)
    if len(questions) < args.n:
        print(f"Only {len(questions)} questions available (requested {args.n}).", file=sys.stderr)

    all_tables: set[str] = set()
    all_texts: set[str] = set()
    all_images: set[str] = set()
    for q in questions:
        t, tx, im = collect_doc_ids(q)
        all_tables |= t
        all_texts |= tx
        all_images |= im

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ds_out = out / "dataset"
    ds_out.mkdir(parents=True, exist_ok=True)

    q_subset_path = ds_out / f"MMQA_{args.split}_n{len(questions)}.jsonl"
    with q_subset_path.open("w", encoding="utf-8") as fq:
        for q in questions:
            fq.write(json.dumps(q, ensure_ascii=False) + "\n")
    gzip_file(q_subset_path, ds_out / f"MMQA_{args.split}_n{len(questions)}.jsonl.gz")

    def context_opener(fname: str):
        p = ctx_local(fname)
        if p:
            return lambda: _open_gz_path(p)
        if args.no_download:
            raise FileNotFoundError(f"Missing {fname} under {mmqa_root} and --no-download set.")
        url = f"{GITHUB_DATASET}/{fname}"
        print(f"Streaming {fname} from {url}")
        return lambda u=url: _open_gz_url(u)

    for label, fname, keep in (
        ("tables", "MMQA_tables.jsonl.gz", all_tables),
        ("texts", "MMQA_texts.jsonl.gz", all_texts),
        ("images", "MMQA_images.jsonl.gz", all_images),
    ):
        sub_path = ds_out / f"MMQA_{label}_n{len(questions)}.jsonl"
        try:
            opener = context_opener(fname)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            return 2
        filter_context_stream(opener(), keep, sub_path, label)
        gzip_file(sub_path, ds_out / f"{sub_path.name}.gz")

    img_dir = out / "final_dataset_images"
    copied, miss = copy_image_files(ds_out / f"MMQA_images_n{len(questions)}.jsonl", images_root, img_dir)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "n_questions_requested": args.n,
        "n_questions_written": len(questions),
        "qids": [str(q.get("qid", "")) for q in questions],
        "unique_table_doc_ids": len(all_tables),
        "unique_text_doc_ids": len(all_texts),
        "unique_image_doc_ids": len(all_images),
        "mmqa_root_used": str(mmqa_root) if mmqa_root else None,
        "images_root_used": str(images_root) if images_root else None,
        "images_copied": copied,
        "images_missing_at_source": miss,
        "output_dir": str(out),
        "dataset_files": {
            "questions_jsonl": str(q_subset_path.relative_to(out)),
            "tables_jsonl": f"dataset/MMQA_tables_n{len(questions)}.jsonl",
            "texts_jsonl": f"dataset/MMQA_texts_n{len(questions)}.jsonl",
            "images_jsonl": f"dataset/MMQA_images_n{len(questions)}.jsonl",
        },
        "note": "Image binaries are only populated when --images-root points at extracted MMQA images.",
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote toy bundle under {out}")
    print(f"  Questions: {len(questions)}  Tables: {len(all_tables)}  Texts: {len(all_texts)}  Images: {len(all_images)}")
    print(f"  Image files copied: {copied}  missing from source: {miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
