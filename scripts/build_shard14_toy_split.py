#!/usr/bin/env python3
"""
Build WebQA shard_00014 toy split for pipeline testing.

Creates pipeline input files (webqa_slice/) from VAL set questions
that reference images in shard_00014 (image IDs 30070000–30074999).

Default output layout (under repo ``data/webqa/``):

  WebQA_imgs_7z_chunks/webqa_shard14_toy/
    webqa_slice/
    webqa_questions.jsonl
    webqa_images.jsonl
    webqa_texts.jsonl
    webqa_tables.jsonl
  manifest.json

Does NOT copy images; references existing shard_00014 paths.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_WEBQA_DATA = _REPO_ROOT / "data" / "webqa"


SHARD14_IMG_ID_MIN = 30070000
SHARD14_IMG_ID_MAX = 30074999
SHARD14_IMG_PATH = str(
    _WEBQA_DATA / "WebQA_imgs_7z_chunks" / "imgs" / "all_png" / "shard_00014"
)

_BASELINE = (
    _WEBQA_DATA
    / "WebQA-main"
    / "baseline_output_files"
    / "Baseline_prediction_files_on_Val"
)
DEFAULT_VLP_PREDS = str(_BASELINE / "VLP_x101_combinedTraining_val_end2end_predictions.json")
DEFAULT_IMG_TSV = str(
    _BASELINE / "img_queries_VLP_x101fpn_combinedTraining_val_beam5_img_step11_cleaned.tsv"
)
DEFAULT_TXT_TSV = str(
    _BASELINE / "txt_queries_VLP_x101fpn_combinedTraining_val_beam5_txt_step11_cleaned.tsv"
)
DEFAULT_OUTPUT_ROOT = str(
    _WEBQA_DATA / "WebQA_imgs_7z_chunks" / "webqa_shard14_toy"
)


def load_vlp_predictions(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_tsv_qa(img_tsv: str, txt_tsv: str) -> dict:
    """Load TSV files and merge by Guid."""
    qa = {}
    for tsv_path in [img_tsv, txt_tsv]:
        if not os.path.isfile(tsv_path):
            continue
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                guid = row.get("Guid")
                if guid and guid not in qa:
                    qa[guid] = row
    return qa


def find_shard14_qids(vlp_preds: dict) -> list[tuple[str, list[int]]]:
    """Return list of (qid, [matched_img_ids]) for questions referencing shard_00014."""
    shard14_ids = set(range(SHARD14_IMG_ID_MIN, SHARD14_IMG_ID_MAX + 1))
    results = []
    for qid, pred in vlp_preds.items():
        sources = pred.get("predicted sources", [])
        matched = [s for s in sources if isinstance(s, int) and s in shard14_ids]
        if matched:
            results.append((qid, matched))
    return results


def img_id_to_path(img_id: int) -> str:
    """Convert image ID to file path in shard_00014."""
    # Filename format: {line_idx}_{img_id}.png
    # line_idx = img_id - 30000000 for shard_00014 range
    line_idx = img_id - 30000000
    fname = f"{line_idx:08d}_{img_id}.png"
    return os.path.join(SHARD14_IMG_PATH, fname)


def build_toy_split(
    vlp_preds: dict,
    tsv_qa: dict,
    output_root: str,
    max_questions: int | None = None,
) -> dict:
    """Build toy split files and return manifest."""
    shard14_matches = find_shard14_qids(vlp_preds)
    print(f"Found {len(shard14_matches)} VAL questions referencing shard_00014")

    # Filter to those with TSV Q+A
    valid = [(qid, imgs) for qid, imgs in shard14_matches if qid in tsv_qa]
    print(f"  {len(valid)} have Q+A in TSV")

    if max_questions and len(valid) > max_questions:
        valid = valid[:max_questions]
        print(f"  Limited to {max_questions} questions")

    # Prepare output dirs
    slice_dir = Path(output_root) / "webqa_slice"
    slice_dir.mkdir(parents=True, exist_ok=True)

    questions_out = []
    images_out = {}  # img_id -> record (dedupe)
    texts_out = []

    for qid, img_ids in valid:
        row = tsv_qa[qid]
        q_text = row.get("Q", "")
        a_text = row.get("A", "[]")
        qcate = row.get("Qcate", "Others")
        keywords_a = row.get("Keywords_A", "")

        # Parse answer (stored as JSON list string)
        try:
            answers = json.loads(a_text)
            if isinstance(answers, list) and answers:
                primary_answer = answers[0]
            else:
                primary_answer = str(answers)
        except json.JSONDecodeError:
            primary_answer = a_text

        # Build question record
        q_rec = {
            "Guid": qid,
            "Q": q_text,
            "A": [primary_answer] if isinstance(primary_answer, str) else answers,
            "Keywords_A": keywords_a,
            "Qcate": qcate,
            "split": "val",
            "img_posFacts": [],
            "txt_posFacts": [],
            "metadata": {
                "image_doc_ids": [str(i) for i in img_ids],
                "text_doc_ids": [],
            },
        }

        # Add image references
        for img_id in img_ids:
            img_path = img_id_to_path(img_id)
            if os.path.isfile(img_path):
                img_rec = {
                    "image_id": str(img_id),
                    "title": f"WebQA image {img_id}",
                    "caption": "",
                    "url": img_path,
                }
                q_rec["img_posFacts"].append(img_rec)
                images_out[img_id] = img_rec

        # Build synthetic text fact from question + answer + keywords
        # Keywords_A alone is often too short (e.g. "no", "Yes") for extraction
        synth_parts = []
        if q_text:
            synth_parts.append(f"Question: {q_text}")
        if primary_answer and str(primary_answer).lower() not in ("", "[]"):
            synth_parts.append(f"Answer: {primary_answer}")
        if keywords_a and keywords_a.lower() not in ("", "no", "yes"):
            synth_parts.append(f"Keywords: {keywords_a}")
        synth_text = " ".join(synth_parts) if synth_parts else keywords_a
        if synth_text:
            txt_rec = {
                "id": f"{qid}_txt",
                "text": synth_text,
                "source": "synth_qa",
            }
            q_rec["txt_posFacts"].append({"snippet_id": txt_rec["id"], "fact": keywords_a})
            q_rec["metadata"]["text_doc_ids"].append(txt_rec["id"])
            texts_out.append(txt_rec)

        questions_out.append(q_rec)

    # Write JSONL files
    with open(slice_dir / "webqa_questions.jsonl", "w", encoding="utf-8") as f:
        for rec in questions_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(slice_dir / "webqa_images.jsonl", "w", encoding="utf-8") as f:
        for rec in images_out.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(slice_dir / "webqa_texts.jsonl", "w", encoding="utf-8") as f:
        for rec in texts_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Empty tables placeholder
    with open(slice_dir / "webqa_tables.jsonl", "w", encoding="utf-8") as f:
        pass

    # Manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shard": "shard_00014",
        "shard_img_id_range": [SHARD14_IMG_ID_MIN, SHARD14_IMG_ID_MAX],
        "shard_img_path": SHARD14_IMG_PATH,
        "source_vlp_preds": DEFAULT_VLP_PREDS,
        "source_img_tsv": DEFAULT_IMG_TSV,
        "source_txt_tsv": DEFAULT_TXT_TSV,
        "questions_count": len(questions_out),
        "unique_images_count": len(images_out),
        "texts_count": len(texts_out),
        "question_ids": [q["Guid"] for q in questions_out],
        "image_ids": list(images_out.keys()),
    }

    with open(Path(output_root) / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build WebQA shard_00014 toy split")
    ap.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    ap.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (default: all available)",
    )
    ap.add_argument(
        "--vlp-preds",
        default=DEFAULT_VLP_PREDS,
        help="VLP combined predictions JSON",
    )
    ap.add_argument(
        "--img-tsv",
        default=DEFAULT_IMG_TSV,
        help="Image queries TSV",
    )
    ap.add_argument(
        "--txt-tsv",
        default=DEFAULT_TXT_TSV,
        help="Text queries TSV",
    )
    args = ap.parse_args()

    print(f"Loading VLP predictions from {args.vlp_preds}")
    vlp_preds = load_vlp_predictions(args.vlp_preds)
    print(f"  {len(vlp_preds)} predictions loaded")

    print(f"Loading TSV Q+A from {args.img_tsv} and {args.txt_tsv}")
    tsv_qa = load_tsv_qa(args.img_tsv, args.txt_tsv)
    print(f"  {len(tsv_qa)} Q+A records loaded")

    print(f"Building toy split in {args.output_root}")
    manifest = build_toy_split(
        vlp_preds,
        tsv_qa,
        args.output_root,
        args.max_questions,
    )

    print("\nToy split created:")
    print(f"  Questions: {manifest['questions_count']}")
    print(f"  Unique images: {manifest['unique_images_count']}")
    print(f"  Texts: {manifest['texts_count']}")
    print(f"  Output: {args.output_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
