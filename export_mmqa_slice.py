"""
Export MMQA toy slice to pipeline-standard JSONL files under result/<RUN_ID>/mmqa_slice/.

MMQA data is already pipeline-compatible (qid/question/metadata fields match).
This script normalises file names and records a build manifest.

Input:  data/multimodalqa/dataset/MMQA_{split}_n{N}.jsonl  (and texts/images/tables)
Output: result/multimodalqa/<YYYYMMDD_HHMMSS>_mmqa_export/mmqa_slice/ when MMGRAPHRAG_RUN_ID is unset (see util/result_layout.py).
  mmqa_questions.jsonl
  mmqa_texts.jsonl
  mmqa_images.jsonl
  mmqa_tables.jsonl
  mmqa_export_meta.json

Env:
  MMQA_SPLIT         dev (default) | train | test
  MMQA_DATA_DIR      path to dataset dir (default: <repo>/data/multimodalqa/dataset)
  MMGRAPHRAG_RUN_ID  shared run ID
  MMQA_SLICE_DIR     override output directory
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from util.result_layout import ensure_multimodalqa_run_id_has_stamp_prefix, multimodalqa_stamped_run_id


def _find_mmqa_file(data_dir: Path, glob_prefix: str) -> Path | None:
    """Return newest matching MMQA_<prefix>_n*.jsonl, or None."""
    for p in sorted(data_dir.glob(f"{glob_prefix}_n*.jsonl"), reverse=True):
        return p
    return None


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    base = Path(__file__).resolve().parent
    raw = os.getenv("MMGRAPHRAG_RUN_ID", "").strip()
    run_id = (
        ensure_multimodalqa_run_id_has_stamp_prefix(raw)
        if raw
        else multimodalqa_stamped_run_id("mmqa_export")
    )
    split = os.getenv("MMQA_SPLIT", "dev").strip().lower()

    data_dir_env = os.getenv("MMQA_DATA_DIR", "").strip()
    data_dir = Path(data_dir_env) if data_dir_env else base / "data" / "multimodalqa" / "dataset"

    out_dir = Path(os.getenv("MMQA_SLICE_DIR", str(base / "result" / run_id / "mmqa_slice")))
    out_dir.mkdir(parents=True, exist_ok=True)

    q_src = _find_mmqa_file(data_dir, f"MMQA_{split}")
    if q_src is None:
        raise FileNotFoundError(f"No MMQA_{split}_n*.jsonl found in {data_dir}")

    context_files = {
        "tables": _find_mmqa_file(data_dir, "MMQA_tables"),
        "texts":  _find_mmqa_file(data_dir, "MMQA_texts"),
        "images": _find_mmqa_file(data_dir, "MMQA_images"),
    }
    for label, path in context_files.items():
        if path is None:
            raise FileNotFoundError(f"No MMQA_{label}_n*.jsonl found in {data_dir}")

    counts: dict[str, int] = {}

    for out_name, src in [
        ("mmqa_questions.jsonl", q_src),
        ("mmqa_texts.jsonl",     context_files["texts"]),
        ("mmqa_images.jsonl",    context_files["images"]),
        ("mmqa_tables.jsonl",    context_files["tables"]),
    ]:
        key = out_name.split("_")[1].split(".")[0]  # "questions" / "texts" / "images" / "tables"
        n = 0
        with (out_dir / out_name).open("w", encoding="utf-8") as fout:
            for row in _iter_jsonl(src):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        counts[key] = n

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "source_data_dir": str(data_dir.resolve()),
        "source_questions_file": str(q_src),
        "n_questions": counts["questions"],
        "n_texts": counts["texts"],
        "n_images": counts["images"],
        "n_tables": counts["tables"],
        "out_dir": str(out_dir.resolve()),
    }
    (out_dir / "mmqa_export_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Wrote {counts['questions']} questions, {counts['texts']} texts, "
        f"{counts['images']} images, {counts['tables']} tables → {out_dir}"
    )


if __name__ == "__main__":
    from pathlib import Path

    from util.pipeline_session_log import run_with_session_stdio_tee

    run_with_session_stdio_tee(Path(__file__).resolve().parent, "export_mmqa_slice", main)
