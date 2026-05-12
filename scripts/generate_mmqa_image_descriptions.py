#!/usr/bin/env python3
"""
Write one UTF-8 text file per MMQA image row so construct.py can build IMAGE nodes.

Official MMQA image JSONL does not include captions. This generates **fallback**
descriptions from `title` and `url`, suitable until you replace files with VL captions.

Default output matches construct.py fallback when MMQA_IMAGE_DESC_DIR is unset:
  data/multimodalqa/image_descriptions/{id}.txt

Usage:

  python scripts/generate_mmqa_image_descriptions.py \\
    --images-jsonl data/multimodalqa/dataset/MMQA_images_n100.jsonl \\
    --out-dir data/multimodalqa/image_descriptions

  MMQA_IMAGE_DESC_DIR=/custom/dir python scripts/generate_mmqa_image_descriptions.py --out-dir /custom/dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Generate MMQA image description .txt stubs.")
    ap.add_argument(
        "--images-jsonl",
        type=Path,
        default=repo / "data" / "multimodalqa" / "dataset" / "MMQA_images_n100.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "data" / "multimodalqa" / "image_descriptions",
    )
    ap.add_argument(
        "--mode",
        choices=("title_url", "title"),
        default="title_url",
        help="Stub content layout (default includes Wikipedia URL line).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.images_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img_id = str(row.get("id", "")).strip()
            title = str(row.get("title", "")).strip()
            url = str(row.get("url", "")).strip()
            path = str(row.get("path", "")).strip()
            if not img_id:
                continue
            if args.mode == "title_url":
                body_parts = []
                body_parts.append(
                    title
                    if title
                    else f"(MMQA image {img_id}; file={path})"
                )
                if url:
                    body_parts.append("")
                    body_parts.append(f"Article: {url}")
                elif path:
                    body_parts.append("")
                    body_parts.append(f"File: {path}")
                body = "\n".join(body_parts).strip() + "\n"
            else:
                body = (title or path or img_id).strip() + "\n"

            txt_path = out_dir / f"{img_id}.txt"
            txt_path.write_text(body, encoding="utf-8")
            n += 1

    print(f"Wrote {n} description files under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
