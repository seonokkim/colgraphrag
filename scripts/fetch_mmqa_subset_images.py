#!/usr/bin/env python3
"""
Download the official MultiModalQA image zip (cached under data/multimodalqa/.cache/)
and extract only files listed in MMQA_images_*.jsonl into final_dataset_images/.

Or copy from a local folder (absolute path, or repo-relative e.g.
``data/multimodalqa/final_dataset_images``).

Tries ``<root>/<basename>`` and ``<root>/final_dataset_images/<basename>`` (nested unzip).

Zip mode does not re-download if a valid cache zip exists. Resume: ``curl -C - ...``.

Usage (zip):

  python scripts/fetch_mmqa_subset_images.py \\
    --images-jsonl data/multimodalqa/dataset/MMQA_images_n100.jsonl \\
    --dest data/multimodalqa/final_dataset_images

Usage (local):

  python scripts/fetch_mmqa_subset_images.py \\
    --copy-from data/multimodalqa/final_dataset_images
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DEFAULT_ZIP_URL = (
    "https://multimodalqa-images.s3-us-west-2.amazonaws.com/"
    "final_dataset_images/final_dataset_images.zip"
)


def basenames_from_jsonl(path: Path) -> list[str]:
    names: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            p = str(row.get("path", "")).strip()
            if p:
                names.append(Path(p).name)
    return sorted(set(names))


def resolve_zip_member(zf: zipfile.ZipFile, basename: str) -> str | None:
    namelist = zf.namelist()
    as_set = set(namelist)
    for prefix in ("", "final_dataset_images/", "imgs/", "images/"):
        cand = prefix + basename
        if cand in as_set:
            return cand
    for n in namelist:
        if n.rstrip("/").endswith("/" + basename) or n.endswith(basename):
            return n
    return None


def ensure_zip(zip_path: Path, url: str, use_curl: bool) -> None:
    if zip_path.is_file() and zip_path.stat().st_size > 1000:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if zf.testzip() is None:
                    print(f"Using existing zip: {zip_path}")
                    return
        except zipfile.BadZipFile:
            print(f"Cache zip corrupt, will re-download: {zip_path}", file=sys.stderr)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading (large, ~2.2GiB): {url}\n -> {zip_path}")
    if use_curl:
        cmd = [
            "curl",
            "-fL",
            "--retry",
            "3",
            "-C",
            "-",
            "-o",
            str(zip_path),
            url,
        ]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(r.returncode)
    else:
        import urllib.request

        urllib.request.urlretrieve(url, zip_path)


def copy_from_local(
    src_root: Path,
    basenames: list[str],
    dest: Path,
) -> tuple[int, list[str]]:
    """Copy basenames from disk; try root and root/final_dataset_images."""
    dest.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    ok = 0
    for base in basenames:
        candidates = [
            src_root / base,
            src_root / "final_dataset_images" / base,
        ]
        src: Path | None = None
        for c in candidates:
            if c.is_file():
                src = c
                break
        if src is None:
            missing.append(base)
            continue
        shutil.copy2(src, dest / base)
        ok += 1
    return ok, missing


def extract_subset(
    zip_path: Path,
    basenames: list[str],
    dest: Path,
) -> tuple[int, int, list[str]]:
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    missing_members: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for base in basenames:
            member = resolve_zip_member(zf, base)
            if member is None:
                missing_members.append(base)
                continue
            info = zf.getinfo(member)
            if info.is_dir():
                missing_members.append(base)
                continue
            out_file = dest / base
            with zf.open(member, "r") as src, open(out_file, "wb") as dst:
                shutil.copyfileobj(src, dst)
            ok += 1
    return ok, len(missing_members), missing_members


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract MMQA images subset from official zip.")
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--images-jsonl",
        type=Path,
        default=repo / "data" / "multimodalqa" / "dataset" / "MMQA_images_n100.jsonl",
    )
    ap.add_argument(
        "--dest",
        type=Path,
        default=repo / "data" / "multimodalqa" / "final_dataset_images",
    )
    ap.add_argument(
        "--zip-cache",
        type=Path,
        default=repo / "data" / "multimodalqa" / ".cache" / "final_dataset_images.zip",
    )
    ap.add_argument("--zip-url", default=DEFAULT_ZIP_URL)
    ap.add_argument(
        "--copy-from",
        type=Path,
        default=None,
        help="Local directory with MMQA images (skip zip download). "
        "Tries <root>/<file> and <root>/final_dataset_images/<file>.",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="With zip mode only: require existing --zip-cache; only extract.",
    )
    ap.add_argument(
        "--no-curl",
        action="store_true",
        help="Use urllib instead of curl (no resume).",
    )
    args = ap.parse_args()

    basenames = basenames_from_jsonl(args.images_jsonl.resolve())
    if not basenames:
        print("No paths in images jsonl.", file=sys.stderr)
        return 2

    dest = args.dest.resolve()

    if args.copy_from is not None:
        root = args.copy_from.resolve()
        if not root.is_dir():
            print(f"Not a directory: {root}", file=sys.stderr)
            return 2
        ok, miss = copy_from_local(root, basenames, dest)
        print(f"Copied {ok} files from {root} -> {dest}")
        if miss:
            print(
                f"Missing on disk ({len(miss)}): {miss[:10]}{'...' if len(miss) > 10 else ''}",
                file=sys.stderr,
            )
            return 1
        return 0

    zip_path = args.zip_cache.resolve()
    if not args.skip_download:
        ensure_zip(zip_path, args.zip_url, use_curl=not args.no_curl)
    elif not zip_path.is_file():
        print(f"Missing zip: {zip_path}", file=sys.stderr)
        return 2

    ok, nmiss, miss = extract_subset(zip_path, basenames, dest)
    print(f"Extracted {ok} files -> {args.dest}")
    if miss:
        print(f"Missing in zip ({nmiss}): {miss[:10]}{'...' if len(miss) > 10 else ''}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
