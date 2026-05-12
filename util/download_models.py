#!/usr/bin/env python3
"""
Download model weights from ``config/model.yaml``: Hugging Face snapshots into ``models/``,
and optional Ollama models via ``ollama pull`` (Ollama default store, not under ``models/``).

Loads secrets from the repo-root ``.env`` (``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN``).

Example::

    cd /path/to/colgraphrag_webqa
    python util/download_models.py
    python util/download_models.py --only gemma colembed
    python util/download_models.py --only ollama
    python util/download_models.py --dry-run

Layout after download::

    models/
      mllm/gemma-4-E4B-it/          (google/gemma-4-E4B-it)
      retriever/llama-nemotron-.../  (nvidia/llama-nemotron-colembed-vl-3b-v2)
      eval/bart-large-cnn/           (facebook/bart-large-cnn)

    Ollama pulls (tags under ``ollama`` in config, e.g. ``model``, ``model_e4b``) are not copied under ``models/``;
    they stay in Ollama's default storage (typically ``~/.ollama``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS_ROOT = _REPO_ROOT / "models"
_CONFIG_YAML = _REPO_ROOT / "config" / "model.yaml"

# Subdirectory mapping by component
_SUBDIR_MAP = {
    "gemma": "mllm",
    "colembed": "retriever",
    "fluency": "eval",
}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env")


def _hf_token() -> str | None:
    t = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
    )
    return t or None


def _load_model_yaml() -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required. Install with: pip install PyYAML"
        ) from e
    if not _CONFIG_YAML.is_file():
        raise SystemExit(f"Missing {_CONFIG_YAML}")
    with _CONFIG_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _repo_local_dir(repo_id: str, component: str) -> Path:
    # Use Hub repo name only (segment after "/"); place under subdir by component.
    tail = repo_id.strip().split("/")[-1]
    subdir = _SUBDIR_MAP.get(component, "")
    if subdir:
        return _MODELS_ROOT / subdir / tail
    return _MODELS_ROOT / tail


def _colembed_repo_id(cfg: dict) -> str | None:
    c = cfg.get("colembed") or {}
    if not isinstance(c, dict):
        return None
    mid = (c.get("model_id") or "").strip()
    if mid:
        return mid
    rid = (c.get("hf_repo_id") or "").strip()
    return rid or None


def _ollama_pull_tags(cfg: dict) -> list[str]:
    """Distinct Ollama model tags from ``ollama.model``, ``ollama.model_e4b``, etc. (order preserved)."""
    oll = cfg.get("ollama") or {}
    if not isinstance(oll, dict):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for key in ("model", "model_e4b"):
        t = (oll.get(key) or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _iter_jobs(cfg: dict) -> list[tuple[str, str]]:
    """Return list of (repo_id, human_label)."""
    jobs: list[tuple[str, str]] = []

    g = cfg.get("gemma") or {}
    if isinstance(g, dict):
        rid = (g.get("hf_repo_id") or "").strip()
        if rid:
            jobs.append((rid, "gemma (pattern / extraction / inference)"))

    cr = _colembed_repo_id(cfg)
    if cr:
        jobs.append((cr, "colembed (MaxSim retrieval)"))

    flu = cfg.get("fluency") or {}
    if isinstance(flu, dict):
        rid = (flu.get("model") or "").strip()
        if rid:
            jobs.append((rid, "fluency (BARTScore)"))

    for mid in _ollama_pull_tags(cfg):
        jobs.append((mid, "ollama (default ~/.ollama store)"))

    return jobs


def _component(name: str) -> str:
    n = name.lower()
    if "ollama" in n:
        return "ollama"
    if "gemma" in n and "colembed" not in n:
        return "gemma"
    if "colembed" in n:
        return "colembed"
    if "fluency" in n or "bart" in n:
        return "fluency"
    return "unknown"


def _download(
    repo_id: str,
    local_dir: Path,
    *,
    token: str | None,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[dry-run] would download {repo_id!r} -> {local_dir}")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
    )
    print(f"Done: {repo_id} -> {local_dir}")


def _download_ollama(model: str, *, dry_run: bool) -> None:
    """Run ``ollama pull``; uses Ollama's default model directory (no custom output path)."""
    if dry_run:
        default_hint = Path.home() / ".ollama"
        print(
            f"[dry-run] would run: ollama pull {model!r} "
            f"(default store ~ {default_hint})"
        )
        return
    if not shutil.which("ollama"):
        raise SystemExit(
            "ollama CLI not found in PATH. Install from https://ollama.com"
        )
    print("(Using Ollama default model directory; not writing under repo models/.)")
    try:
        subprocess.run(
            ["ollama", "pull", model],
            check=True,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"Failed to run ollama: {e}") from e
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ollama pull {model!r} failed (exit {e.returncode})") from e
    print(f"Done: ollama pull {model}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download models from config/model.yaml (HF snapshots into models/; Ollama via ollama pull)"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        choices=("gemma", "colembed", "fluency", "ollama"),
        help="Subset to download (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads only",
    )
    args = parser.parse_args()

    _load_dotenv()
    token = _hf_token()
    only_set = set(args.only) if args.only else None
    needs_hf_token = only_set is None or not only_set <= {"ollama"}
    if not token and not args.dry_run and needs_hf_token:
        print(
            "Warning: HF_TOKEN / HUGGING_FACE_HUB_TOKEN not set in environment or .env. "
            "Gated models (e.g. Gemma) may fail.",
            file=sys.stderr,
        )

    cfg = _load_model_yaml()
    raw_jobs = _iter_jobs(cfg)
    jobs: list[tuple[str, str, str]] = []
    for repo_id, label in raw_jobs:
        comp = _component(label)
        jobs.append((comp, repo_id, label))

    if not jobs:
        print(
            "No hf_repo_id / model / ollama entries found in config/model.yaml",
            file=sys.stderr,
        )
        return 1

    only = set(args.only) if args.only else None

    ran = 0
    for comp, repo_id, label in jobs:
        if only is not None and comp not in only:
            continue
        print(f"--- {label}: {repo_id}")
        if comp == "ollama":
            _download_ollama(repo_id, dry_run=args.dry_run)
        else:
            dest = _repo_local_dir(repo_id, comp)
            _download(repo_id, dest, token=token, dry_run=args.dry_run)
        ran += 1

    if ran == 0:
        print("No matching components for --only filter.", file=sys.stderr)
        return 1

    if not args.dry_run:
        print()
        gemma_rid = ((cfg.get("gemma") or {}).get("hf_repo_id") or "").strip()
        col_rid = _colembed_repo_id(cfg)
        flu_rid = ((cfg.get("fluency") or {}).get("model") or "").strip()
        hf_hints = bool(gemma_rid or col_rid or flu_rid)
        if hf_hints:
            print("Example environment overrides (local snapshots under models/):")
            if gemma_rid:
                print(f"  GEMMA4_E4B_IT_MODEL_PATH={_repo_local_dir(gemma_rid, 'gemma')}")
            if col_rid:
                print(f"  COLEMBED_MODEL_PATH={_repo_local_dir(col_rid, 'colembed')}")
            if flu_rid:
                print(f"  WEBQA_FLUENCY_MODEL={_repo_local_dir(flu_rid, 'fluency')}")
        oll_tags = _ollama_pull_tags(cfg)
        if oll_tags:
            if hf_hints:
                print()
            home = Path.home() / ".ollama"
            for t in oll_tags:
                print(
                    f"Ollama model {t!r}: stored under the default Ollama directory ({home}), "
                    "not repo models/."
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
