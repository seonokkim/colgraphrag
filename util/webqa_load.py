"""
WebQA helpers for colgraphrag_webqa (no dependency on sibling project folders).

Used by pattern.py and export_webqa_slice.py for stable val/test slicing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from util.repo_config import ensure_repo_config_applied

ensure_repo_config_applied()

WebQAProfile = Literal["val_n100", "test_full"]


_REPO_ROOT = Path(__file__).resolve().parents[1]


def default_webqa_data_root() -> Path:
    env = os.getenv("WEBQA_DATA_ROOT", "").strip()
    if env:
        return Path(env)
    return _REPO_ROOT / "data" / "webqa"


def default_train_val_path() -> Path:
    return Path(
        os.getenv(
            "WEBQA_TRAIN_VAL_JSON",
            str(
                default_webqa_data_root()
                / "WebQA_data_first_release"
                / "WebQA_train_val.json"
            ),
        )
    )


def default_test_path() -> Path:
    return Path(
        os.getenv(
            "WEBQA_TEST_JSON",
            str(
                default_webqa_data_root()
                / "WebQA_data_first_release"
                / "WebQA_test.json"
            ),
        )
    )


def default_pattern_json_path() -> Path:
    """Default `PATTERN_JSON_FILE_PATH` from `WEBQA_RUN_PROFILE` / `WEBQA_JSON_FILE` inference."""
    jf = os.getenv("WEBQA_JSON_FILE", "").strip()
    if jf:
        return Path(jf)
    return default_test_path() if resolve_profile() == "test_full" else default_train_val_path()


def is_webqa_json_path(path: str) -> bool:
    """Check if path is a WebQA data file (JSON or JSONL)."""
    p = path.replace("\\", "/").lower()
    return "webqa" in p and (p.endswith(".json") or p.endswith(".jsonl"))


def is_test_split_json(path: str) -> bool:
    return "webqa_test" in path.replace("\\", "/").lower()


def resolve_profile() -> WebQAProfile:
    """
    Which WebQA slice to run. Default is **test_full** (entire `WebQA_test.json`).

    Set `WEBQA_RUN_PROFILE=val_n100` for the 100-example validation slice only.
    """
    p = os.getenv("WEBQA_RUN_PROFILE", "").strip().lower()
    if p in ("test", "test_full"):
        return "test_full"
    if p in ("val", "val_n100", "validation", "dev"):
        return "val_n100"
    jf = os.getenv("WEBQA_JSON_FILE", "").strip()
    if jf and is_test_split_json(jf):
        return "test_full"
    if jf and "train_val" in jf.replace("\\", "/").lower():
        return "val_n100"
    return "test_full"


def load_webqa_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict root in {path}")
    return raw


def records_for_pattern(json_path: str | Path, profile: WebQAProfile | None = None) -> list[dict[str, Any]]:
    """
    Return ordered list of question dicts (full WebQA rows) for pattern generation.

    - val_n100: split == val, sorted by Guid, first 100 if PATTERN_MAX_SAMPLES == 100 else cap by env.
    - test_full: all test rows, sorted by Guid; cap only if PATTERN_MAX_SAMPLES > 0.
    - Supports both JSON dict format and JSONL format.
    """
    json_path = Path(json_path)
    
    # Handle JSONL format (one JSON object per line)
    if str(json_path).lower().endswith(".jsonl"):
        with json_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        data = load_webqa_dict(json_path)
        rows = list(data.values())
    test_mode = is_test_split_json(str(json_path))
    if test_mode:
        pool = [r for r in rows if r.get("split") == "test" or "txt_Facts" in r]
        if not pool:
            pool = rows
    else:
        pool = [r for r in rows if r.get("split") == "val"]
    pool.sort(key=lambda r: str(r.get("Guid", "")))
    max_samples = int(os.getenv("PATTERN_MAX_SAMPLES", "0"))
    prof = profile or resolve_profile()
    if prof == "test_full" and not test_mode:
        raise ValueError(
            "WEBQA_RUN_PROFILE implies test_full but JSON path is not WebQA_test.json. "
            "Set PATTERN_JSON_FILE_PATH (or WEBQA_JSON_FILE) to "
            ".../WebQA_data_first_release/WebQA_test.json"
        )
    if prof == "val_n100" and test_mode:
        raise ValueError(
            "WEBQA_RUN_PROFILE=val_n100 requires WebQA_train_val.json, not WebQA_test.json."
        )
    if prof == "val_n100" and max_samples > 0:
        return pool[:max_samples]
    if prof == "test_full":
        if max_samples > 0:
            return pool[:max_samples]
        return pool
    if max_samples > 0:
        return pool[:max_samples]
    return pool
