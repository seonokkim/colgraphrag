"""
MMQA QA evaluation CLI (Phase 6): list EM / F1 plus WebQA leaderboard–aligned QA-FL / QA-Acc / QA.

Implements scoring by delegating to :mod:`evaluate_webqa_qa`; that module contains the shared
evaluation logic used for both datasets. Prefer this script in MMQA pipelines and tutorials so
discovery matches the dataset without duplicating implementation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_impl = Path(__file__).resolve().parent / "evaluate_webqa_qa.py"
_spec = importlib.util.spec_from_file_location("_evaluate_webqa_qa_shared_impl", _impl)
if _spec is None or _spec.loader is None:
    raise ImportError(str(_impl))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

main = getattr(_mod, "main")


if __name__ == "__main__":
    raise SystemExit(main())
