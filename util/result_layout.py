"""
Result directory layout helpers.

All pipeline phases read ``MMGRAPHRAG_RUN_ID``.  Nesting with a ``/`` creates
``result/<first>/<rest>/…``, so dataset roots stay separate without code forks:

- WebQA:   ``MMGRAPHRAG_RUN_ID=webqa/<stamp>_<slug>``   → ``result/webqa/<stamp>_<slug>/``
- MMQA:    ``MMGRAPHRAG_RUN_ID=multimodalqa/<stamp>_<slug>`` → ``result/multimodalqa/<stamp>_<slug>/``

Use **today’s date-time** as the folder prefix (via :func:`default_stamp`) so runs sort
chronologically under ``result/webqa/`` / ``result/multimodalqa/``, matching
``colgraphrag_webqa/result``-style layouts.

Use the same ``MMGRAPHRAG_RUN_ID`` for export / pattern / extraction / construct / inference.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from util.run_id import default_stamp

WEBQA_RESULT_PREFIX = "webqa"
MULTIMODALQA_RESULT_PREFIX = "multimodalqa"

# First path segment under result/multimodalqa/ must look like YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS_<rest>
_MULTIMODALQA_STAMP_HEAD_RE = re.compile(r"^[0-9]{8}_[0-9]{6}(_|$)")


def _safe_slug(s: str) -> str:
    return s.strip().strip("/").replace("..", "").strip("_")


def webqa_run_id(experiment_slug: str) -> str:
    s = _safe_slug(experiment_slug)
    return f"{WEBQA_RESULT_PREFIX}/{s}" if s else WEBQA_RESULT_PREFIX


def multimodalqa_run_id(experiment_slug: str) -> str:
    s = _safe_slug(experiment_slug)
    return f"{MULTIMODALQA_RESULT_PREFIX}/{s}" if s else MULTIMODALQA_RESULT_PREFIX


def webqa_stamped_run_id(slug: str = "run") -> str:
    """``webqa/{YYYYMMDD_HHMMSS}_{slug}`` (stamp first, under ``result/webqa/``)."""
    stamp = default_stamp()
    s = _safe_slug(slug)
    return f"{WEBQA_RESULT_PREFIX}/{stamp}_{s}" if s else f"{WEBQA_RESULT_PREFIX}/{stamp}"


def multimodalqa_stamped_run_id(slug: str = "run") -> str:
    """``multimodalqa/{YYYYMMDD_HHMMSS}_{slug}``."""
    stamp = default_stamp()
    s = _safe_slug(slug)
    return (
        f"{MULTIMODALQA_RESULT_PREFIX}/{stamp}_{s}"
        if s
        else f"{MULTIMODALQA_RESULT_PREFIX}/{stamp}"
    )


def default_stamped_dataset_run_id(dataset: str, slug: str = "run") -> str:
    """
    Default ``MMGRAPHRAG_RUN_ID`` when unset: stamped folder under the dataset tree.

    ``dataset`` is ``MMGRAPHRAG_DATASET`` (``webqa`` | ``mmqa`` | ``multimodalqa``).
    """
    ds = (dataset or "webqa").strip().lower()
    if ds in ("mmqa", "multimodalqa", "mm_multimodalqa"):
        return multimodalqa_stamped_run_id(slug)
    return webqa_stamped_run_id(slug)


def discover_latest_multimodalqa_run_id(repo_root: Path) -> str | None:
    """
    ``multimodalqa/<subdir>`` with an existing ``mmqa_slice/mmqa_questions.jsonl``;
    picks the subdirectory whose questions file has the newest mtime.
    """
    root = (repo_root / "result" / MULTIMODALQA_RESULT_PREFIX).resolve()
    if not root.is_dir():
        return None
    best_mtime = -1.0
    best_name: str | None = None
    for d in root.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        pq = d / "mmqa_slice" / "mmqa_questions.jsonl"
        if not pq.is_file():
            continue
        m = pq.stat().st_mtime
        if m > best_mtime:
            best_mtime = m
            best_name = d.name
    if best_name is None:
        return None
    return f"{MULTIMODALQA_RESULT_PREFIX}/{best_name}"


def discover_latest_webqa_run_id(repo_root: Path) -> str | None:
    """
    ``webqa/<subdir>`` with ``webqa_slice/webqa_questions.jsonl``; newest question file mtime wins.
    """
    root = (repo_root / "result" / WEBQA_RESULT_PREFIX).resolve()
    if not root.is_dir():
        return None
    best_mtime = -1.0
    best_name: str | None = None
    for d in root.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        pq = d / "webqa_slice" / "webqa_questions.jsonl"
        if not pq.is_file():
            continue
        m = pq.stat().st_mtime
        if m > best_mtime:
            best_mtime = m
            best_name = d.name
    if best_name is None:
        return None
    return f"{WEBQA_RESULT_PREFIX}/{best_name}"


def ensure_multimodalqa_run_id_has_stamp_prefix(run_id: str) -> str:
    """
    Normalize ``multimodalqa/<subdir>`` so ``<subdir>`` starts with ``YYYYMMDD_HHMMSS_``.

    If the first subdirectory already begins with ``YYYYMMDD_HHMMSS`` (followed by ``_``
    or end-of-segment), the id is unchanged. Otherwise ``default_stamp()`` is prepended::

        multimodalqa/planAB  -> multimodalqa/<stamp>_planAB

    Set ``MMQA_ALLOW_LEGACY_UNSTAMPED_RUN_ID=1`` to skip rewriting (reuse old dirs).

    Accepts bare ``subdir`` without the ``multimodalqa/`` prefix.
    """
    if os.getenv("MMQA_ALLOW_LEGACY_UNSTAMPED_RUN_ID", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return run_id.strip()

    pref = MULTIMODALQA_RESULT_PREFIX + "/"
    r = run_id.strip().replace("\\", "/")
    if not r:
        raise ValueError("empty MMQA run id")
    low = r.lower().rstrip("/")
    if low == MULTIMODALQA_RESULT_PREFIX:
        return f"{MULTIMODALQA_RESULT_PREFIX}/{default_stamp()}"

    if low.startswith(pref):
        remainder = r[len(MULTIMODALQA_RESULT_PREFIX) :].lstrip("/")
    elif "/" not in r:
        remainder = r
    else:
        remainder = r

    remainder = remainder.lstrip("./")
    if not remainder:
        return f"{MULTIMODALQA_RESULT_PREFIX}/{default_stamp()}"

    parts = remainder.split("/", 1)
    head = parts[0].strip("_")
    tail = parts[1] if len(parts) > 1 else ""

    if head and _MULTIMODALQA_STAMP_HEAD_RE.match(head):
        sub = head if not tail else f"{head}/{tail}"
        return f"{MULTIMODALQA_RESULT_PREFIX}/{sub}"

    stamp = default_stamp()
    new_head = f"{stamp}_{head}" if head else stamp
    if tail:
        return f"{MULTIMODALQA_RESULT_PREFIX}/{new_head}/{tail}"
    return f"{MULTIMODALQA_RESULT_PREFIX}/{new_head}"


def resolve_mmqa_bash_pipeline_run_id() -> str:
    """
    Run id for ``scripts/mmqa_pipeline_n.sh`` (reads ``MMQA_RUN_ID_OVERRIDE``, ``N_QUESTIONS``).

    Default: ``multimodalqa/{YYYYMMDD_HHMMSS}_n{N}``.
    Override: ``MMQA_RUN_ID_OVERRIDE`` may be ``planAB``, ``multimodalqa/planAB``, etc.;
    stamped via :func:`ensure_multimodalqa_run_id_has_stamp_prefix` unless already stamped.
    """
    override = os.getenv("MMQA_RUN_ID_OVERRIDE", "").strip()
    n = (os.getenv("N_QUESTIONS", "5").strip() or "5").strip()
    stamp = default_stamp()
    if not override:
        return f"{MULTIMODALQA_RESULT_PREFIX}/{stamp}_n{n}"
    rid = override if "/" in override else f"{MULTIMODALQA_RESULT_PREFIX}/{override}"
    return ensure_multimodalqa_run_id_has_stamp_prefix(rid)


def resolve_pipeline_run_id(repo_root: Path, dataset: str) -> str:
    """
    Resolve ``MMGRAPHRAG_RUN_ID`` after ``strip()`` (also handles empty-string env).

    - If set → use after MMQA normalization (``multimodalqa/…`` gets ``YYYYMMDD_HHMMSS_``).
    - Else for MMQA: reuse latest exported run under ``result/multimodalqa/``, or stamped new id.
    - Else WebQA: reuse latest run under ``result/webqa/`` if any, or stamped new id.
    """
    rid = os.getenv("MMGRAPHRAG_RUN_ID", "").strip()
    if rid:
        ds = (dataset or "webqa").strip().lower()
        if ds in ("mmqa", "multimodalqa", "mm_multimodalqa"):
            low = rid.lower()
            if not low.startswith(f"{WEBQA_RESULT_PREFIX}/"):
                return ensure_multimodalqa_run_id_has_stamp_prefix(rid)
        return rid
    ds = (dataset or "webqa").strip().lower()
    if ds in ("mmqa", "multimodalqa", "mm_multimodalqa"):
        return discover_latest_multimodalqa_run_id(repo_root) or multimodalqa_stamped_run_id(
            "run"
        )
    return discover_latest_webqa_run_id(repo_root) or webqa_stamped_run_id("run")
