"""
Per-run session logs under ``<repo>/logs`` with a ``YYYYMMDD_HHMMSS_`` prefix.

Set ``MMGRAPHRAG_LOG_DIR`` to override the logs directory (default: repo ``logs/``).
Set ``MMGRAPHRAG_DISABLE_SESSION_LOG=1`` to skip stdout/stderr tee when running a driver.

When tee is active, ``MMGRAPHRAG_SESSION_LOG_PATH`` is set to the log file so
``inference.setup_logger`` can avoid opening a second file.
"""

from __future__ import annotations

import io
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

_REPO_ROOT = Path(__file__).resolve().parents[1]

T = TypeVar("T")


def repo_logs_dir(repo_root: Path | None = None) -> Path:
    """Resolve ``logs`` directory: ``MMGRAPHRAG_LOG_DIR`` or absolute ``<repo>/logs``."""
    override = os.getenv("MMGRAPHRAG_LOG_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    root = repo_root if repo_root is not None else _REPO_ROOT
    return (root / "logs").resolve()


def session_log_filename(stem: str) -> str:
    """``YYYYMMDD_HHMMSS_<safe_stem>.log``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem.strip()) or "run"
    return f"{stamp}_{safe}.log"


def new_session_log_path(repo_root: Path | None, stem: str) -> Path:
    d = repo_logs_dir(repo_root)
    d.mkdir(parents=True, exist_ok=True)
    return d / session_log_filename(stem)


class TeeTextStream(io.TextIOBase):
    """Duplicate writes to a primary stream (e.g. terminal) and a log file."""

    def __init__(self, primary: io.TextIOBase, secondary: io.TextIOBase) -> None:
        super().__init__()
        self._primary = primary
        self._secondary = secondary

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", None) or "utf-8"

    def write(self, s: str) -> int:
        self._primary.write(s)
        self._secondary.write(s)
        self._primary.flush()
        self._secondary.flush()
        return len(s)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return bool(self._primary.isatty())


def run_with_session_stdio_tee(
    repo_root: Path,
    session_stem: str,
    fn: Callable[[], T],
) -> T:
    """
    Tee ``stdout`` and ``stderr`` to ``logs/YYYYMMDD_HHMMSS_<session_stem>.log``.

    Sets ``MMGRAPHRAG_SESSION_LOG_PATH`` for the process duration.
    """
    if os.getenv("MMGRAPHRAG_DISABLE_SESSION_LOG", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return fn()

    path = new_session_log_path(repo_root, session_stem)
    prev_path = os.environ.get("MMGRAPHRAG_SESSION_LOG_PATH", "")
    os.environ["MMGRAPHRAG_SESSION_LOG_PATH"] = str(path.resolve())
    f = path.open("w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = TeeTextStream(old_out, f)
    sys.stderr = TeeTextStream(old_err, f)
    print(f"[session] log_file={path.resolve()}", flush=True)
    try:
        return fn()
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        f.close()
        if prev_path:
            os.environ["MMGRAPHRAG_SESSION_LOG_PATH"] = prev_path
        else:
            os.environ.pop("MMGRAPHRAG_SESSION_LOG_PATH", None)
