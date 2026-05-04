"""
Datetime stamps for ``result/<RUN_ID>/`` artifact directories.

Default format: ``YYYYMMDD_HHMMSS`` (optionally ``YYYYMMDD_HHMMSS_<suffix>``).

Override with ``MMGRAPHRAG_RUN_ID`` for any phase script.
"""

from __future__ import annotations

from datetime import datetime

STAMP_FORMAT = "%Y%m%d_%H%M%S"


def default_stamp() -> str:
    """Return current local time as ``YYYYMMDD_HHMMSS``."""
    return datetime.now().strftime(STAMP_FORMAT)


def stamped_run_id(suffix: str) -> str:
    """``{YYYYMMDD_HHMMSS}_{suffix}`` with normalized spacing."""
    s = suffix.strip().strip("_")
    return f"{default_stamp()}_{s}" if s else default_stamp()
