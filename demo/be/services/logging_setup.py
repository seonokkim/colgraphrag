"""File logging for demo/be — writes under ``<demo/be>/logs``.

Each server session creates a new log file: ``YYYYMMDD_HHMMSS.log`` based on
server start time. All uvicorn access/error logs and application logs are
written to this file in real-time.

Aligned with ``search-main/demo/be/src/services/logging_setup.py`` session rules.
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import datetime
from pathlib import Path

_CONFIGURED = False
_SESSION_LOG_PATH: Path | None = None


def configure_demo_file_logging(be_dir: Path) -> Path:
    """Configure session-based file logging.

    Creates ``be_dir/logs/YYYYMMDD_HHMMSS.log`` using server start timestamp.
    Attaches handlers to root logger and uvicorn loggers so all output
    (access logs, errors, application logs) goes to this file.

    Returns the log file path for reference.
    """
    global _CONFIGURED, _SESSION_LOG_PATH
    if _CONFIGURED:
        return _SESSION_LOG_PATH  # type: ignore[return-value]

    logs_dir = be_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{session_ts}.log"
    _SESSION_LOG_PATH = log_path

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

    try:
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except (AttributeError, OSError):
        utf8_stdout = sys.stdout

    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    for uv_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(uv_name)
        uv_logger.handlers.clear()
        uv_logger.addHandler(file_handler)
        uv_logger.addHandler(console_handler)
        uv_logger.propagate = False

    app_logger = logging.getLogger("demo.webqa")
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = True

    logging.getLogger(__name__).info("Session log file: %s", log_path)

    _CONFIGURED = True
    return log_path
