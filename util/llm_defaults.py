"""
Local HF Gemma 4 E4B IT defaults for ``colgraphrag_webqa``.

When the in-process backend ``VIDORE_TEXT_LLM_BACKEND=hf_gemma_4_e4b_it`` is used,
weights are loaded from the directory in ``GEMMA4_E4B_IT_MODEL_PATH``. If that
environment variable is **unset**, it defaults to:

1. ``<repo>/models/mllm/gemma-4-E4B-it`` (local-first, from ``util/download_models.py``)
2. ``/workspace/models/mllm/gemma-4-e4b-it`` (workspace fallback)

Override at any time via ``GEMMA4_E4B_IT_MODEL_PATH`` or CLI flags in drivers
(e.g. ``tests/test_2query_pipeline.py --gemma-model-path``).
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_GEMMA_PATH = "GEMMA4_E4B_IT_MODEL_PATH"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_GEMMA_PATH = _REPO_ROOT / "models" / "mllm" / "gemma-4-E4B-it"
_WORKSPACE_GEMMA_PATH = Path("/workspace/models/mllm/gemma-4-e4b-it")


def _resolve_default_gemma_path() -> str:
    """Return local-first path: <repo>/models/mllm/gemma-4-E4B-it if exists, else /workspace/..."""
    if _LOCAL_GEMMA_PATH.is_dir():
        return str(_LOCAL_GEMMA_PATH)
    return str(_WORKSPACE_GEMMA_PATH)


# Lazily computed once
DEFAULT_GEMMA4_E4B_IT_MODEL_PATH = _resolve_default_gemma_path()


def effective_gemma4_e4b_it_model_path() -> str:
    """Return the directory used for Gemma 4 E4B IT weights (env or package default)."""
    v = os.environ.get(_ENV_GEMMA_PATH, "").strip()
    return v if v else _resolve_default_gemma_path()


def ensure_default_gemma4_e4b_it_path() -> str:
    """
    If ``GEMMA4_E4B_IT_MODEL_PATH`` is unset, set it to the resolved default.

    Returns the path that will be used (existing env wins).
    """
    v = os.environ.get(_ENV_GEMMA_PATH, "").strip()
    if v:
        return v
    resolved = _resolve_default_gemma_path()
    os.environ[_ENV_GEMMA_PATH] = resolved
    return resolved
