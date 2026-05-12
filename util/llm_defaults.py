"""
Local HF Gemma 4 E4B IT defaults for ``colgraphrag_webqa``.

When the in-process backend ``VIDORE_TEXT_LLM_BACKEND=hf_gemma_4_e4b_it`` is used,
weights are loaded from the directory in ``GEMMA4_E4B_IT_MODEL_PATH``. If that
environment variable is **unset**, it defaults to
``<repo>/models/mllm/gemma-4-E4B-it`` (resolved absolute path, from ``util/download_models.py``).

Override at any time via ``GEMMA4_E4B_IT_MODEL_PATH`` or CLI flags in drivers
(e.g. ``tests/test_2query_pipeline.py --gemma-model-path``).
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_GEMMA_PATH = "GEMMA4_E4B_IT_MODEL_PATH"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_GEMMA_PATH = _REPO_ROOT / "models" / "mllm" / "gemma-4-E4B-it"


def _resolve_default_gemma_path() -> str:
    """
    Canonical checkpoint directory: always ``<repo>/models/mllm/gemma-4-E4B-it``
    (resolved with :func:`Path.resolve`), aligned with ``config/model.yaml`` and
    ``util/download_models.py``.  Use ``GEMMA4_E4B_IT_MODEL_PATH`` to override.
    """
    return str(_LOCAL_GEMMA_PATH.resolve())


# Lazily computed once
DEFAULT_GEMMA4_E4B_IT_MODEL_PATH = _resolve_default_gemma_path()


def effective_gemma4_e4b_it_model_path() -> str:
    """Return the directory used for Gemma 4 E4B IT weights (env or package default)."""
    v = os.environ.get(_ENV_GEMMA_PATH, "").strip()
    return v if v else _resolve_default_gemma_path()


def ensure_default_gemma4_e4b_it_path() -> str:
    """
    Canonical layout: Hugging Face snapshot under ``models/mllm/gemma-4-E4B-it``.

    When that directory exists and contains checkpoint files, **GEMMA4_E4B_IT_MODEL_PATH**
    is forced to that path for this process (overrides unrelated env leftovers) so Gemma
    always loads from the repo-local tree when weights are installed there.
    """
    resolved = _resolve_default_gemma_path()
    p = Path(resolved)
    v = os.environ.get(_ENV_GEMMA_PATH, "").strip()

    def _weights_present(repo: Path) -> bool:
        if not repo.is_dir() or not (repo / "config.json").is_file():
            return False
        if (repo / "model.safetensors").is_file() or (repo / "pytorch_model.bin").is_file():
            return True
        return (repo / "model.safetensors.index.json").is_file()

    if _weights_present(p):
        os.environ[_ENV_GEMMA_PATH] = resolved
        return resolved
    if v:
        return v
    os.environ[_ENV_GEMMA_PATH] = resolved
    return resolved
