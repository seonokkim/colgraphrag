import logging
import os

import torch
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()
_repo_root = Path(__file__).resolve().parents[1]
_root_env = _repo_root / ".env"
if _root_env.is_file():
    _override = os.getenv("LLM_DOTENV_OVERRIDE", "0").strip().lower() in {"1", "true", "yes", "on"}
    load_dotenv(_root_env, override=_override)

from util.repo_config import ensure_repo_config_applied

ensure_repo_config_applied()

LOG = logging.getLogger("webqa.request")


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def load_pretrained_model_with_fallback(model_cls, model_ref: str, **kwargs):
    """Prefer CUDA auto placement, then fall back to CPU."""
    if torch.cuda.is_available():
        try:
            return model_cls.from_pretrained(model_ref, device_map="auto", **kwargs)
        except Exception as exc:
            print(f"CUDA load failed for {model_ref}; falling back to CPU. reason={exc}")
    return model_cls.from_pretrained(model_ref, device_map={"": "cpu"}, **kwargs)


def _is_non_empty_answer(content: str | None) -> bool:
    """Strict empty-answer guard.

    An answer is considered empty when:
    * ``None`` / whitespace-only
    * strip-lower equals ``'unknown'``
    * length after strip is below ``LLM_EMPTY_MIN_CHARS`` (default 1)
    * every non-whitespace character is punctuation (no alphanumerics)
    """
    if content is None:
        return False
    stripped = content.strip()
    if not stripped:
        return False
    if stripped.lower() == "unknown":
        return False
    min_chars = max(1, int(os.getenv("LLM_EMPTY_MIN_CHARS", "1")))
    if len(stripped) < min_chars:
        return False
    if not any(ch.isalnum() for ch in stripped):
        return False
    return True
