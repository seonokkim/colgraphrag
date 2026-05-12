"""
Ollama **gemma4:e2b** (text or image + text → text) via the local Ollama HTTP API.

Requires a running ``ollama serve`` and a pulled model tag (default ``gemma4:e2b``).

Environment:

- ``OLLAMA_HOST`` — API base URL (default from ollama-python: ``http://127.0.0.1:11434``).
- ``OLLAMA_GEMMA4_E2B_MODEL`` — model tag (default ``gemma4:e2b``). Mirrors ``config/model.yaml`` ``ollama.model`` via :func:`util.repo_config.ensure_repo_config_applied` when unset.

Install locally: ``pip install ollama`` (inside the repo venv is fine).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import ollama
from ollama import Client

_ENV_MODEL = "OLLAMA_GEMMA4_E2B_MODEL"
_DEFAULT_MODEL = "gemma4:e2b"


def default_model() -> str:
    """Resolved model tag: env ``OLLAMA_GEMMA4_E2B_MODEL`` or ``gemma4:e2b``."""
    m = os.getenv(_ENV_MODEL, "").strip()
    return m or _DEFAULT_MODEL


def resolve_client(host: Optional[str] = None) -> Client:
    """``ollama.Client``; respects ``OLLAMA_HOST`` when ``host`` is None."""
    if host is None or not str(host).strip():
        return Client()
    return Client(host=str(host).strip())


def configured(
    *,
    model: Optional[str] = None,
    host: Optional[str] = None,
) -> bool:
    """True when the API is reachable and the model appears in ``ollama list``."""
    name = model or default_model()
    try:
        cli = resolve_client(host)
        lst = cli.list()
        models = lst.models if lst else []
        return any(getattr(x, "model", None) == name for x in models)
    except (ConnectionError, OSError):  # pragma: no cover - environment-specific
        return False
    except Exception:  # pragma: no cover - network / parsing
        return False


def chat(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    host: Optional[str] = None,
    client: Optional[Client] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Raw chat helper: returns assistant message text only (non-streaming).
    ``messages`` use Ollama's chat schema (including optional ``images`` per user turn).
    """
    cli = client or resolve_client(host)
    m = model or default_model()
    kwargs: Dict[str, Any] = {"model": m, "messages": messages, "stream": False}
    if options:
        kwargs["options"] = dict(options)
    out = cli.chat(**kwargs)
    return (out.message.content or "").strip()


def generate_text(
    user_prompt: str,
    *,
    model: Optional[str] = None,
    host: Optional[str] = None,
    client: Optional[Client] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> str:
    """Text-only Q&A: single user turn."""
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    return chat(messages, model=model, host=host, client=client, options=options)


def generate_from_image(
    image_path: Union[str, Path],
    user_prompt: str,
    *,
    model: Optional[str] = None,
    host: Optional[str] = None,
    client: Optional[Client] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Image + text Q&A. ``image_path`` is read as bytes and passed in the ``images`` field.
    """
    ip = Path(image_path).expanduser().resolve()
    if not ip.is_file():
        raise FileNotFoundError(f"Image not found: {ip}")
    data = ip.read_bytes()
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": user_prompt,
            "images": [data],
        }
    ]
    return chat(messages, model=model, host=host, client=client, options=options)
