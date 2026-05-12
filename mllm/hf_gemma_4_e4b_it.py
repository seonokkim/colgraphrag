"""
Local Hugging Face **Gemma 4 E4B IT** (``google/gemma-4-E4B-it``, image + text → text).

Default checkpoint directory (override with ``GEMMA4_E4B_IT_MODEL_PATH``):

``<repo>/models/mllm/gemma-4-E4B-it`` (from ``util/download_models.py --only gemma``).

Requires: CUDA GPU (set ``GEMMA4_ALLOW_CPU=1`` only for CPU debugging); ``torch``, ``transformers``
(Gemma 4 / ``gemma4``), ``accelerate`` (recommended for ``device_map="auto"``),
``pillow``, ``torchvision`` (per HF model card).

Weight files must be a complete HF snapshot; a truncated ``*.safetensors`` causes ``SafetensorError``.
Re-download with ``python util/download_models.py --only gemma`` if deserialization fails.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_ENV_MODEL_DIR = "GEMMA4_E4B_IT_MODEL_PATH"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_UNIX = _REPO_ROOT / "models" / "mllm" / "gemma-4-E4B-it"
_HF_REPO_ID = "google/gemma-4-E4B-it"

_ModelProcessor = Tuple[Any, Any]


def configured(model_dir: Optional[Union[str, Path]] = None) -> bool:
    """True when the Gemma 4 E4B IT weights directory is present on disk."""
    try:
        resolve_model_dir(model_dir)
    except (OSError, FileNotFoundError):
        return False
    return True


def _default_local_model_dir_for_os(os_name: str) -> Path:
    """Return ``<repo>/models/mllm/gemma-4-E4B-it`` (same layout as colgraphrag_webqa)."""
    return _LOCAL_UNIX


def default_local_model_dir() -> Path:
    """Preferred layout on disk (see ``util/download_models.py --only gemma``)."""
    return _default_local_model_dir_for_os(os.name)


def resolve_model_dir(explicit: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve directory containing ``config.json`` (HF snapshot layout).

    Precedence: ``explicit`` → env ``GEMMA4_E4B_IT_MODEL_PATH`` → OS default.
    """
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Gemma 4 E4B IT model directory not found: {p}")
        return p
    env = os.environ.get(_ENV_MODEL_DIR, "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"{_ENV_MODEL_DIR} is set but not a directory: {p}")
        return p
    p = default_local_model_dir()
    if not p.is_dir():
        raise FileNotFoundError(
            f"Gemma 4 E4B IT weights not found under {p}. "
            f"Download with python util/download_models.py --only gemma "
            f"or set {_ENV_MODEL_DIR} to a directory containing {_HF_REPO_ID} snapshot files."
        )
    return p


def build_text_only_messages(user_text: str) -> List[Dict[str, Any]]:
    """Single-turn user message with text only (pattern / extraction prompts)."""
    return [{"role": "user", "content": [{"type": "text", "text": user_text}]}]


def build_user_messages(image_path: Union[str, Path], user_text: str) -> List[Dict[str, Any]]:
    """
    Single-turn user message for Gemma 4 multimodal chat (image before text, per HF guidance).

    Local files are opened with Pillow and passed as RGB ``PIL.Image`` objects (``type: image``).
    """
    from PIL import Image

    ip = Path(image_path).expanduser().resolve()
    if not ip.is_file():
        raise FileNotFoundError(f"Image not found: {ip}")
    try:
        pil_image = Image.open(ip).convert("RGB")
    except OSError as exc:
        raise RuntimeError(f"Could not decode image: {ip}") from exc
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def load_gemma_4_e4b_it(
    model_dir: Optional[Union[str, Path]] = None,
    *,
    device_map: Optional[str] = None,
    trust_remote_code: bool = True,
) -> _ModelProcessor:
    """
    Load Gemma 4 E4B + ``AutoProcessor`` from disk via ``AutoModelForMultimodalLM``.

    ``model_dir`` defaults to :func:`resolve_model_dir`.
    Default loads on single CUDA device (``model.to("cuda")``).
    Set ``device_map="auto"`` for multi-GPU with ``accelerate``.
    """
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    root = resolve_model_dir(model_dir)
    _allow_cpu = os.getenv("GEMMA4_ALLOW_CPU", "0").strip().lower() in ("1", "true", "yes")
    if not torch.cuda.is_available():
        if _allow_cpu:
            logger.warning(
                "Loading Gemma without CUDA (GEMMA4_ALLOW_CPU=1); set GEMMA4_ALLOW_CPU=0 on GPU machines."
            )
        else:
            raise RuntimeError(
                "CUDA GPU is required to load Gemma 4 E4B IT (torch.cuda.is_available() is False). "
                "Run on a GPU environment with a working PyTorch+CUDA build, or set GEMMA4_ALLOW_CPU=1 "
                "only for intentional CPU debugging."
            )
    _dt = os.getenv("GEMMA4_E4B_IT_TORCH_DTYPE", "").strip().lower()
    if _dt in ("fp32", "float32", "f32"):
        dtype = torch.float32
    elif _dt in ("fp16", "float16", "f16"):
        dtype = torch.float16
    elif _dt in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    _attn = os.getenv("GEMMA4_ATTN_IMPLEMENTATION", "eager").strip()
    logger.info("Loading Gemma 4 E4B IT from %s (dtype=%s, attn=%s, device_map=%s)",
                root, dtype, _attn, device_map)
    processor = AutoProcessor.from_pretrained(str(root), trust_remote_code=trust_remote_code)
    kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": _attn,
    }
    dm = device_map
    if dm is not None:
        try:
            import accelerate  # noqa: F401
            kwargs["device_map"] = dm
        except ImportError:
            logger.warning("accelerate not installed; skipping device_map.")
            dm = None
    if dm is None and torch.cuda.is_available():
        kwargs["device_map"] = {"": "cuda:0"}
        dm = "cuda:0"
    model = AutoModelForMultimodalLM.from_pretrained(str(root), **kwargs)
    return model, processor


def _decode_new_tokens(processor: Any, input_len: int, out_ids: Any) -> str:
    """Decode only generated suffix; normalize via ``parse_response`` when available."""
    row = out_ids[0][input_len:]
    response = processor.decode(row, skip_special_tokens=False)
    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(response)
        if isinstance(parsed, dict) and "content" in parsed:
            return str(parsed["content"])
    return response


_INPROC_MODEL: Any = None
_INPROC_PROCESSOR: Any = None
_INPROC_ROOT: Optional[str] = None


def _ensure_inproc(
    model_dir: Optional[Union[str, Path]],
    trust_remote_code: bool,
) -> Tuple[Any, Any]:
    global _INPROC_MODEL, _INPROC_PROCESSOR, _INPROC_ROOT

    root = resolve_model_dir(model_dir)
    root_str = str(root)
    if _INPROC_ROOT != root_str or _INPROC_MODEL is None:
        _dm = os.getenv("GEMMA4_DEVICE_MAP", "").strip() or None
        _INPROC_MODEL, _INPROC_PROCESSOR = load_gemma_4_e4b_it(
            root, device_map=_dm, trust_remote_code=trust_remote_code
        )
        _INPROC_ROOT = root_str
    return _INPROC_MODEL, _INPROC_PROCESSOR


def _resolve_max_new_tokens(max_new_tokens: Optional[int]) -> int:
    if max_new_tokens is not None:
        return int(max_new_tokens)
    return int(os.getenv("GEMMA4_E4B_IT_MAX_NEW_TOKENS", "256"))


def _generate_from_messages(
    messages: List[Dict[str, Any]],
    *,
    model_dir: Optional[Union[str, Path]] = None,
    model: Any = None,
    processor: Any = None,
    max_new_tokens: Optional[int] = None,
    trust_remote_code: bool = True,
) -> str:
    import torch

    m_new = _resolve_max_new_tokens(max_new_tokens)
    if model is None or processor is None:
        model, processor = _ensure_inproc(model_dir, trust_remote_code)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    dev = getattr(model, "device", None)
    if dev is None:
        dev = next(model.parameters()).device
    inputs = inputs.to(dev)
    input_len = int(inputs["input_ids"].shape[-1])
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=m_new, do_sample=False)
    return _decode_new_tokens(processor, input_len, out_ids)


def generate_from_image(
    image_path: Union[str, Path],
    user_prompt: str,
    *,
    model_dir: Optional[Union[str, Path]] = None,
    model: Any = None,
    processor: Any = None,
    max_new_tokens: Optional[int] = None,
    trust_remote_code: bool = True,
) -> str:
    """
    One multimodal generation: image + text -> assistant text (parsed ``content`` when supported).

    If ``model`` and ``processor`` are omitted, loads weights via :func:`load_gemma_4_e4b_it`
    (cached in-process on repeated calls with the same ``model_dir``).
    """
    messages = build_user_messages(image_path, user_prompt)
    return _generate_from_messages(
        messages,
        model_dir=model_dir,
        model=model,
        processor=processor,
        max_new_tokens=max_new_tokens,
        trust_remote_code=trust_remote_code,
    )


def generate_text(
    user_prompt: str,
    *,
    model_dir: Optional[Union[str, Path]] = None,
    model: Any = None,
    processor: Any = None,
    max_new_tokens: Optional[int] = None,
    trust_remote_code: bool = True,
) -> str:
    """Text-only generation for ViDoRe pattern / extraction when HF backend is Gemma 4 E4B IT."""
    messages = build_text_only_messages(user_prompt)
    return _generate_from_messages(
        messages,
        model_dir=model_dir,
        model=model,
        processor=processor,
        max_new_tokens=max_new_tokens,
        trust_remote_code=trust_remote_code,
    )


def clear_inprocess_cache() -> None:
    """Drop cached model/processor (e.g. between tests)."""
    global _INPROC_MODEL, _INPROC_PROCESSOR, _INPROC_ROOT
    _INPROC_MODEL = None
    _INPROC_PROCESSOR = None
    _INPROC_ROOT = None
