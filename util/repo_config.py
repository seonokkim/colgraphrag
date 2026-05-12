"""
Load ``config/data.yaml`` and ``config/model.yaml`` into environment defaults.

Also hosts **strict real pipeline** helpers (``MMGRAPHRAG_STRICT_REAL``, default on)
used by ``pattern.py`` / ``extraction.py`` / ``inference.py``—see
``forbid_dry_run``, ``require_gemma_local``, ``require_cuda``,
``require_colembed_for_inference``.

Precedence: explicit process environment and ``.env`` (via dotenv) win; YAML only
fills missing keys.
"""

from __future__ import annotations

import os
from pathlib import Path

_APPLIED = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((repo_root() / p).resolve())


def _setdefault_env(key: str, value: str | int | bool | None) -> None:
    if value is None or value == "":
        return
    if os.getenv(key, "").strip():
        return
    if isinstance(value, bool):
        os.environ[key] = "1" if value else "0"
    else:
        os.environ[key] = str(value)


def ensure_repo_config_applied() -> None:
    """Load dotenv (if available) then YAML defaults once per process."""
    global _APPLIED
    if _APPLIED:
        return

    try:
        from dotenv import load_dotenv

        load_dotenv()
        rt = repo_root()
        root_env = rt / ".env"
        if root_env.is_file():
            override = os.getenv("LLM_DOTENV_OVERRIDE", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            load_dotenv(root_env, override=override)
    except ImportError:
        pass

    try:
        import yaml
    except ImportError:
        _APPLIED = True
        return

    cfg_dir = repo_root() / "config"
    data_path = cfg_dir / "data.yaml"
    model_path = cfg_dir / "model.yaml"

    if data_path.is_file():
        with data_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        w = data.get("webqa")
        if isinstance(w, dict):
            dr = w.get("data_root")
            if dr:
                resolved = _resolve_path(str(dr).strip())
                if resolved and Path(resolved).is_dir():
                    _setdefault_env("WEBQA_DATA_ROOT", resolved)
            for yaml_key, env_key in (
                ("train_val_json", "WEBQA_TRAIN_VAL_JSON"),
                ("test_json", "WEBQA_TEST_JSON"),
            ):
                v = w.get(yaml_key)
                if v:
                    r = _resolve_path(v)
                    _setdefault_env(env_key, r or str(v))
        pat = data.get("pattern")
        if isinstance(pat, dict):
            jf = pat.get("json_file")
            if jf:
                r = _resolve_path(jf)
                _setdefault_env("WEBQA_JSON_FILE", r or str(jf))

    if model_path.is_file():
        with model_path.open(encoding="utf-8") as f:
            model = yaml.safe_load(f) or {}
        g = model.get("gemma")
        if isinstance(g, dict):
            p = g.get("e4b_it_path")
            if p:
                r = _resolve_path(p)
                # Always point at repo ``models/mllm/gemma-4-E4B-it`` when configured (explicit env wins).
                if r:
                    _setdefault_env("GEMMA4_E4B_IT_MODEL_PATH", r)
        c = model.get("colembed")
        if isinstance(c, dict):
            mp = c.get("model_path")
            if mp:
                r = _resolve_path(mp)
                # Only set if resolved path exists (local-first)
                if r and Path(r).is_dir():
                    _setdefault_env("COLEMBED_MODEL_PATH", r)
            mid = c.get("model_id")
            if not mid:
                mid = c.get("hf_repo_id")
            if mid:
                _setdefault_env("COLEMBED_MODEL_ID", str(mid))
            if "trust_remote_code" in c:
                _setdefault_env("COLEMBED_TRUST_REMOTE_CODE", bool(c["trust_remote_code"]))
            if c.get("max_input_tiles") is not None:
                _setdefault_env("COLEMBED_MAX_INPUT_TILES", c["max_input_tiles"])
            if c.get("topk_images") is not None:
                _setdefault_env("COLEMBED_TOPK_IMAGES", c["topk_images"])
        flu = model.get("fluency")
        if isinstance(flu, dict) and flu.get("model"):
            _setdefault_env("WEBQA_FLUENCY_MODEL", str(flu["model"]))
        olla = model.get("ollama")
        if isinstance(olla, dict):
            m_e2 = (olla.get("model") or "").strip()
            if m_e2:
                _setdefault_env("OLLAMA_GEMMA4_E2B_MODEL", m_e2)
            m_e4 = (olla.get("model_e4b") or "").strip()
            if m_e4:
                _setdefault_env("OLLAMA_GEMMA4_E4B_MODEL", m_e4)

    _APPLIED = True


def strict_real_enabled() -> bool:
    v = os.getenv("MMGRAPHRAG_STRICT_REAL", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def forbid_dry_run(phase: str, *, dry_run: bool) -> None:
    if not strict_real_enabled() or not dry_run:
        return
    raise SystemExit(
        f"[{phase}] MMGRAPHRAG_STRICT_REAL=1 (default): dry-run is not allowed. "
        "Install Gemma (python util/download_models.py --only gemma), use GPU/CUDA as needed, "
        "and unset PATTERN_DRY_RUN / EXTRACTION_DRY_RUN / INFERENCE_DRY_RUN. "
        "For placeholders only, set MMGRAPHRAG_STRICT_REAL=0."
    )


def require_gemma_local(phase: str) -> None:
    if not strict_real_enabled():
        return
    try:
        from mllm import hf_gemma_4_e4b_it as gemma
    except ImportError as e:
        raise SystemExit(f"[{phase}] Cannot import Gemma module: {e}") from e
    if not gemma.configured():
        from util.llm_defaults import effective_gemma4_e4b_it_model_path

        p = effective_gemma4_e4b_it_model_path()
        raise SystemExit(
            f"[{phase}] Gemma 4 E4B IT weights not found under {p}. "
            "Run: .venv/bin/python util/download_models.py --only gemma"
        )


def require_cuda(phase: str) -> None:
    if not strict_real_enabled():
        return
    if os.getenv("MMGRAPHRAG_REQUIRE_CUDA", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            f"[{phase}] CUDA is required when MMGRAPHRAG_STRICT_REAL=1 "
            "(set MMGRAPHRAG_REQUIRE_CUDA=0 to allow CPU, not recommended for Gemma 4)."
        )


def require_colembed_for_inference() -> None:
    if not strict_real_enabled():
        return
    if os.getenv("INFERENCE_COLEMBED_RETRIEVAL", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    env_p = os.getenv("COLEMBED_MODEL_PATH", "").strip()
    p = Path(env_p) if env_p else repo_root() / "models" / "retriever" / "llama-nemotron-colembed-vl-3b-v2"
    if not p.is_dir():
        raise SystemExit(
            f"[inference] ColEmbed directory not found: {p}. "
            "Run: .venv/bin/python util/download_models.py --only colembed"
        )
    if not any(p.glob("*.safetensors")) and not any(p.glob("*.bin")):
        raise SystemExit(
            f"[inference] No ColEmbed weights under {p}. "
            "Run: .venv/bin/python util/download_models.py --only colembed"
        )
