"""
Load ``config/data.yaml`` and ``config/model.yaml`` into environment defaults.

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
                # Only set if resolved path exists (local-first: skip if not downloaded yet)
                if r and Path(r).is_dir():
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

    _APPLIED = True
