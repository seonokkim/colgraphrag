"""
Resolve demo BE paths from ``paths.yaml``.

- ``run.run_id: latest`` → newest directory under ``<repo>/result/`` by mtime.
- All relative paths are anchored at the colgraphrag_webqa repo root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(__file__).resolve().parent


def _default_repo_root() -> Path:
    # demo/be/config/ -> parents[3] = colgraphrag_webqa
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for demo BE config") from e
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def _resolve_under(base: Path, p: str) -> Path:
    s = (p or "").strip()
    if not s:
        return base
    x = Path(s)
    return x if x.is_absolute() else (base / x).resolve()


@dataclass(frozen=True)
class DemoPaths:
    """Resolved absolute paths for inference / demo."""

    repo_root: Path
    run_id: str
    result_run_dir: Path
    webqa_slice_dir: Path
    webqa_questions_jsonl: Path
    webqa_imgs_dir: Path
    phase4_graphs_out: Path
    phase5_inference_dir: Path
    predictions_json: Path
    predictions_retrieval_json: Path
    gemma_model_dir: Path
    colembed_model_dir: Path
    fluency_bart_dir: Path
    train_val_json: Path | None


def load_demo_paths(
    config_path: Path | None = None,
    *,
    repo_root_override: Path | None = None,
) -> DemoPaths:
    """
    Load ``paths.yaml`` next to this module (or ``config_path``) and return resolved paths.

    Raises:
        FileNotFoundError: no ``result/`` runs when ``run_id`` is ``latest``.
    """
    cfg_path = config_path or (_CONFIG_DIR / "paths.yaml")
    cfg = _load_yaml(cfg_path)

    raw_repo = (cfg.get("repo_root") or "").strip()
    repo = Path(raw_repo).resolve() if raw_repo else _default_repo_root()
    if repo_root_override is not None:
        repo = repo_root_override.resolve()

    run_cfg = cfg.get("run") or {}
    explicit = (run_cfg.get("explicit_result_dir") or "").strip()
    rid = (run_cfg.get("run_id") or "latest").strip()

    result_root = repo / "result"
    if explicit:
        result_run = _resolve_under(repo, explicit)
    elif rid.lower() == "latest":
        if not result_root.is_dir():
            raise FileNotFoundError(f"No result directory: {result_root}")
        candidates = [p for p in result_root.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No run directories under {result_root}")
        result_run = max(candidates, key=lambda p: p.stat().st_mtime)
        rid = result_run.name
    else:
        result_run = (result_root / rid).resolve()

    layout = cfg.get("result_layout") or {}
    slice_name = (layout.get("webqa_slice") or "webqa_slice").strip()
    p4 = (layout.get("phase4_graphs_out") or "phase4_graphs_out").strip()
    p5 = (layout.get("phase5_inference") or "phase5_inference").strip()

    w = cfg.get("webqa") or {}
    slice_rel = (w.get("shard14_slice") or "data/webqa/webqa_shard14_toy/webqa_slice").strip()
    imgs_rel = (
        w.get("imgs_dir")
        or "data/webqa/WebQA_imgs_7z_chunks/imgs/all_png/shard_00014"
    )
    imgs_rel = str(imgs_rel).strip()
    tv_rel = (w.get("train_val_json") or "").strip()

    m = cfg.get("models") or {}
    gemma_rel = (m.get("gemma") or "models/mllm/gemma-4-E4B-it").strip()
    colembed_rel = (m.get("colembed") or "models/retriever/llama-nemotron-colembed-vl-3b-v2").strip()
    bart_rel = (m.get("fluency_bart") or "models/eval/bart-large-cnn").strip()

    webqa_slice_dir = _resolve_under(repo, slice_rel)
    webqa_questions = webqa_slice_dir / "webqa_questions.jsonl"
    webqa_imgs = _resolve_under(repo, imgs_rel)

    phase4 = result_run / p4
    phase5 = result_run / p5
    pred = phase5 / "predictions.json"
    pret = phase5 / "predictions_retrieval.json"

    train_val: Path | None = None
    if tv_rel:
        train_val = _resolve_under(repo, tv_rel)

    return DemoPaths(
        repo_root=repo,
        run_id=rid,
        result_run_dir=result_run,
        webqa_slice_dir=webqa_slice_dir,
        webqa_questions_jsonl=webqa_questions,
        webqa_imgs_dir=webqa_imgs,
        phase4_graphs_out=phase4,
        phase5_inference_dir=phase5,
        predictions_json=pred,
        predictions_retrieval_json=pret,
        gemma_model_dir=_resolve_under(repo, gemma_rel),
        colembed_model_dir=_resolve_under(repo, colembed_rel),
        fluency_bart_dir=_resolve_under(repo, bart_rel),
        train_val_json=train_val,
    )


def demo_paths_as_dict(dp: DemoPaths) -> dict[str, str]:
    """Flat string map for JSON responses or env injection."""

    out: dict[str, str] = {
        "repo_root": str(dp.repo_root),
        "run_id": dp.run_id,
        "result_run_dir": str(dp.result_run_dir),
        "webqa_slice_dir": str(dp.webqa_slice_dir),
        "webqa_questions_jsonl": str(dp.webqa_questions_jsonl),
        "webqa_imgs_dir": str(dp.webqa_imgs_dir),
        "phase4_graphs_out": str(dp.phase4_graphs_out),
        "phase5_inference_dir": str(dp.phase5_inference_dir),
        "predictions_json": str(dp.predictions_json),
        "predictions_retrieval_json": str(dp.predictions_retrieval_json),
        "gemma_model_dir": str(dp.gemma_model_dir),
        "colembed_model_dir": str(dp.colembed_model_dir),
        "fluency_bart_dir": str(dp.fluency_bart_dir),
    }
    if dp.train_val_json is not None:
        out["train_val_json"] = str(dp.train_val_json)
    return out


if __name__ == "__main__":
    dp = load_demo_paths()
    for k, v in demo_paths_as_dict(dp).items():
        print(f"{k}={v}")
