"""
Resolve demo BE paths from ``paths.yaml``.

- ``run.run_id: latest`` → newest directory under ``<repo>/result/`` by mtime.
- All relative paths are anchored at the colgraphrag_webqa repo root.
- MMQA results live under ``<repo>/result/multimodalqa/`` (separate subtree).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(__file__).resolve().parent


def _resolve_config_file(config_path: Path | None) -> Path:
    """
    Prefer ``paths.yaml`` (local overrides); if missing, use ``paths.example.yaml``
    so the demo BE works on a fresh clone without copying config.
    """
    if config_path is not None:
        p = Path(config_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Demo config not found: {p}")
        return p
    for name in ("paths.yaml", "paths.example.yaml"):
        p = _CONFIG_DIR / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No demo config under {_CONFIG_DIR}: copy paths.example.yaml to paths.yaml "
        "or restore paths.example.yaml in the repo."
    )


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
    cfg_path = _resolve_config_file(config_path)
    cfg = _load_yaml(cfg_path)

    raw_repo = (cfg.get("repo_root") or "").strip()
    repo = Path(raw_repo).resolve() if raw_repo else _default_repo_root()
    if repo_root_override is not None:
        repo = repo_root_override.resolve()

    run_cfg = cfg.get("run") or {}
    explicit = (run_cfg.get("explicit_result_dir") or "").strip()
    rid = (run_cfg.get("run_id") or "latest").strip()
    # result_root: optional sub-path for "latest" scan (e.g. "result/webqa")
    result_root_rel = (run_cfg.get("result_root") or "result/webqa").strip()
    result_root = _resolve_under(repo, result_root_rel)

    if explicit:
        result_run = _resolve_under(repo, explicit)
    elif rid.lower() == "latest":
        found = _find_latest_run(result_root)
        if found is None:
            raise FileNotFoundError(f"No run directories under {result_root}")
        result_run = found
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

    run_slice_dir = result_run / slice_name
    run_questions_jsonl = run_slice_dir / "webqa_questions.jsonl"
    if run_questions_jsonl.is_file():
        webqa_slice_dir = run_slice_dir.resolve()
        webqa_questions = run_questions_jsonl.resolve()
    else:
        webqa_slice_dir = _resolve_under(repo, slice_rel)
        webqa_questions = webqa_slice_dir / "webqa_questions.jsonl"

    webqa_imgs = _resolve_under(repo, imgs_rel)

    phase4 = _phase4_graph_dir(result_run, p4)
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


@dataclass(frozen=True)
class MmqaDemoPaths:
    """Resolved absolute paths for the MMQA dataset in the demo."""

    repo_root: Path
    run_id: str
    result_run_dir: Path
    mmqa_slice_dir: Path
    mmqa_questions_jsonl: Path
    mmqa_imgs_dir: Path
    phase4_graphs_out: Path
    phase5_inference_dir: Path


@dataclass(frozen=True)
class MultiDatasetPaths:
    """Container for all dataset paths loaded at startup."""

    webqa: DemoPaths
    mmqa: MmqaDemoPaths | None


def _find_latest_run(result_root: Path) -> Path | None:
    """Return newest subdirectory under result_root by mtime, or None."""
    if not result_root.is_dir():
        return None
    candidates = [p for p in result_root.iterdir() if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _find_latest_mmqa_completed_run(result_root: Path) -> Path | None:
    """
    For MMQA ``run_id: latest``, prefer a **finished pipeline** run: one that has
    ``phase5_inference/predictions.json``. Otherwise ``latest`` by mtime alone can pick a
    partial folder (patterns-only, etc.) with no graphs or scores.

    If no run has Phase 5 predictions, fall back to :func:`_find_latest_run`.
    """
    if not result_root.is_dir():
        return None
    completed: list[Path] = []
    for p in result_root.iterdir():
        if not p.is_dir():
            continue
        pred = p / "phase5_inference" / "predictions.json"
        if pred.is_file():
            completed.append(p)
    if completed:
        return max(completed, key=lambda x: x.stat().st_mtime)
    return _find_latest_run(result_root)


def _phase4_graph_dir(result_run: Path, configured_subdir: str) -> Path:
    """
    Resolve the Phase-4 graph folder under a pipeline run.

    ``construct.py`` defaults to ``phase4_graphs_real`` when
    ``CONSTRUCT_OUTPUT_GRAPH_DIR`` is unset; notebooks / ``test_pipeline`` use
    ``phase4_graphs_out``. Prefer whichever directory actually contains
    ``*_graph.graphml`` files.
    """
    primary = (result_run / configured_subdir.strip()).resolve()
    fallback = (result_run / "phase4_graphs_real").resolve()

    def _has_any_graphml(d: Path) -> bool:
        if not d.is_dir():
            return False
        return next(d.glob("*_graph.graphml"), None) is not None

    if _has_any_graphml(primary):
        return primary
    if _has_any_graphml(fallback):
        return fallback
    return primary


def load_mmqa_demo_paths(
    config_path: Path | None = None,
    *,
    repo_root_override: Path | None = None,
) -> MmqaDemoPaths | None:
    """
    Load MMQA paths from config and return ``MmqaDemoPaths``, or ``None`` when
    no MMQA run directory exists yet.

    When ``run_id`` is ``latest``, the newest run with
    ``phase5_inference/predictions.json`` is chosen; if none qualify, the newest
    directory by mtime is used (same as before).
    """
    cfg_path = _resolve_config_file(config_path)
    cfg = _load_yaml(cfg_path)

    raw_repo = (cfg.get("repo_root") or "").strip()
    repo = Path(raw_repo).resolve() if raw_repo else _default_repo_root()
    if repo_root_override is not None:
        repo = repo_root_override.resolve()

    mmqa_cfg = cfg.get("mmqa") or {}
    result_root_rel = (mmqa_cfg.get("result_root") or "result/multimodalqa").strip()
    result_root = _resolve_under(repo, result_root_rel)

    rid = (mmqa_cfg.get("run_id") or "latest").strip()
    if rid.lower() == "latest":
        result_run = _find_latest_mmqa_completed_run(result_root)
        if result_run is None:
            return None
        rid = result_run.name
    else:
        result_run = (result_root / rid).resolve()
        if not result_run.is_dir():
            return None

    slice_name = (mmqa_cfg.get("slice_dir") or "mmqa_slice").strip()
    imgs_rel = (mmqa_cfg.get("imgs_dir") or "data/multimodalqa/final_dataset_images").strip()

    mmqa_slice_dir = result_run / slice_name
    mmqa_questions = mmqa_slice_dir / "mmqa_questions.jsonl"
    mmqa_imgs = _resolve_under(repo, imgs_rel)

    layout = cfg.get("result_layout") or {}
    p4 = (layout.get("phase4_graphs_out") or "phase4_graphs_out").strip()
    phase4 = _phase4_graph_dir(result_run, p4)
    phase5 = result_run / "phase5_inference"

    return MmqaDemoPaths(
        repo_root=repo,
        run_id=rid,
        result_run_dir=result_run,
        mmqa_slice_dir=mmqa_slice_dir,
        mmqa_questions_jsonl=mmqa_questions,
        mmqa_imgs_dir=mmqa_imgs,
        phase4_graphs_out=phase4,
        phase5_inference_dir=phase5,
    )


def load_all_demo_paths(
    config_path: Path | None = None,
    *,
    repo_root_override: Path | None = None,
) -> MultiDatasetPaths:
    """Load WebQA and MMQA paths together."""
    webqa = load_demo_paths(config_path, repo_root_override=repo_root_override)
    mmqa = load_mmqa_demo_paths(config_path, repo_root_override=repo_root_override)
    return MultiDatasetPaths(webqa=webqa, mmqa=mmqa)


if __name__ == "__main__":
    dp = load_demo_paths()
    for k, v in demo_paths_as_dict(dp).items():
        print(f"{k}={v}")
    mmqa = load_mmqa_demo_paths()
    if mmqa:
        print(f"mmqa run_id={mmqa.run_id} questions={mmqa.mmqa_questions_jsonl}")
