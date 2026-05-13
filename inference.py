"""Phase 5 — GraphRAG 최종 추론 CLI.

입력:
  - Phase 1 슬라이스 ``*_questions.jsonl``
  - Phase 4 산출 ``phase4_graphs_real/<qid>_graph.graphml`` (또는 env 로 지정한 디렉터리)

처리 개요:
  1) (선택) 메타의 ``image_doc_ids`` 에 대해 ColEmbed 질문–이미지 MaxSim 재순위.
     순위는 ``*_retrieval.json`` 등으로만 기록되고 **답변 프롬프트에는 넣지 않음**.
  2) 그래프를 ``graph_to_str`` 로 텍스트 요약(텍스트/이미지/테이블 블록 + 관계).
  3) 데이터셋에 따라 ``LLM_ANSWER_PROMPT``(WebQA·일반) 또는 ``MMQA_ANSWER_PROMPT``(MMQA)로
     HF Gemma 4 E4B IT 에 최종 답 생성.

``INFERENCE_DRY_RUN=1`` 이면 ColEmbed/Gemma 호출 없이 그래프 휴리스틱만 사용.
"""
import io
import json
import logging
import os
import re
from collections.abc import Mapping
from glob import glob
from pathlib import Path

import networkx as nx
import torch

from prompt import *

from util.request import load_pretrained_model_with_fallback
from util.llm_defaults import effective_gemma4_e4b_it_model_path, ensure_default_gemma4_e4b_it_path
from util.pipeline_session_log import new_session_log_path, repo_logs_dir
from util.repo_config import (
    forbid_dry_run,
    require_colembed_for_inference,
    require_cuda,
    require_gemma_local,
)
from util.result_layout import resolve_pipeline_run_id
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

# INFERENCE_DRY_RUN=1: ColEmbed·Gemma 없이 그래프 휴리스틱 답만(스모크/CI)
DRY_RUN = os.getenv("INFERENCE_DRY_RUN", "0") == "1"
_HF_GEMMA_MODULE = None

ensure_default_gemma4_e4b_it_path()


def _get_hf_gemma_module():
    global _HF_GEMMA_MODULE
    if _HF_GEMMA_MODULE is not None:
        return _HF_GEMMA_MODULE
    try:
        from mllm import hf_gemma_4_e4b_it as gemma_module
        if gemma_module.configured():
            _HF_GEMMA_MODULE = gemma_module
            return gemma_module
    except ImportError:
        pass
    return None


def _hf_gemma_generate_text(prompt: str) -> tuple[str, dict]:
    """프로세스 내 HF Gemma 4 E4B IT 로 답 생성; sidecar용 메타 dict 동반."""
    gemma = _get_hf_gemma_module()
    if gemma is None:
        raise RuntimeError("HF Gemma module not available or not configured")
    text = gemma.generate_text(prompt, max_new_tokens=512)
    llm_meta = {
        "tier": "hf_gemma_4_e4b_it",
        "model": effective_gemma4_e4b_it_model_path(),
        "base_url": "(in-process)",
    }
    return text, llm_meta


# --- 경로·질문 파일·출력: 환경변수로 덮어쓰기(노트북/CI에서 RUN_ID·슬라이스 동기화) ---
_BASE_DIR = Path(__file__).resolve().parent
_DATASET = os.getenv("MMGRAPHRAG_DATASET", "webqa").strip().lower()
_RUN_ID = resolve_pipeline_run_id(_BASE_DIR, _DATASET)
GRAPH_DIR = os.getenv(
    "INFERENCE_GRAPH_DIR",
    str(_BASE_DIR / "result" / _RUN_ID / "phase4_graphs_real"),
)
_SLICE = _BASE_DIR / "result" / _RUN_ID / f"{_DATASET}_slice"
QUESTION_FILE = os.getenv(
    "INFERENCE_QUESTION_FILE",
    str(_SLICE / f"{_DATASET}_questions.jsonl"),
)
OUTPUT_JSON = os.getenv(
    "INFERENCE_OUTPUT_JSON",
    str(_BASE_DIR / "result" / _RUN_ID / "phase5_inference" / "predictions.json"),
)
INFERENCE_RETRIEVAL_JSON = os.getenv("INFERENCE_RETRIEVAL_JSON", "")
MAX_QUESTIONS = int(os.getenv("INFERENCE_MAX_QUESTIONS", "20"))

_INFER_COLEMBED_FALLBACK_LOG: Path | None = None


def _colembed_fallback_log_file() -> Path:
    """When no ``MMGRAPHRAG_SESSION_LOG_PATH`` (tee), use timestamped file under ``logs/``."""
    global _INFER_COLEMBED_FALLBACK_LOG
    if _INFER_COLEMBED_FALLBACK_LOG is None:
        _INFER_COLEMBED_FALLBACK_LOG = new_session_log_path(_BASE_DIR, "inference_colembed")
    return _INFER_COLEMBED_FALLBACK_LOG

# ColEmbed MaxSim 전역 캐시(이 프로세스에서 한 번만 로드)
model = None
processor = None
logger = logging.getLogger("colembed_mm_graph_rag")


def _default_retrieval_json_path(output_json_path: str) -> str:
    if output_json_path.endswith(".json"):
        return output_json_path[:-5] + "_retrieval.json"
    return output_json_path + "_retrieval.json"


def _default_models_sidecar_path(output_json_path: str) -> str:
    """예측 JSON 옆에 ``*_models.json`` — qid별 사용 LLM 티어/모델 경로 기록. 예측 파일 포맷은 평가기 호환 유지."""
    if output_json_path.endswith(".json"):
        return output_json_path[:-5] + "_models.json"
    return output_json_path + "_models.json"


def _get_qid(question: dict) -> str:
    """Get question ID (supports both 'qid' and 'Guid' formats for WebQA)."""
    return question.get("qid") or question.get("Guid") or ""


def _get_question_text(question: dict) -> str:
    """Get question text (supports both 'question' and 'Q' field names)."""
    return question.get("question") or question.get("Q") or ""


def _summarise_qid_to_llm(qid_to_llm: dict[str, dict]) -> dict:
    """sidecar 요약: tier·모델 문자열 별 건수."""
    by_tier: dict[str, int] = {}
    by_model: dict[str, int] = {}
    for meta in qid_to_llm.values():
        if not isinstance(meta, dict):
            continue
        tier = str(meta.get("tier", "unknown"))
        model = str(meta.get("model", "(unknown)"))
        by_tier[tier] = by_tier.get(tier, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
    return {"by_tier": by_tier, "by_model": by_model}


def _write_models_sidecar(path: str, qid_to_llm: dict[str, dict]) -> None:
    """qid→LLM 메타 맵과 run 요약을 한 파일에 기록."""
    payload = {
        "meta": {
            "run_id": _RUN_ID,
            "backend": "hf_gemma_4_e4b_it",
            "model_path": os.getenv("GEMMA4_E4B_IT_MODEL_PATH", "(default)"),
        },
        "qid_to_llm": qid_to_llm,
        "summary": _summarise_qid_to_llm(qid_to_llm),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _strip_question_id_prefix(qid: str, token: str) -> str:
    """Normalize MMQA attribution tokens like `{qid}_{doc_id}` to gold `doc_id`."""
    t = token.strip()
    if not qid or not t:
        return t
    prefix = f"{qid}_"
    return t[len(prefix) :] if t.startswith(prefix) else t


def _tokens_for_retrieval_attrs(node_or_edge_attrs: Mapping, qid: str) -> list[str]:
    """노드/엣지 속성에서 검색용 문서 id 목록: ``doc_id`` 우선, 없으면 ``source_id`` 쉼표 분해 후 qid 접두 제거."""
    if not isinstance(node_or_edge_attrs, Mapping):
        return []
    doc_id_raw = node_or_edge_attrs.get("doc_id")
    if isinstance(doc_id_raw, str) and doc_id_raw.strip():
        parts = [p.strip() for p in doc_id_raw.split(",") if p.strip()]
        out = [p for p in parts if p]
        if out:
            return out
    raw_sid = node_or_edge_attrs.get("source_id")
    if raw_sid is None:
        return []
    chunks = [p.strip() for p in str(raw_sid).split(",") if p.strip()]
    return [_strip_question_id_prefix(qid, ch) for ch in chunks]


# --- 그래프만 쓰는 후보 순위: 노드·엣지의 doc_id/source_id에 그래프 차수(연결 강도)를 가중치로 부여 ---
def canonical_graph_doc_candidates(graph: nx.Graph, qid: str, capacity: int = 64) -> list[dict]:
    """그래프 위상만 사용한 후보 순위: 노드·엣지에 붙은 doc/table id 에 차수 기반 가중치 부여."""
    scores: dict[str, float] = {}

    def _bump(candidate_ids: list[str], strength: float) -> None:
        for cid in candidate_ids:
            if not cid:
                continue
            prev = scores.get(cid)
            if prev is None or strength > prev:
                scores[cid] = strength

    for nodename, attrs in graph.nodes(data=True):
        deg = float(graph.degree[nodename])
        _bump(_tokens_for_retrieval_attrs(attrs, qid), deg)

    for u, v, attrs in graph.edges(data=True):
        deg_max = max(float(graph.degree[u]), float(graph.degree[v]))
        _bump(_tokens_for_retrieval_attrs(attrs, qid), deg_max)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ranked: list[dict] = []
    for idx, (doc_id_val, scr) in enumerate(ordered[:capacity]):
        ranked.append(
            {
                "id": doc_id_val,
                "score": scr,
                "rank": idx + 1,
                "source": "graph_topological",
            }
        )
    return ranked


def extract_ranked_source_ids_from_graph(graph: nx.Graph, top_k: int = 10):
    ranked = canonical_graph_doc_candidates(graph, qid="", capacity=max(top_k, 32))
    return ranked[:top_k]


def _merge_colembed_with_graph_rank(
    colembed_entries: list[dict],
    graph: nx.Graph,
    top_k: int,
    qid: str,
) -> list[dict]:
    """ColEmbed 이미지 순위를 앞에 두고, 남는 슬롯은 ``canonical_graph_doc_candidates`` 로 채움."""
    gc = canonical_graph_doc_candidates(graph, qid, capacity=96)
    if not colembed_entries:
        return gc[:top_k]

    merged: list[dict] = []
    seen: set[str] = set()
    lowest = float(colembed_entries[-1].get("score", 0.0))
    step = 0.001
    rank = 0

    for ce in colembed_entries:
        if len(merged) >= top_k:
            break
        iid = str(ce.get("id", "")).strip()
        if not iid or iid in seen:
            continue
        seen.add(iid)
        rank += 1
        row = dict(ce)
        row["id"] = iid
        row["rank"] = rank
        merged.append(row)

    for ge in gc:
        if len(merged) >= top_k:
            break
        iid = str(ge.get("id", "")).strip()
        if not iid or iid in seen:
            continue
        seen.add(iid)
        rank += 1
        merged.append(
            {
                "id": iid,
                "score": float(ge.get("score", lowest - step * rank)),
                "rank": rank,
                "source": "graph_enriched",
            }
        )
    return merged


# --- 슬라이스 이미지 행의 path/url/id 로 디스크 실경로 결정(ColEmbed MaxSim 픽셀 입력) ---
def _default_webqa_imgs_root() -> Path:
    env_root = os.getenv("WEBQA_DATA_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
    else:
        local = _BASE_DIR / "data" / "webqa"
        root = local
    return Path(
        os.getenv("WEBQA_IMGS_DIR", str(root / "WebQA_data_first_release" / "imgs"))
    )


def _default_mmqa_imgs_root() -> Path:
    env = os.getenv("MMQA_IMAGES_DIR", "").strip()
    if env:
        return Path(env)
    return _BASE_DIR / "data" / "multimodalqa" / "final_dataset_images"


def _load_slice_images_index(slice_dir: Path) -> dict[str, dict]:
    """``<dataset>_images.jsonl`` (또는 webqa 호환 파일명)에서 image id → 메타 row."""
    idx: dict[str, dict] = {}
    for fname in (f"{_DATASET}_images.jsonl", "webqa_images.jsonl"):
        p = slice_dir / fname
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rid = str(row.get("id", "") or row.get("image_id", "")).strip()
                    if rid:
                        idx[rid] = row
            return idx
    return idx


# 외부 호출 호환 별칭
_load_webqa_slice_images_index = _load_slice_images_index


def _resolve_webqa_image_fs_path(row: dict, imgs_root: Path) -> str | None:
    raw = (row.get("path") or row.get("url") or "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return str(p.resolve())
        rel = imgs_root / raw.lstrip("/\\")
        if rel.is_file():
            return str(rel.resolve())
        repo_rel = _BASE_DIR / raw.lstrip("/\\")
        if repo_rel.is_file():
            return str(repo_rel.resolve())
    iid = str(row.get("id") or row.get("image_id") or "").strip()
    if not iid:
        return None
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".JPG", ".PNG"):
        rel = imgs_root / f"{iid}{ext}"
        if rel.is_file():
            return str(rel.resolve())
    matches = glob(str(imgs_root / f"*{iid}*"))
    for m in matches:
        mp = Path(m)
        if mp.is_file():
            return str(mp.resolve())
    return None


def _ensure_colembed_model_loaded() -> None:
    """Load llama-nemotron-colembed-vl-3b-v2 (MaxSim) once for retrieval / shortlist."""
    global model, processor
    if model is not None and processor is not None:
        return
    model_ref = resolve_vision_model_ref()
    trust_remote_code = use_trust_remote_code()
    model = load_pretrained_model_with_fallback(
        AutoModel, model_ref, trust_remote_code=trust_remote_code
    )
    ensure_colembed_retrieval_api(model)
    processor = _load_colembed_processor(model_ref, trust_remote_code=trust_remote_code)
    logger.info(
        "ColEmbed model loaded | ref=%s | cuda=%s",
        model_ref,
        torch.cuda.is_available(),
    )


def colembed_maxsim_ranked_sources(
    question_text: str,
    image_doc_ids: list,
    images_index: dict[str, dict],
    imgs_root: Path,
    top_k: int,
) -> tuple[list[dict], bool]:
    """
    ColEmbed VL 체크포인트로 질문–이미지 MaxSim 점수 산출(GPU 우선).
    (ranked 행 목록, 성공 여부) 반환; 픽셀을 LLM 답 프롬프트에 넣지는 않음.
    """
    paths: list[str] = []
    ids: list[str] = []
    for iid in image_doc_ids:
        row = images_index.get(str(iid))
        if not row:
            continue
        fs = _resolve_webqa_image_fs_path(row, imgs_root)
        if fs:
            paths.append(fs)
            ids.append(str(iid))
    if not paths:
        return [], False
    _ensure_colembed_model_loaded()
    imgs = [Image.open(p).convert("RGB") for p in paths]
    texts = [question_text or ""]
    with torch.no_grad():
        query_embeddings = model.forward_queries(texts, batch_size=1)
        image_embeddings = model.forward_images(imgs, batch_size=max(1, len(imgs)))
        scores = model.get_scores(query_embeddings, image_embeddings)
    s1 = scores[0]
    if s1.dim() > 1:
        s1 = s1.flatten()
    n = min(int(top_k), int(s1.shape[0]))
    if n <= 0:
        return [], False
    top = torch.topk(s1, k=n, dim=-1)
    ref = resolve_vision_model_ref()
    out: list[dict] = []
    for rank, (ix, val) in enumerate(
        zip(top.indices.tolist(), top.values.detach().float().cpu().tolist())
    ):
        i = int(ix)
        out.append(
            {
                "id": ids[i],
                "score": float(val),
                "rank": rank + 1,
                "source": "colembed_maxsim",
                "model_ref": ref,
            }
        )
    return out, True


def _extract_dry_run_answer_from_graph(g: nx.Graph, question_text: str) -> str:
    """
    드라이런 전용: 플레이스홀더가 아닌 TEXT→IMAGE→기타 엔티티 순으로 첫 후보 문자열 선택.
    """
    PLACEHOLDER_TERMS = {"placeholder", "dummy", "webqa placeholder", "webqa placeholder dummy", "unknown"}
    question_lower = question_text.lower().strip()
    
    # Collect candidates by type
    text_candidates = []
    image_candidates = []
    other_candidates = []
    
    for node_id, data in g.nodes(data=True):
        entity_name = str(data.get("entity_name", "")).strip()
        node_type = str(data.get("type", "")).upper()
        description = str(data.get("description", "")).strip()
        
        # Skip placeholders and empty values
        if not entity_name or entity_name.lower() in PLACEHOLDER_TERMS:
            continue
        if "placeholder" in entity_name.lower() or "dummy" in entity_name.lower():
            continue
        
        # Skip if it's the question itself
        if entity_name.lower() == question_lower or entity_name.lower() in question_lower:
            continue
        if node_type == "QUESTION":
            continue
        
        # Categorize by type
        if node_type == "TEXT":
            text_candidates.append(entity_name)
        elif node_type == "IMAGE":
            # For images, try to extract meaningful info from description
            if description and "placeholder" not in description.lower():
                image_candidates.append(entity_name)
        elif node_type != "TABLE":
            other_candidates.append(entity_name)
    
    # Return best candidate
    if text_candidates:
        return text_candidates[0]
    if image_candidates:
        return image_candidates[0]
    if other_candidates:
        return other_candidates[0]
    
    return "unknown"


def setup_logger() -> logging.Logger:
    repo_logs_dir(_BASE_DIR).mkdir(parents=True, exist_ok=True)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    session_log = os.getenv("MMGRAPHRAG_SESSION_LOG_PATH", "").strip()
    file_handler: logging.FileHandler | None = None
    if session_log:
        log_dest = str(Path(session_log).expanduser().resolve())
    else:
        fh_path = _colembed_fallback_log_file()
        file_handler = logging.FileHandler(fh_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        log_dest = str(fh_path.resolve())

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info("Logger initialized. log_file=%s", log_dest)

    request_logger = logging.getLogger("webqa.request")
    if not request_logger.handlers:
        request_logger.setLevel(logging.INFO)
        if file_handler is not None:
            request_logger.addHandler(file_handler)
        request_logger.addHandler(stream_handler)
        request_logger.propagate = False
    return logger

def _colembed_local_first_default() -> str:
    """Prefer <repo>/models/retriever/...; else same path (install via util/download_models.py)."""
    local = _BASE_DIR / "models" / "retriever" / "llama-nemotron-colembed-vl-3b-v2"
    if local.is_dir():
        return str(local.resolve())
    return str(local)


def resolve_vision_model_ref() -> str:
    return (
        os.getenv("COLEMBED_MODEL_PATH", "").strip()
        or os.getenv("COLEMBED_MODEL_ID", "").strip()
        or _colembed_local_first_default()
    )


def use_trust_remote_code() -> bool:
    return os.getenv("COLEMBED_TRUST_REMOTE_CODE", "1").strip().lower() not in {"0", "false", "no"}


def use_score_debug_log() -> bool:
    return os.getenv("COLEMBED_DEBUG_SCORES", "0").strip().lower() in {"1", "true", "yes"}


def resolve_colembed_max_input_tiles() -> int:
    """추론 시 ``AutoProcessor`` 의 ``max_input_tiles``. 기본 8; 세부는 논문/내부 노트 참고. env 로 재정의."""
    raw = os.getenv("COLEMBED_MAX_INPUT_TILES", "").strip()
    if not raw:
        return 8
    try:
        parsed = int(raw)
    except ValueError:
        return 8
    return parsed if parsed > 0 else 8


def resolve_colembed_topk_images() -> int:
    """VLM 전 ColEmbed 단계에서 가져갈 이미지 shortlist 크기. 기본 10; env ``COLEMBED_TOPK_IMAGES``."""
    raw = os.getenv("COLEMBED_TOPK_IMAGES", "").strip()
    if not raw:
        return 10
    try:
        parsed = int(raw)
    except ValueError:
        return 10
    return parsed if parsed > 0 else 10


def _load_colembed_processor(model_ref: str, trust_remote_code: bool):
    """ColEmbed ``AutoProcessor`` 단일 생성 지점. ``max_input_tiles`` 를 kwargs 로 넣고, 미지원이면 사후 setattr"""
    tiles = resolve_colembed_max_input_tiles()
    try:
        proc = AutoProcessor.from_pretrained(
            model_ref,
            trust_remote_code=trust_remote_code,
            max_input_tiles=tiles,
        )
        _log_tiles_once(tiles, success=True)
    except TypeError:
        proc = AutoProcessor.from_pretrained(
            model_ref, trust_remote_code=trust_remote_code
        )
        _apply_tiles_post_load(proc, tiles)
        _log_tiles_once(tiles, success=False)
    return proc


def _apply_tiles_post_load(processor, tiles: int) -> None:
    """from_pretrained 가 타일 인자를 무시한 프로세서에 대한 best-effort 보정."""
    if hasattr(processor, "max_input_tiles"):
        try:
            setattr(processor, "max_input_tiles", tiles)
        except (AttributeError, TypeError):
            pass


_TILES_LOG_DONE = False


def _log_tiles_once(tiles: int, *, success: bool) -> None:
    """프로세스당 1회: 유효 ``max_input_tiles`` 및 top_k 로그."""
    global _TILES_LOG_DONE
    if _TILES_LOG_DONE:
        return
    _TILES_LOG_DONE = True
    try:
        msg = (
            f"[Colembed] max_input_tiles={tiles} (via_kwarg={success}); "
            f"top_k_images={resolve_colembed_topk_images()}"
        )
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
    except Exception:
        pass


def ensure_colembed_retrieval_api(loaded_model) -> None:
    """MaxSim 경로에 필요한 메서드 존재 여부 선검증."""
    required = ("forward_queries", "forward_images", "get_scores")
    missing = [name for name in required if not hasattr(loaded_model, name)]
    if missing:
        raise RuntimeError(
            f"Colembed retrieval API missing required methods: {missing}. "
            "Model must support forward_queries/forward_images/get_scores."
        )


def probe_model_output_contract(model_ref: str = "", device_map: str = "auto") -> dict:
    """
    개발/진단용: 모델 출력에 text_embeds·image_embeds·logits_per_text 등이 있는지 프로브.
    """
    ref = model_ref or resolve_vision_model_ref()
    trust_remote_code = use_trust_remote_code()
    setup_logger()
    logger.info(
        "Phase A probe started. model_ref=%s device_map=%s trust_remote_code=%s",
        ref,
        device_map,
        trust_remote_code,
    )
    if device_map == "auto":
        local_model = load_pretrained_model_with_fallback(
            AutoModel, ref, trust_remote_code=trust_remote_code
        )
    else:
        local_model = AutoModel.from_pretrained(
            ref, device_map=device_map, trust_remote_code=trust_remote_code
        )
    local_processor = _load_colembed_processor(ref, trust_remote_code=trust_remote_code)
    dummy_image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    inputs = local_processor(
        text=["contract probe"],
        images=[dummy_image],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    # 프로브 입력 텐서를 모델 파라미터와 동일 디바이스로 이동(cuda/cpu 불일치 방지)
    model_device = next(local_model.parameters()).device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model_device)
    with torch.no_grad():
        outputs = local_model(**inputs)
    result = {
        "model_ref": ref,
        "has_forward_queries": hasattr(local_model, "forward_queries"),
        "has_forward_images": hasattr(local_model, "forward_images"),
        "has_get_scores": hasattr(local_model, "get_scores"),
        "has_text_embeds": hasattr(outputs, "text_embeds"),
        "has_image_embeds": hasattr(outputs, "image_embeds"),
        "has_logits_per_text": hasattr(outputs, "logits_per_text"),
    }
    logger.info("Phase A probe result: %s", json.dumps(result, ensure_ascii=False))
    return result

def str_to_dict_list(str_dict_list):
    """``json.loads`` 로 리스트/dict 복원; 실패 시 None."""
    try:
        # JSON 문자열을 파이썬 list/dict 로 복원(실패 시 None)
        result_list = json.loads(str_dict_list)
        return result_list
    except json.JSONDecodeError as e:
        return None

def graph_to_graphml_str(graph):
    """GraphML 바이트 → UTF-8 문자열(소규모 헬퍼)."""
    with io.BytesIO() as byte_output:
        nx.write_graphml(graph, byte_output)
        byte_output.seek(0)
        graphml_str = byte_output.read().decode('utf-8')
    return graphml_str


# --- Gemma 프롬프트에 들어가는 그래프 요약(픽셀 없음; 노드/엣지의 entity·description 텍스트만) ---
def graph_to_str(graph):
    """Gemma 프롬프트용: 픽셀 대신 노드·엣지의 이름/타입/설명을 블록 텍스트로 요약."""
    output = []
    text_nodes = []
    image_nodes = []
    table_nodes = []

    for node in graph.nodes(data=True):
        node_id = node[0]
        node_data = node[1]
        node_info = {
            'id': node_id,
            'name': node_data.get('entity_name', ''),
            'type': node_data.get('type', ''),
            'description': node_data.get('description', '')
        }
        if node_id.endswith('IMAGE'):
            image_nodes.append(node_info)
        elif node_id.endswith('TABLE'):
            table_nodes.append(node_info)
        else:
            text_nodes.append(node_info)
    output.append("======= BEGIN: TEXT NODES BLOCK =======")
    for node in text_nodes:
        if node['name'] and node['type']:
            output.append(f"Name: {node['name']}")
            output.append(f"Type: {node['type']}")
            output.append(f"Description: {node['description']}")
            output.append("---")
    output.append("======= END: TEXT NODES BLOCK =======")
    output.append("")

    output.append("======= BEGIN: IMAGE NODES BLOCK =======")
    for node in image_nodes:
        if node['name']:
            output.append(f"Name: {node['name']}")
            output.append(f"Type: image")
            output.append(f"Description: {node['description']}")
            output.append("---")
    output.append("======= END: IMAGE NODES BLOCK =======")
    output.append("")

    output.append("======= BEGIN: TABLE NODES BLOCK =======")
    for node in table_nodes:
        if node['name']:
            output.append(f"Name: {node['name']}")
            output.append(f"Type: table")
            output.append(f"Description: {node['description']}")
            output.append("---")
    output.append("======= END: TABLE NODES BLOCK =======")
    output.append("")

    output.append("======= BEGIN: RELATIONSHIPS BLOCK =======")
    for edge in graph.edges(data=True):
        source_node = graph.nodes[edge[0]]
        target_node = graph.nodes[edge[1]]
        edge_data = edge[2]

        if source_node.get('entity_name') and target_node.get('entity_name'):
            output.append(f"Node 1 Name: {source_node['entity_name']}")
            if source_node.get('type') and source_node.get('type') != 'unspecified':
                output.append(f"Node 1 Type: {source_node['type']}")
            output.append(f"Node 2 Name: {target_node['entity_name']}")
            if target_node.get('type') and target_node.get('type') != 'unspecified':
                output.append(f"Node 2 Type: {target_node['type']}")
            if edge_data.get('description') and edge_data.get('description') != 'unspecified':
                output.append(f"Relationship between Node 1 and Node 2: {edge_data['description']}")
            output.append("----------")
    output.append("======= END: RELATIONSHIPS BLOCK =======")

    return '\n'.join(output)


# 다른 모듈용: 경로 리스트에 대한 ColEmbed 상위 n 이미지 인덱스(추론 CLI 메인 루프는 graph_to_str 경로 사용)
def text_to_image_feature(image_paths, texts, n=None):
    """ColEmbed MaxSim: ``texts[0]`` 에 대한 상위 ``n`` 이미지 인덱스. ``n`` 기본은 env 기반."""
    global model, processor
    if n is None:
        n = resolve_colembed_topk_images()
    if use_score_debug_log():
        setup_logger()
    _ensure_colembed_model_loaded()
    images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
    if not texts:
        return []
    with torch.no_grad():
        # Phase B enforcement: both query and image are encoded by the same Colembed model.
        query_embeddings = model.forward_queries(texts, batch_size=max(1, len(texts)))
        image_embeddings = model.forward_images(images, batch_size=max(1, len(images)))
        scores = model.get_scores(query_embeddings, image_embeddings)
        # 레거시 반환: 첫 번째 텍스트 쿼리에 대한 이미지 인덱스 목록
    scores_1d = scores[0]
    k = min(int(n), int(scores_1d.shape[0]))
    if k <= 0:
        return []
    topk = torch.topk(scores_1d, k=k, dim=-1)
    indices = topk.indices.tolist()
    if use_score_debug_log():
        values = topk.values.detach().float().cpu()
        logger.info(
            "Colembed score debug | texts=%d images=%d score_shape=%s topk_indices=%s topk_min=%.6f topk_max=%.6f",
            len(texts),
            len(images),
            tuple(scores.shape),
            indices,
            float(values.min().item()),
            float(values.max().item()),
        )
    return indices

def extract_answer_list(text, answer_pattern=r'<\|Answer\|>([\s\S]*?)<\|\\Answer\|>'):
    """``<|Answer|>...<|\Answer|>`` 안의 파이썬 리스트 문자열을 파싱해 첫 목록 반환."""
    output = text
    answers = []
    match = re.findall(answer_pattern, output)
    for item in match:
        item = item.strip()
        try:
            item_list = str_to_dict_list(item)
            if item_list is not None:
                answers.append(item_list)
        except (ValueError, SyntaxError):
            pass

    return answers[0] if answers else []


def load_jsonl_data(path):
    """질문/기타 JSONL 로드."""
    with open(path, "r", encoding='UTF-8') as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    from pathlib import Path

    from util.pipeline_session_log import run_with_session_stdio_tee

    def _inference_cli_main() -> None:
        # dry-run: Gemma/ColEmbed 실호출 없이 그래프 휴리스틱 답 + retrieval 슬롯만 채움
        # 실제 실행: ColEmbed(옵션)로 retrieval_predictions 기록 후, 답은 graph_to_str + Gemma
        forbid_dry_run("inference", dry_run=DRY_RUN)
        if not DRY_RUN:
            require_cuda("inference")
            require_gemma_local("inference")
            require_colembed_for_inference()
        setup_logger()
        questions = load_jsonl_data(QUESTION_FILE)
        if MAX_QUESTIONS > 0:
            questions = questions[:MAX_QUESTIONS]
        predictions = {}
        retrieval_predictions = {}
        retrieval_output_json = (
            INFERENCE_RETRIEVAL_JSON.strip()
            if INFERENCE_RETRIEVAL_JSON.strip()
            else _default_retrieval_json_path(OUTPUT_JSON)
        )
        # qid -> LLM attribution (same shape as util.request.text_request_with_meta
        # returns). Written to <output>_models.json as a sidecar; the main
        # predictions file stays {qid: answer_str} so the QA evaluator is
        # untouched.
        qid_to_llm: dict[str, dict] = {}
        models_sidecar_json = _default_models_sidecar_path(OUTPUT_JSON)

        # INFERENCE_SLICE_DIR is preferred; INFERENCE_WEBQA_SLICE_DIR kept for backwards compat
        _slice_for_retrieval = Path(os.getenv("INFERENCE_SLICE_DIR", os.getenv("INFERENCE_WEBQA_SLICE_DIR", str(_SLICE))))
        _use_colembed_retrieval = os.getenv("INFERENCE_COLEMBED_RETRIEVAL", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        _images_index = (
            _load_slice_images_index(_slice_for_retrieval) if _use_colembed_retrieval else {}
        )
        _imgs_root = _default_mmqa_imgs_root() if _DATASET == "mmqa" else _default_webqa_imgs_root()
        _topk_colembed = int(os.getenv("INFERENCE_COLEMBED_TOP_K", "10"))
        _mmqa_colembed_img_cap = int(os.getenv("INFERENCE_MMQA_COLEMBED_IMG_CAP", "5"))

        # 질문 메타의 이미지 id 후보만 ColEmbed로 재순위; 실패 시 그래프 위상 순위로 대체
        def _fill_retrieval(question_dict: dict, graph_obj: nx.Graph) -> list:
            qid_res = _get_qid(question_dict)
            if _use_colembed_retrieval:
                md = question_dict.get("metadata") or {}
                img_ids = md.get("image_doc_ids") or []
                ranked, ok = colembed_maxsim_ranked_sources(
                    _get_question_text(question_dict),
                    img_ids,
                    _images_index,
                    _imgs_root,
                    max(_topk_colembed, _mmqa_colembed_img_cap + 5),
                )
                co_cap = (
                    min(_topk_colembed, max(1, _mmqa_colembed_img_cap))
                    if (_DATASET == "mmqa" and ok and ranked)
                    else _topk_colembed
                )
                if ok and ranked:
                    trimmed = ranked[:co_cap]
                    return _merge_colembed_with_graph_rank(
                        trimmed, graph_obj, _topk_colembed, qid_res
                    )
            return canonical_graph_doc_candidates(
                graph_obj, qid_res, capacity=max(_topk_colembed * 4, 32)
            )[:_topk_colembed]

        if DRY_RUN:
            for question in tqdm(questions, desc="Processing Questions (dry-run)"):
                qid = _get_qid(question)
                graph_path = os.path.join(GRAPH_DIR, f"{qid}_graph.graphml")
                if not os.path.exists(graph_path):
                    predictions[qid] = "unknown"
                    retrieval_predictions[qid] = []
                    qid_to_llm[qid] = {"tier": "dry-run", "model": "(none)", "base_url": "(none)", "reason": "missing-graph"}
                    continue
                try:
                    g = nx.read_graphml(graph_path)
                    retrieval_predictions[qid] = _fill_retrieval(question, g)
                    answer = _extract_dry_run_answer_from_graph(g, _get_question_text(question))
                    predictions[qid] = str(answer)
                    qid_to_llm[qid] = {"tier": "dry-run", "model": "(none)", "base_url": "(none)", "reason": "graph-heuristic"}
                except Exception:
                    predictions[qid] = "unknown"
                    retrieval_predictions[qid] = []
                    qid_to_llm[qid] = {"tier": "dry-run", "model": "(none)", "base_url": "(none)", "reason": "graph-read-error"}
            Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            with open(retrieval_output_json, "w", encoding="utf-8") as f:
                json.dump(retrieval_predictions, f, ensure_ascii=False, indent=2)
            _write_models_sidecar(models_sidecar_json, qid_to_llm)
        else:
            unknown_qids: list[str] = []
            primary_fail_count = 0
            logger.info(
                "real inference starting | n=%d | backend=hf_gemma_4_e4b_it | model=%s",
                len(questions),
                os.getenv("GEMMA4_E4B_IT_MODEL_PATH", "(default)"),
            )
            for question in tqdm(questions, desc="Processing Questions (real)"):
                qid = _get_qid(question)
                graph_path = os.path.join(GRAPH_DIR, f"{qid}_graph.graphml")
                if not os.path.exists(graph_path):
                    logger.warning("qid=%s missing graph: %s", qid, graph_path)
                    predictions[qid] = "unknown"
                    retrieval_predictions[qid] = []
                    qid_to_llm[qid] = {"tier": "none", "model": "(none)", "base_url": "(none)", "reason": "missing-graph"}
                    unknown_qids.append(qid)
                    continue
                try:
                    g = nx.read_graphml(graph_path)
                    retrieval_predictions[qid] = _fill_retrieval(question, g)
                    # retrieval 순위는 prompt에 넣지 않음; 컨텍스트는 그래프 텍스트 전부
                    _ans_tpl = MMQA_ANSWER_PROMPT if _DATASET == "mmqa" else LLM_ANSWER_PROMPT
                    prompt = _ans_tpl.replace("{question}", _get_question_text(question)).replace(
                        "{GraphML}", graph_to_str(g)
                    )
                    answer, llm_meta = _hf_gemma_generate_text(prompt)
                    answer = (answer or "").strip()
                    if not answer or answer.lower() == "unknown":
                        logger.warning("qid=%s LLM returned empty/unknown", qid)
                        predictions[qid] = "unknown"
                        qid_to_llm[qid] = {
                            "tier": "none", "model": "(none)",
                            "base_url": "(none)", "reason": "empty-answer",
                        }
                        unknown_qids.append(qid)
                    else:
                        predictions[qid] = answer
                        qid_to_llm[qid] = llm_meta
                except Exception as exc:
                    primary_fail_count += 1
                    logger.warning(
                        "qid=%s LLM raised (%s: %s)",
                        qid, type(exc).__name__, exc,
                    )
                    predictions[qid] = "unknown"
                    retrieval_predictions.setdefault(qid, [])
                    qid_to_llm[qid] = {
                        "tier": "none", "model": "(none)",
                        "base_url": "(none)",
                        "reason": f"exception:{type(exc).__name__}",
                    }
                    unknown_qids.append(qid)
            Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            with open(retrieval_output_json, "w", encoding="utf-8") as f:
                json.dump(retrieval_predictions, f, ensure_ascii=False, indent=2)
            _write_models_sidecar(models_sidecar_json, qid_to_llm)
            logger.info(
                "inference summary | total=%d | unknown=%d | exceptions=%d",
                len(questions), len(unknown_qids), primary_fail_count,
            )
            if unknown_qids:
                logger.warning(
                    "inference finished with %d 'unknown' predictions out of %d; "
                    "first 10 qids: %s",
                    len(unknown_qids), len(questions), unknown_qids[:10],
                )

    run_with_session_stdio_tee(Path(__file__).resolve().parent, "inference", _inference_cli_main)
