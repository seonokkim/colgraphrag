import io
import json
import logging
import os
import re
from glob import glob
from datetime import datetime
from pathlib import Path

import networkx as nx
import torch

from prompt import *

from util.request import load_pretrained_model_with_fallback
from util.run_id import default_stamp
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

DRY_RUN = os.getenv("INFERENCE_DRY_RUN", "0") == "1"
_HF_GEMMA_MODULE = None


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
    """Generate answer text using in-process HF Gemma 4 E4B IT."""
    gemma = _get_hf_gemma_module()
    if gemma is None:
        raise RuntimeError("HF Gemma module not available or not configured")
    text = gemma.generate_text(prompt, max_new_tokens=512)
    llm_meta = {
        "tier": "hf_gemma_4_e4b_it",
        "model": os.getenv("GEMMA4_E4B_IT_MODEL_PATH", "/workspace/models/mllm/gemma-4-e4b-it"),
        "base_url": "(in-process)",
    }
    return text, llm_meta
_BASE_DIR = Path(__file__).resolve().parent
_RUN_ID = os.getenv("MMGRAPHRAG_RUN_ID", default_stamp()).strip()
GRAPH_DIR = os.getenv(
    "INFERENCE_GRAPH_DIR",
    str(_BASE_DIR / "result" / _RUN_ID / "phase4_graphs_real"),
)
_SLICE = _BASE_DIR / "result" / _RUN_ID / "webqa_slice"
QUESTION_FILE = os.getenv(
    "INFERENCE_QUESTION_FILE",
    str(_SLICE / "webqa_questions.jsonl"),
)
OUTPUT_JSON = os.getenv(
    "INFERENCE_OUTPUT_JSON",
    str(_BASE_DIR / "result" / _RUN_ID / "phase5_predictions_real.json"),
)
INFERENCE_RETRIEVAL_JSON = os.getenv("INFERENCE_RETRIEVAL_JSON", "")
MAX_QUESTIONS = int(os.getenv("INFERENCE_MAX_QUESTIONS", "20"))
LOG_ROOT = Path(os.getenv("MMGRAPHRAG_LOG_DIR", str(_BASE_DIR / "logs")))
_LOG_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_FILE_PATH = LOG_ROOT / f"{_LOG_TIMESTAMP}_colembed_inference.log"

model = None
processor = None
logger = logging.getLogger("colembed_mm_graph_rag")


def _default_retrieval_json_path(output_json_path: str) -> str:
    if output_json_path.endswith(".json"):
        return output_json_path[:-5] + "_retrieval.json"
    return output_json_path + "_retrieval.json"


def _default_models_sidecar_path(output_json_path: str) -> str:
    """Sidecar path that records *which* LLM tier produced each qid's answer.

    Shape on disk::

        {
          "meta": {"run_id": ..., "configured_primary": ..., "fallback_chain": ...},
          "qid_to_llm": {
            "<qid>": {"tier": "fallback-2", "model": "gemma4:e2b", ...},
            ...
          },
          "summary": {
            "by_tier":  {"primary": 0, "fallback-2": 100, ...},
            "by_model": {"gemma4:e2b": 100, ...}
          }
        }

    Written next to ``phase5_predictions_real.json`` as
    ``phase5_predictions_real_models.json`` so the evaluator's original
    ``{qid: answer_str}`` file stays untouched.
    """
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
    """Tally ``{tier -> count}`` and ``{model -> count}`` for the sidecar."""
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
    """Persist the qid->LLM attribution map with a run-level summary.

    Kept idempotent and side-effect-free beyond the file write so the
    real-inference loop can call it unconditionally.
    """
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


def extract_ranked_source_ids_from_graph(graph: nx.Graph, top_k: int = 10):
    ranked = []
    seen = set()
    for node_name, node_data in graph.nodes(data=True):
        source_id = node_data.get("source_id")
        if not source_id:
            continue
        sid = str(source_id)
        if sid in seen:
            continue
        seen.add(sid)
        ranked.append(
            {
                "id": sid,
                "score": float(graph.degree[node_name]),
            }
        )
    ranked.sort(key=lambda x: (-x["score"], x["id"]))
    return ranked[:top_k]


def _default_webqa_imgs_root() -> Path:
    env_root = os.getenv("WEBQA_DATA_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
    else:
        local = _BASE_DIR / "data" / "webqa"
        root = local if local.is_dir() else Path("/workspace/data/webqa")
    return Path(
        os.getenv("WEBQA_IMGS_DIR", str(root / "WebQA_data_first_release" / "imgs"))
    )


def _load_webqa_slice_images_index(slice_dir: Path) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    p = slice_dir / "webqa_images.jsonl"
    if not p.is_file():
        return idx
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("id", "")).strip()
            if rid:
                idx[rid] = row
    return idx


def _resolve_webqa_image_fs_path(row: dict, imgs_root: Path) -> str | None:
    raw = (row.get("path") or "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return str(p.resolve())
        rel = imgs_root / raw.lstrip("/\\")
        if rel.is_file():
            return str(rel.resolve())
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
    Query--image MaxSim scores via the ColEmbed VL checkpoint (GPU when available).
    Returns (ranked entries, success).
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
    Extract a meaningful answer from the graph for dry-run mode.
    
    Priority:
    1. TEXT nodes that are not placeholders and not the question itself
    2. IMAGE node descriptions (non-placeholder)
    3. Fallback to first non-placeholder entity_name
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
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info("Logger initialized. log_file=%s", _LOG_FILE_PATH)

    request_logger = logging.getLogger("webqa.request")
    if not request_logger.handlers:
        request_logger.setLevel(logging.INFO)
        request_logger.addHandler(file_handler)
        request_logger.addHandler(stream_handler)
        request_logger.propagate = False
    return logger

def _colembed_local_first_default() -> str:
    """Local-first: <repo>/models/retriever/..., then /workspace/models/retriever/..."""
    local = Path(__file__).resolve().parent / "models" / "retriever" / "llama-nemotron-colembed-vl-3b-v2"
    if local.is_dir():
        return str(local)
    return "/workspace/models/retriever/llama-nemotron-colembed-vl-3b-v2"


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
    """Inference-time ``max_input_tiles`` for ``AutoProcessor``.

    Default **8** follows the arXiv-26 Nemotron ColEmbed V2 paper
    recommendation (``reference/paper/arXiv-26_ColEmbed V2-2602.03992v1/
    content/3_colembedv2.tex`` § 3.3: dynamic tiling with
    ``max_input_tiles=8`` at inference). The checkpoint shipped in
    ``C:\\workspace\\models\\retriever\\llama-nemotron-colembed-vl-3b-v2``
    defaults to ``max_input_tiles=6`` in its processor config, which is
    visibly too coarse for fine-grained WebQA visual attributes
    (color / shape / number) — see
    ``colgraphrag_webqa/.dev_document/md/20260422_215230_webqa_evaluation_and_dev.md``.

    Override via env: ``set COLEMBED_MAX_INPUT_TILES=4`` etc.
    Values ``<= 0`` (including malformed strings) fall back to **8**.
    """
    raw = os.getenv("COLEMBED_MAX_INPUT_TILES", "").strip()
    if not raw:
        return 8
    try:
        parsed = int(raw)
    except ValueError:
        return 8
    return parsed if parsed > 0 else 8


def resolve_colembed_topk_images() -> int:
    """Image shortlist size fed to the VLM reranker from ``text_to_image_feature``.

    Default **10** (was hard-coded ``n=3``). For WebQA color / shape /
    Others questions the gold image is frequently outside the top-3, so
    widening the ColEmbed shortlist before the VLM selector materially
    helps recall without changing the selector contract -- see the
    score-improvement plan (Step #2) in
    ``20260422_215230_webqa_evaluation_and_dev.md``.

    Override via env: ``set COLEMBED_TOPK_IMAGES=5``.
    Values ``<= 0`` (including malformed strings) fall back to **10**.
    """
    raw = os.getenv("COLEMBED_TOPK_IMAGES", "").strip()
    if not raw:
        return 10
    try:
        parsed = int(raw)
    except ValueError:
        return 10
    return parsed if parsed > 0 else 10


def _load_colembed_processor(model_ref: str, trust_remote_code: bool):
    """Single place that instantiates the ColEmbed ``AutoProcessor``.

    Threads ``max_input_tiles=resolve_colembed_max_input_tiles()`` through
    ``AutoProcessor.from_pretrained(..., max_input_tiles=N)``. Older HF
    versions / custom processors that do not expose the kwarg fall back
    silently to the no-kwarg call so dev environments without the patched
    processor keep working. A one-shot info log records the effective
    value so operators can verify the env toggle took effect.
    """
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
    """Best-effort post-load override for processors that ignored the kwarg."""
    if hasattr(processor, "max_input_tiles"):
        try:
            setattr(processor, "max_input_tiles", tiles)
        except (AttributeError, TypeError):
            pass


_TILES_LOG_DONE = False


def _log_tiles_once(tiles: int, *, success: bool) -> None:
    """Info-log the effective ``max_input_tiles`` exactly once per process."""
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
    required = ("forward_queries", "forward_images", "get_scores")
    missing = [name for name in required if not hasattr(loaded_model, name)]
    if missing:
        raise RuntimeError(
            f"Colembed retrieval API missing required methods: {missing}. "
            "Model must support forward_queries/forward_images/get_scores."
        )


def probe_model_output_contract(model_ref: str = "", device_map: str = "auto") -> dict:
    """
    Phase-A contract probe:
    checks whether model forward outputs expose
    - text_embeds
    - image_embeds
    - logits_per_text
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
    # Keep probe tensors on same device as model to avoid cuda/cpu mismatch.
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
    try:
        # 使用 json.loads 将字符串转换为列表或字典
        result_list = json.loads(str_dict_list)
        return result_list
    except json.JSONDecodeError as e:
        return None

def graph_to_graphml_str(graph):
    with io.BytesIO() as byte_output:
        nx.write_graphml(graph, byte_output)
        byte_output.seek(0)
        graphml_str = byte_output.read().decode('utf-8')
    return graphml_str

def graph_to_str(graph):
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

def text_to_image_feature(image_paths, texts, n=None):
    """ColEmbed MaxSim shortlist: returns indices of the top-``n`` images for ``texts[0]``.

    ``n`` defaults to :func:`resolve_colembed_topk_images` (env
    ``COLEMBED_TOPK_IMAGES``, default **10**). Callers that still pass
    an explicit ``n`` keep their original behaviour unchanged. Raising
    ``n`` from the legacy hard-coded ``3`` is Step #2 of the score
    improvement plan — it widens the pre-VLM shortlist so color / shape
    / Others questions can recover from retrieval misses outside the
    top-3.

    The processor is instantiated via :func:`_load_colembed_processor`
    which threads ``COLEMBED_MAX_INPUT_TILES`` (default **8**, Step #3)
    into ``AutoProcessor.from_pretrained``.
    """
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
    # Keep legacy return contract: indices for the first text query.
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
    with open(path, "r", encoding='UTF-8') as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
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

    _slice_for_retrieval = Path(os.getenv("INFERENCE_WEBQA_SLICE_DIR", str(_SLICE)))
    _use_colembed_retrieval = os.getenv("INFERENCE_COLEMBED_RETRIEVAL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    _images_index = (
        _load_webqa_slice_images_index(_slice_for_retrieval) if _use_colembed_retrieval else {}
    )
    _imgs_root = _default_webqa_imgs_root()
    _topk_colembed = int(os.getenv("INFERENCE_COLEMBED_TOP_K", "10"))

    def _fill_retrieval(question_dict: dict, graph_obj: nx.Graph) -> list:
        if _use_colembed_retrieval:
            md = question_dict.get("metadata") or {}
            img_ids = md.get("image_doc_ids") or []
            ranked, ok = colembed_maxsim_ranked_sources(
                _get_question_text(question_dict),
                img_ids,
                _images_index,
                _imgs_root,
                _topk_colembed,
            )
            if ok and ranked:
                return ranked
        return extract_ranked_source_ids_from_graph(graph_obj, top_k=10)

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
                prompt = LLM_ANSWER_PROMPT.replace("{question}", _get_question_text(question)).replace(
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
