"""
Phase 3 --- relation extraction conditioned on Phase 2 graph patterns.

For each question row in ``*_questions.jsonl`` and referenced text snippets in ``*_texts.jsonl``,
this stage builds ``EXTRACT_RELATION_PROMPT`` (``prompt.EXTRACT_RELATION_PROMPT``) by injecting the parsed
pattern from ``phase2_pattern_cache/<Guid>.json`` and the slice text passages. Outputs are keyed JSON blobs
stored under ``phase3_extraction_cache`` so downstream graph construction can stay deterministic.
"""

import asyncio
import os
import html
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import re
import aiohttp

from util.llm_defaults import DEFAULT_GEMMA4_E4B_IT_MODEL_PATH, ensure_default_gemma4_e4b_it_path
from util.repo_config import forbid_dry_run, require_cuda, require_gemma_local
from util.result_layout import resolve_pipeline_run_id
from prompt import EXTRACT_RELATION_PROMPT

ensure_default_gemma4_e4b_it_path()

# --- Runtime configuration (environment overrides mirror tests / notebooks) ---
# CONCURRENCY: max parallel ``make_request`` tasks buffered before awaiting the batch.
CONCURRENCY = int(os.getenv("EXTRACTION_CONCURRENCY", "8"))
_BASE_DIR = Path(__file__).resolve().parent
_DATASET = os.getenv("MMGRAPHRAG_DATASET", "webqa").strip().lower()
_RUN_ID = resolve_pipeline_run_id(_BASE_DIR, _DATASET)
CACHE_DIR = os.getenv(
    "EXTRACTION_CACHE_DIR",
    str(_BASE_DIR / "result" / _RUN_ID / "phase3_extraction_cache"),
)
_SLICE = _BASE_DIR / "result" / _RUN_ID / f"{_DATASET}_slice"
# Questions / texts slices follow ``<dataset>_questions.jsonl`` naming (``webqa_questions.jsonl``, ``mmqa_questions.jsonl``).
QUESTION_FILE = os.getenv(
    "EXTRACTION_QUESTION_FILE",
    str(_SLICE / f"{_DATASET}_questions.jsonl"),
)
TEXT_FILE = os.getenv(
    "EXTRACTION_TEXT_FILE",
    str(_SLICE / f"{_DATASET}_texts.jsonl"),
)
PATTERN_CACHE_DIR = os.getenv(
    "EXTRACTION_PATTERN_CACHE_DIR",
    str(_BASE_DIR / "result" / _RUN_ID / "phase2_pattern_cache"),
)
MAX_QUESTIONS = int(os.getenv("EXTRACTION_MAX_QUESTIONS", "20"))
DRY_RUN = os.getenv("EXTRACTION_DRY_RUN", "0") == "1"
_HF_GEMMA_MODULE: Optional[object] = None

# --- Example artifact layout (representative repo path: ``result/webqa/<run>/phase3_extraction_cache``) ---
#
# (1) Questions JSONL excerpt (each line JSON; WebQA uses ``Guid`` + ``Q``, MMQA often uses ``qid`` + ``question``).
#
# (2) Pattern sidecar consumed from Phase 2: ``phase2_pattern_cache/<Guid>.json`` with string field ``response``
#     matching GRAPH_PATTERN_PROMPT grammar (parsed by ``process_graph_pattern``).
#
# (3) Texts JSONL: each line has ``{"id": "<snippet_id>", "text": "..."}``. ``question["metadata"]["text_doc_ids"]``
#     selects which snippets to concatenate (truncated here to keep prompts bounded).
#
# (4) Output filename: ``<EXTRACTION_CACHE_DIR>/<qid>_<text_doc_id>.json`` --- see ``task_qid`` in ``main``.
#     Minimal body (adapted from run ``20260512_144218_notebook_tutorial`` artifacts):
#
#       {
#         "response":
#           '\"entity\"<|>\"Coprinus comatus\"...##\"relationship\"<|>\"...\"<|>\"...\"<|>\"have different cap colors\"',
#         "llm": {"tier": "hf_gemma_4_e4b_it", "model": ".../gemma-4-E4B-it", "base_url": "(in-process)"},
#         "qid": "d5cb27900dba11ecb1e81171463288e9_d5cb27900dba11ecb1e81171463288e9_txt",
#         "text_content": "Question: Are the tops ...",
#         "graph_pattern": {"type_list": [...], "edge_list": [...], "question_text": "..."}
#       }
#
# ``response`` follows EXTRACT_RELATION_PROMPT delimiter rules (``entity`` / ``relationship`` records joined by ``##``).
#
# Dry runs synthesize placeholders via string templates rather than invoking Gemma.


def _get_hf_gemma_module():
    """Lazy import of the HF Gemma wrapper (shared contract with ``pattern.py``)."""
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


def _hf_generate_text(prompt: str) -> Tuple[str, dict]:
    """Single synchronous decoding pass; attaches lightweight provenance metadata for cache JSON."""
    gemma = _get_hf_gemma_module()
    if gemma is None:
        raise RuntimeError("HF Gemma module not available or not configured")
    text = gemma.generate_text(prompt, max_new_tokens=512)
    llm_meta = {
        "tier": "hf_gemma_4_e4b_it",
        "model": os.getenv("GEMMA4_E4B_IT_MODEL_PATH", DEFAULT_GEMMA4_E4B_IT_MODEL_PATH),
        "base_url": "(in-process)",
    }
    return text, llm_meta


def _get_qid(record: dict) -> str:
    """Get question ID from record (supports both 'qid' and 'Guid' formats)."""
    return record.get("qid") or record.get("Guid") or ""

async def load_question_data(path: str) -> List[Dict]:
    """Load JSONL questions; returns dict[str, record] keyed by ``Guid`` or ``qid`` (still typed loosely)."""
    with open(path, "r", encoding="UTF-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
        return {_get_qid(r): r for r in records}


async def load_text_data(path: str) -> Dict[str, Dict]:
    """Load JSONL text bundles keyed by snippet ``id`` for quick lookup while assembling prompts."""
    with open(path, "r", encoding="UTF-8") as file:
        return {json.loads(line)["id"]: json.loads(line) for line in file}


def hash_prompt(prompt: str) -> str:
    """MD5 fingerprint helper (legacy hooks / debugging; cache keys presently use structured qid strings)."""
    return hashlib.md5(prompt.encode()).hexdigest()


def cache_exists(prompt_hash: str) -> bool:
    """Return True if a legacy MD5-keyed cache artifact already exists under ``CACHE_DIR``."""
    return os.path.exists(f"{CACHE_DIR}/{prompt_hash}.json")


def validate_json_file(file_path):
    """Return True only when ``file_path`` parses as strict JSON."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json.load(file)
        return True
    except json.JSONDecodeError as e:
        return False
    except Exception as e:
        return False


def str_list(str: str) -> List[str]:
    """Parse the bracketed TYPE list emitted as the leading segment of GRAPH_PATTERN_PROMPT completions."""
    matches = re.search(r'\[([^\]]+)\]', str)
    if matches:
        entity_types_list = matches.group(1).split(", ")
        entity_types_list = [item.strip() for item in entity_types_list]
    else:
        entity_types_list = []
    return entity_types_list


def clean_str(input: Any) -> str:
    """Collapse HTML entities/control chars so dry-run scaffolding can stringify safely."""
    if not isinstance(input, str):
        return input
    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


async def process_graph_pattern(graph_pattern: str):
    """Split the condensed pattern head string into editable type tokens and typed edge templates."""
    record_delimiter = "##"
    tuple_delimiter = "<|>"  # carried for readability / parity with pattern grammar
    complete_delimiter = "<|COMPLETE|>"
    records = [r.strip() for r in graph_pattern.split(record_delimiter)]
    entity_list = str_list(records[0])
    edge_type = []
    for record in records[1:]:
        edge_type.append(record.replace(complete_delimiter, ""))
    return {"type_list": entity_list, "edge_list": edge_type}


async def make_request(session: aiohttp.ClientSession, prompt: str, text_content: str, qid: str, graph_pattern: Dict):
    """
    One (question, snippet) pair triggers one deterministic cache file identified by composite ``qid``.

    Existing cache entries short-circuit to keep reruns idempotent under the same EXTRACTION_CACHE_DIR.
    Prompt assembly splices concatenated corpus text plus pattern hints before Gemma decoding (dry-run uses templates).
    """
    if session is None:
        print("Error: session is None")
        return
    cache_file = f"{CACHE_DIR}/{qid}.json"
    if os.path.exists(cache_file):
        return
    prompt = prompt.replace("{input_text}", text_content)
    result = {}

    if DRY_RUN:
        question_text = graph_pattern.get("question_text", "")
        q_entity = clean_str(question_text).strip("?").strip().replace('"', "")[:120]
        t_entity = clean_str(text_content.split(".")[0]).strip().replace('"', "")[:120]
        if not t_entity:
            t_entity = "CONTEXT"
        result["response"] = (
            f'"entity"<|>"{q_entity}"<|>"QUESTION"<|>"query entity"##'
            f'"entity"<|>"{t_entity}"<|>"TEXT"<|>"context entity"##'
            f'"relationship"<|>"{q_entity}"<|>"{t_entity}"<|>"related context"'
        )
        result["llm"] = {"tier": "dry-run", "model": "(none)", "base_url": "(none)"}
    else:
        content, llm_meta = _hf_generate_text(prompt)
        result['response'] = content
        result['llm'] = llm_meta
    result['qid'] = qid
    result['text_content'] = text_content
    result['graph_pattern'] = graph_pattern
    with open(cache_file, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


async def main():
    """Enumerate limited questions, join slice texts, hydrate prompts from Phase 2, fan out concurrency-limited batches."""
    forbid_dry_run("extraction", dry_run=DRY_RUN)
    if not DRY_RUN:
        require_cuda("extraction")
        require_gemma_local("extraction")
    os.makedirs(CACHE_DIR, exist_ok=True)
    questions = await load_question_data(QUESTION_FILE)
    text_data = await load_text_data(TEXT_FILE)
    question_items = list(questions.items())[:MAX_QUESTIONS] if MAX_QUESTIONS > 0 else list(questions.items())
    k = 1
    async with aiohttp.ClientSession() as session:
        tasks = []
        for key, question in question_items:
            qid = key
            pattern_file = os.path.join(PATTERN_CACHE_DIR, f"{qid}.json")
            # Default graph scaffolding when Phase 2 file missing / empty --- ``question_text`` prefers ``question`` (MMQA);
            # many WebQA rows store prose under ``Q`` instead, so blanks there only affect dry-run heuristics.
            graph_pattern = {"type_list": [], "edge_list": [], "question_text": question.get("question", "")}
            if os.path.exists(pattern_file):
                with open(pattern_file, "r", encoding="utf-8") as file:
                    graph_pattern_raw = json.load(file)
                graph_pattern = await process_graph_pattern(graph_pattern_raw.get("response", ""))
                graph_pattern["question_text"] = question.get("question", "")
            new_template = EXTRACT_RELATION_PROMPT.replace("{Graph_pattern}",
                                            "Entity types: [" + ",".join(graph_pattern["type_list"]) + "]\n" + "\n".join(
                                                graph_pattern["edge_list"]))
            # WebQA attaches up to two text snippet ids referencing ``*_texts.jsonl``.
            text_doc_ids = question.get("metadata", {}).get("text_doc_ids", [])[:2]
            joined_texts = []
            for text_doc_id in text_doc_ids:
                entry = text_data.get(text_doc_id)
                if entry and entry.get("text"):
                    joined_texts.append(entry["text"][:1000])
            text_content = "\n".join(joined_texts)
            if not text_doc_ids:
                text_doc_ids = [f"{qid}_NO_TEXT"]
            # Composite stem keeps per-snippet caches distinct inside ``phase3_extraction_cache``.
            for text_doc_id in text_doc_ids:
                task_qid = f"{qid}_{text_doc_id}"
                task = make_request(session, new_template, text_content, task_qid, graph_pattern)
                tasks.append(task)
                if len(tasks) >= CONCURRENCY:
                    await asyncio.gather(*tasks)
                    tasks = []

            k += 1
            print(f'{datetime.now()}:{k / len(question_items)}')  # coarse progress (fraction of ``question_items``)
        if len(tasks) > 0:
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from util.pipeline_session_log import run_with_session_stdio_tee

    _repo = Path(__file__).resolve().parent
    run_with_session_stdio_tee(_repo, "extraction", lambda: asyncio.run(main()))