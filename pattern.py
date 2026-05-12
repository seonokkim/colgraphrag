import asyncio
import os
from datetime import datetime
from pathlib import Path

import aiohttp
import hashlib
import json
from typing import List, Dict, Optional, Tuple

import math

from prompt import GRAPH_PATTERN_PROMPT
from util.llm_defaults import DEFAULT_GEMMA4_E4B_IT_MODEL_PATH, ensure_default_gemma4_e4b_it_path
from util.repo_config import forbid_dry_run, require_cuda, require_gemma_local
from util.result_layout import resolve_pipeline_run_id
from util.webqa_load import (
    default_pattern_json_path,
    is_webqa_json_path,
    records_for_pattern,
    resolve_profile,
)

ensure_default_gemma4_e4b_it_path()

CONCURRENCY = int(os.getenv("PATTERN_CONCURRENCY", "16"))
_BASE_DIR = Path(__file__).resolve().parent
_DATASET = os.getenv("MMGRAPHRAG_DATASET", "webqa").strip().lower()
_RUN_ID = resolve_pipeline_run_id(_BASE_DIR, _DATASET)

_default_json = (
    str(_BASE_DIR / "result" / _RUN_ID / "mmqa_slice" / "mmqa_questions.jsonl")
    if _DATASET == "mmqa"
    else str(default_pattern_json_path())
)
JSON_FILE_PATH = os.getenv("PATTERN_JSON_FILE_PATH", _default_json)
CACHE_DIR = os.getenv(
    "PATTERN_CACHE_DIR",
    str(_BASE_DIR / "result" / _RUN_ID / "phase2_pattern_cache"),
)
MAX_SAMPLES = int(os.getenv("PATTERN_MAX_SAMPLES", "0"))
DRY_RUN = os.getenv("PATTERN_DRY_RUN", "0") == "1"
_HF_GEMMA_MODULE: Optional[object] = None


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


def _hf_generate_text(prompt: str) -> Tuple[str, dict]:
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


async def load_json_data() -> List[Dict]:
    if is_webqa_json_path(JSON_FILE_PATH):
        return records_for_pattern(JSON_FILE_PATH, resolve_profile())
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def validate_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json.load(file)
        return True
    except json.JSONDecodeError:
        return False
    except Exception:
        return False


async def make_request(session: aiohttp.ClientSession, prompt: str, data: Dict):
    webqa = is_webqa_json_path(JSON_FILE_PATH)
    key = data["Guid"] if webqa else data["qid"]
    cache_file = f"{CACHE_DIR}/{key}.json"

    if os.path.exists(cache_file):
        if validate_json_file(cache_file):
            return
        os.remove(cache_file)

    if DRY_RUN:
        result = {
            "response": '["ENTITY","ATTRIBUTE"]##ENTITY <|> related_to <|> ATTRIBUTE<|COMPLETE|>',
            "question": data,
            "dry_run": True,
            "created_at": datetime.now().isoformat(),
            "llm": {"tier": "dry-run", "model": "(none)", "base_url": "(none)"},
        }
        out_path = f"{CACHE_DIR}/{key}.json"
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        return

    content, llm_meta = _hf_generate_text(prompt)
    result = {"response": content, "question": data, "llm": llm_meta}
    with open(f"{CACHE_DIR}/{key}.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


async def process_batch(session: aiohttp.ClientSession, template: str, json_data: List[Dict], start_index: int):
    tasks = []
    webqa = is_webqa_json_path(JSON_FILE_PATH)
    for i in range(start_index, min(start_index + CONCURRENCY, len(json_data))):
        qtext = json_data[i]["Q"] if webqa else json_data[i]["question"]
        prompt = template.replace("{question}", qtext)
        tasks.append(make_request(session, prompt, json_data[i]))
    await asyncio.gather(*tasks)


async def main():
    forbid_dry_run("pattern", dry_run=DRY_RUN)
    if not DRY_RUN:
        require_cuda("pattern")
        require_gemma_local("pattern")
    os.makedirs(CACHE_DIR, exist_ok=True)
    template = GRAPH_PATTERN_PROMPT
    json_data = await load_json_data()
    if not is_webqa_json_path(JSON_FILE_PATH) and MAX_SAMPLES > 0:
        json_data = json_data[:MAX_SAMPLES]
    total_batches = max(1, math.ceil(len(json_data) / CONCURRENCY))
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(json_data), CONCURRENCY):
            await process_batch(session, template, json_data, i)
            current_batch = i // CONCURRENCY + 1
            _ = (current_batch / total_batches) * 100


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from util.pipeline_session_log import run_with_session_stdio_tee

    _repo = Path(__file__).resolve().parent
    run_with_session_stdio_tee(_repo, "pattern", lambda: asyncio.run(main()))
