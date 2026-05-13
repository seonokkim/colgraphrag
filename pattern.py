"""
Phase 2 — 질문별 그래프 패턴(스키마 힌트) 합성.

역할:
  - 슬라이스의 각 질문을 ``GRAPH_PATTERN_PROMPT`` 에 넣어 로컬 Gemma 로 완성문을 받고,
    엔티티 타입 리스트 + ``##`` 로 구분된 관계 템플릿 문자열을 생성.
  - 결과는 ``phase2_pattern_cache/<qid>.json`` 에 저장 → Phase 3 가 다시 LLM 부르지 않고 재사용.

입력:
  - WebQA: 프로파일별 JSON 경로(``util.webqa_load``).
  - MMQA 등: ``mmqa_questions.jsonl`` 한 줄당 질문 dict.

출력 캐시 JSON 필드: ``response`` (패턴 문자열), ``question`` (원본), ``llm`` (메타) 등.
"""

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

# --- 런타임 설정: 환경변수로 동시 요청 수·입력 JSON·캐시 경로·상한·드라이런 전환 ---
# CONCURRENCY: 한 웨이브당 비동기 작업 수(질문 1개 → 캐시 파일 1개)
CONCURRENCY = int(os.getenv("PATTERN_CONCURRENCY", "16"))
_BASE_DIR = Path(__file__).resolve().parent
_DATASET = os.getenv("MMGRAPHRAG_DATASET", "webqa").strip().lower()
_RUN_ID = resolve_pipeline_run_id(_BASE_DIR, _DATASET)

# 질문 JSONL: WebQA 는 프로파일 헬퍼 경로, MMQA 는 기본적으로 mmqa_slice 의 questions
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
# MMQA/비-WebQA 에서만 ``PATTERN_MAX_SAMPLES`` 로 상한. WebQA 는 ``records_for_pattern`` 내부에서 자름.
MAX_SAMPLES = int(os.getenv("PATTERN_MAX_SAMPLES", "0"))
DRY_RUN = os.getenv("PATTERN_DRY_RUN", "0") == "1"
_HF_GEMMA_MODULE: Optional[object] = None

# --- 아래는 캐시 파일 형태 예시(result/<run>/phase2_pattern_cache/*.json 참고) ---
#
# (1) One input dict after ``load_json_data()`` (WebQA; other JSONL uses ``qid`` + ``question`` keys).
#
#       {
#         "Guid": "d5bc1c280dba11ecb1e81171463288e9",
#         "Q": "Which organ is adorned with more Christmas like colors ; ...?",
#         "txt_posFacts": [{"snippet_id": "..._txt", "fact": "Gaustadt pipe organ"}],
#         "img_posFacts": [{"image_id": "30073989", "url": "data/webqa/.../00073989_30073989.png"}],
#         ...
#       }
#
# (2) Prompt text: ``GRAPH_PATTERN_PROMPT`` with ``"{question}"`` substituted by ``record["Q"]`` (non-WebQA: ``question`` field).
#
# (3) Output path: ``<PATTERN_CACHE_DIR>/<Guid>.json`` (fallback key ``qid``.json outside WebQA bundles).
#
# (4) Output JSON saved by ``make_request`` (fields vary slightly by run mode):
#
#       {
#         "response":
#           "[\"ENTITY\",\"ATTRIBUTE\"]##ENTITY <|> related_to <|> ATTRIBUTE<|COMPLETE|>",
#         "question": { "...full WebQA record echoed for traceability..." },
#         "dry_run": true,
#         "created_at": "2026-05-11T13:38:58.368846",
#         "llm": {"tier": "dry-run", "model": "(none)", "base_url": "(none)"}
#       }
#
# ``PATTERN_DRY_RUN=0`` omits the ``dry_run``/``created_at`` keys shown above and fills ``response`` plus ``llm`` from HF;
# grammar for ``response`` matches ``prompt.GRAPH_PATTERN_PROMPT``: type list ``[...]`` then ``##`` relation blocks ending in ``<|COMPLETE|>``.


def _get_hf_gemma_module():
    """지연 로드: ``mllm.hf_gemma_4_e4b_it`` 가 구성되어 있을 때만 모듈 객체 반환."""
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
    """동기 1회 디코딩; 반환 텍스트와 캐시에 붙일 ``llm`` 메타."""
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
    """질문 행 로드: WebQA 경로면 프로파일·슬라이싱 유틸, 그 외 일반 JSONL 1줄=1객체."""
    if is_webqa_json_path(JSON_FILE_PATH):
        return records_for_pattern(JSON_FILE_PATH, resolve_profile())
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def hash_prompt(prompt: str) -> str:
    # 과거 MD5 키 캐시용; 현재 파일명은 Guid/qid 기준
    return hashlib.md5(prompt.encode()).hexdigest()


def validate_json_file(file_path):
    """Return True only if the candidate cache file parses as strict JSON."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json.load(file)
        return True
    except json.JSONDecodeError:
        return False
    except Exception:
        return False


async def make_request(session: aiohttp.ClientSession, prompt: str, data: Dict):
    """
    질문 1건 → 캐시 JSON 1개. 이미 유효 JSON 이 있으면 스킵(손상 시 삭제 후 재생성).
    ``session`` 은 배치 오케스트레이션 호환용; 실제 LLM 호출은 프로세스 내 HF Gemma.
    """
    webqa = is_webqa_json_path(JSON_FILE_PATH)
    key = data["Guid"] if webqa else data["qid"]
    cache_file = f"{CACHE_DIR}/{key}.json"

    if os.path.exists(cache_file):
        if validate_json_file(cache_file):
            return
        os.remove(cache_file)

    if DRY_RUN:
        # 드라이런: CUDA/가중치 없이 결정적 스캐폴드로 downstream 파서 검증 가능
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
    """start_index 부터 최대 CONCURRENCY 개까지 프롬프트 발행; 각 완성이 개별 JSON 파일로 기록됨."""
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
    # 로드 후 MMQA 등 비-WebQA 만 여기서 [:MAX_SAMPLES]. WebQA 는 records_for_pattern 안에서 이미 잘림.
    if not is_webqa_json_path(JSON_FILE_PATH) and MAX_SAMPLES > 0:
        json_data = json_data[:MAX_SAMPLES]
    total_batches = max(1, math.ceil(len(json_data) / CONCURRENCY))
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(json_data), CONCURRENCY):
            await process_batch(session, template, json_data, i)
            current_batch = i // CONCURRENCY + 1
            _ = (current_batch / total_batches) * 100  # 배치 진행률(로깅 훅용으로 유지)


if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from util.pipeline_session_log import run_with_session_stdio_tee

    _repo = Path(__file__).resolve().parent
    run_with_session_stdio_tee(_repo, "pattern", lambda: asyncio.run(main()))
