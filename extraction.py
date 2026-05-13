"""
Phase 3 — Phase 2 그래프 패턴을 조건으로 한 관계 추출(Relation extraction).

파이프라인 흐름:
  1) ``*_questions.jsonl`` 의 각 질문과 ``metadata.text_doc_ids`` 가 가리키는 ``*_texts.jsonl`` 스니펫을 읽음.
  2) ``phase2_pattern_cache/<qid>.json`` 의 ``response`` 를 ``process_graph_pattern`` 으로
     타입 리스트 + 엣지 템플릿 구조로 파싱(없으면 빈 스캐폴드).
  3) ``EXTRACT_RELATION_PROMPT`` 에 패턴과 본문을 넣어 Gemma 에게 엔티티/관계 레코드 문자열 생성 요청.
  4) ``phase3_extraction_cache/<qid>_<text_doc_id>.json`` 으로 저장 → Phase 4 ``construct.py`` 가 소비.

출력 ``response`` 형식: ``##`` 로 레코드 구분, ``entity`` / ``relationship`` 튜플은 ``<|>`` 구분(``prompt.py`` 참고).

Dry-run(``EXTRACTION_DRY_RUN=1``)은 CUDA 없이 템플릿 문자열만 써서 캐시 형태를 맞춤.
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

# --- 런타임 설정: 테스트/노트북과 동일하게 환경변수로 덮어쓰기 ---
# CONCURRENCY: ``make_request`` 비동기 태스크를 묶어 await 하기 전 최대 개수
CONCURRENCY = int(os.getenv("EXTRACTION_CONCURRENCY", "8"))
_BASE_DIR = Path(__file__).resolve().parent
_DATASET = os.getenv("MMGRAPHRAG_DATASET", "webqa").strip().lower()
_RUN_ID = resolve_pipeline_run_id(_BASE_DIR, _DATASET)
CACHE_DIR = os.getenv(
    "EXTRACTION_CACHE_DIR",
    str(_BASE_DIR / "result" / _RUN_ID / "phase3_extraction_cache"),
)
_SLICE = _BASE_DIR / "result" / _RUN_ID / f"{_DATASET}_slice"
# 슬라이스 질문·텍스트: ``<dataset>_questions.jsonl`` / ``<dataset>_texts.jsonl`` (webqa_* / mmqa_*)
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

# --- 산출물 예시(result/<run>/phase3_extraction_cache) ---
# (1) 질문 JSONL: WebQA 는 Guid+Q, MMQA 는 qid+question 등 필드명 혼용.
# (2) Phase 2 ``phase2_pattern_cache/<qid>.json`` 의 ``response`` → 이 파일의 ``process_graph_pattern``.
# (3) 텍스트 JSONL: ``{"id","text"}``; ``metadata.text_doc_ids`` 가 붙일 스니펫 선택(프롬프트 길이 제한으로 잘림).
# (4) 출력 파일명: ``<qid>_<text_doc_id>.json`` — ``main`` 의 ``task_qid``.
#     본문 최소 필드: response, llm, qid, text_content, graph_pattern.
# Dry-run 은 Gemma 대신 문자열 템플릿으로 ``response`` 채움.


def _get_hf_gemma_module():
    """``pattern.py`` 와 동일 계약으로 HF Gemma 래퍼 지연 import."""
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
    """동기 1패스 디코딩; 캐시 JSON 에 넣을 ``llm`` 메타를 함께 반환."""
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
    """JSONL 질문 로드; ``Guid`` 또는 ``qid`` 를 키로 하는 dict 반환."""
    with open(path, "r", encoding="UTF-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
        return {_get_qid(r): r for r in records}


async def load_text_data(path: str) -> Dict[str, Dict]:
    """텍스트 JSONL 을 snippet ``id`` → 레코드 dict 로 색인."""
    with open(path, "r", encoding="UTF-8") as file:
        return {json.loads(line)["id"]: json.loads(line) for line in file}


def hash_prompt(prompt: str) -> str:
    """레거시/디버그용 MD5 지문; 캐시 키는 현재 구조화된 qid 문자열."""
    return hashlib.md5(prompt.encode()).hexdigest()


def cache_exists(prompt_hash: str) -> bool:
    """과거 MD5 파일명 규약으로 ``CACHE_DIR`` 에 객체가 있는지 여부."""
    return os.path.exists(f"{CACHE_DIR}/{prompt_hash}.json")


def validate_json_file(file_path):
    """엄격 JSON 로 파싱 가능하면 True."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json.load(file)
        return True
    except json.JSONDecodeError as e:
        return False
    except Exception as e:
        return False


def str_list(str: str) -> List[str]:
    """GRAPH_PATTERN_PROMPT 선두의 대괄호 타입 리스트 문자열을 토큰 리스트로 파싱."""
    matches = re.search(r'\[([^\]]+)\]', str)
    if matches:
        entity_types_list = matches.group(1).split(", ")
        entity_types_list = [item.strip() for item in entity_types_list]
    else:
        entity_types_list = []
    return entity_types_list


def clean_str(input: Any) -> str:
    """드라이런 문자열 조합용: HTML 실체화 + 제어 문자 제거."""
    if not isinstance(input, str):
        return input
    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


async def process_graph_pattern(graph_pattern: str):
    """Phase 2 ``response`` 한 덩어리 → type_list / edge_list 로 분해."""
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
    (질문, 특정 text_doc_id) 조합당 캐시 파일 1개. 기존 파일이 있으면 그대로 두어 재실행 멱등.
    ``{input_text}`` 플레이스홀더를 실제 스니펫 본문으로 치환한 뒤 Gemma 호출(드라이런은 템플릿).
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
    """상한 질문만 순회하며 Phase 2 패턴 + 텍스트로 프롬프트 구성, 동시성 제한 배치로 캐시 생성."""
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
            # Phase 2 캐시 부재 시 빈 그래프 스캐폴드. ``question_text`` 는 MMQA ``question`` 필드 우선; WebQA 는 보통 ``Q``.
            graph_pattern = {"type_list": [], "edge_list": [], "question_text": question.get("question", "")}
            if os.path.exists(pattern_file):
                with open(pattern_file, "r", encoding="utf-8") as file:
                    graph_pattern_raw = json.load(file)
                graph_pattern = await process_graph_pattern(graph_pattern_raw.get("response", ""))
                graph_pattern["question_text"] = question.get("question", "")
            new_template = EXTRACT_RELATION_PROMPT.replace("{Graph_pattern}",
                                            "Entity types: [" + ",".join(graph_pattern["type_list"]) + "]\n" + "\n".join(
                                                graph_pattern["edge_list"]))
            # WebQA: 최대 2개 텍스트 스니펫 id 만 사용(``*_texts.jsonl`` 조회)
            text_doc_ids = question.get("metadata", {}).get("text_doc_ids", [])[:2]
            joined_texts = []
            for text_doc_id in text_doc_ids:
                entry = text_data.get(text_doc_id)
                if entry and entry.get("text"):
                    joined_texts.append(entry["text"][:1000])
            text_content = "\n".join(joined_texts)
            if not text_doc_ids:
                text_doc_ids = [f"{qid}_NO_TEXT"]
            # 파일 stem: ``phase3_extraction_cache`` 안에서 스니펫별 결과를 분리 저장
            for text_doc_id in text_doc_ids:
                task_qid = f"{qid}_{text_doc_id}"
                task = make_request(session, new_template, text_content, task_qid, graph_pattern)
                tasks.append(task)
                if len(tasks) >= CONCURRENCY:
                    await asyncio.gather(*tasks)
                    tasks = []

            k += 1
            print(f'{datetime.now()}:{k / len(question_items)}')  # 대략적 진행률(처리 중인 질문 인덱스 비율)
        if len(tasks) > 0:
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    from util.pipeline_session_log import run_with_session_stdio_tee

    _repo = Path(__file__).resolve().parent
    run_with_session_stdio_tee(_repo, "extraction", lambda: asyncio.run(main()))