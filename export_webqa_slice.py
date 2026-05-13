"""
Phase 1(데이터 준비) — WebQA 레코드를 MMQA 형태 JSONL 슬라이스로 변환.

목적:
  - ``extraction.py``, ``construct.py``, ``inference.py`` 가 공통으로 ``qid`` + jsonl 만 보도록
    외부 WebQA JSON 구조 의존을 이 스크립트 한 곳에 가둠.

동작:
  - 프로파일(``WEBQA_RUN_PROFILE``)과 원본 JSON 경로에 따라 test/val 풀을 고르고,
    각 레코드를 ``webqa_questions.jsonl`` 한 줄(메타에 text/image id, 더미 테이블 id) +
    대응 텍스트/이미지 행으로 전개.
  - WebQA 전용 테이블 본문은 없으므로 ``webqa_tables.jsonl`` 은 플레이스홀더 1행.

실행 예: ``python export_webqa_slice.py``

환경 변수: WEBQA_RUN_PROFILE, WEBQA_JSON_FILE, MMGRAPHRAG_RUN_ID, WEBQA_EXPORT_MAX 등
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from util.webqa_load import (
    default_test_path,
    default_train_val_path,
    is_test_split_json,
    load_webqa_dict,
    resolve_profile,
)
from util.webqa_gold_normalize import normalize_webqa_answer_strings
from util.result_layout import webqa_stamped_run_id


def _fact_text(fact: dict | str) -> str:
    """txt/img 팩트 dict 또는 문자열에서 슬라이스용 짧은 본문 텍스트로 압축."""
    if isinstance(fact, str):
        return fact.strip()
    parts: list[str] = []
    for key in ("title", "caption", "text", "snippet", "url"):
        v = fact.get(key)
        if v:
            parts.append(str(v).strip())
    body = " ".join(parts).strip()
    if body:
        return body[:8000]
    return json.dumps(fact, ensure_ascii=False)[:8000]


def _dummy_table() -> dict:
    """그래프 파이프라인이 table_id 를 기대하므로 넣는 고정 더미 테이블 1개."""
    return {
        "id": "webqa_dummy_table",
        "title": "WebQA placeholder",
        "table": {
            "table_name": "dummy",
            "header": [{"column_name": "note"}],
            "table_rows": [[{"text": "WebQA has no MMQA-style tables in this pipeline."}]],
        },
    }


def _build_rows_for_record(rec: dict, test_mode: bool) -> tuple[dict, list[dict], list[dict]]:
    """단일 WebQA 레코드 → (질문 행, 텍스트 행 목록, 이미지 행 목록).

    test_mode: test 스플릿(txt_Facts/img_Facts) vs 학습용 pos/neg 팩트 선택.
    """
    guid = str(rec["Guid"])
    q = str(rec.get("Q", ""))
    text_rows: list[dict] = []
    image_rows: list[dict] = []
    text_ids: list[str] = []
    image_ids: list[str] = []

    if test_mode:
        txt_facts = rec.get("txt_Facts") or []
        img_facts = rec.get("img_Facts") or []
    else:
        txt_facts = (rec.get("txt_posFacts") or []) + (rec.get("txt_negFacts") or [])
        img_facts = (rec.get("img_posFacts") or []) + (rec.get("img_negFacts") or [])

    for i, fact in enumerate(txt_facts):
        if isinstance(fact, dict):
            tid = str(fact.get("id") or f"{guid}_txt_{i}")
        else:
            tid = f"{guid}_txt_{i}"
        text_ids.append(tid)
        text_rows.append({"id": tid, "text": _fact_text(fact) if isinstance(fact, dict) else str(fact)})

    for i, fact in enumerate(img_facts):
        if not isinstance(fact, dict):
            continue
        iid = str(fact.get("image_id") or fact.get("id") or f"{guid}_img_{i}")
        image_ids.append(iid)
        caption = " ".join(
            str(fact.get(k) or "")
            for k in ("title", "caption")
        ).strip()
        image_rows.append(
            {
                "id": iid,
                "title": str(fact.get("title", "")),
                "path": str(fact.get("path", "")),
                "caption": caption,
            }
        )

    if not text_ids:
        tid = f"{guid}_no_text"
        text_ids.append(tid)
        text_rows.append({"id": tid, "text": q[:2000]})

    gold_parts = normalize_webqa_answer_strings(rec.get("A"))
    multimodal = (
        bool(rec.get("img_posFacts"))
        if not test_mode
        else (len(rec.get("img_Facts") or []) > 0)
    )
    meta = {
        "type": "WebQA",
        "text_doc_ids": text_ids[:8],
        "image_doc_ids": image_ids[:8],
        "table_id": "webqa_dummy_table",
        "webqa": {
            "Guid": guid,
            "split": rec.get("split", ""),
            "Qcate": rec.get("Qcate", ""),
            "multimodal": multimodal,
            "webqa_stratum": "Multimodal" if multimodal else "Unimodal",
        },
    }
    question = {
        "qid": guid,
        "question": q,
        "metadata": meta,
        "answers": [
            {"answer": part, "modality": "text"} for part in gold_parts
        ],
    }
    return question, text_rows, image_rows


def main() -> None:
    """프로파일별로 풀을 자른 뒤 ``webqa_slice`` 디렉터리에 4종 jsonl + meta 작성."""
    base = Path(__file__).resolve().parent
    run_id = os.getenv("MMGRAPHRAG_RUN_ID", "").strip() or webqa_stamped_run_id("export")
    profile = resolve_profile()
    json_path = os.getenv("WEBQA_JSON_FILE", "").strip()
    if not json_path:
        if profile == "test_full":
            json_path = str(default_test_path())
        else:
            json_path = str(default_train_val_path())

    test_mode = is_test_split_json(json_path)
    data = load_webqa_dict(json_path)
    pool = list(data.values())
    if test_mode:
        pool = [r for r in pool if r.get("split") == "test" or "txt_Facts" in r] or pool
    else:
        pool = [r for r in pool if r.get("split") == "val"]
    pool.sort(key=lambda r: str(r.get("Guid", "")))

    max_n = int(os.getenv("WEBQA_EXPORT_MAX", os.getenv("PATTERN_MAX_SAMPLES", "0")))
    if profile == "val_n100":
        cap = max_n if max_n > 0 else 100
        pool = pool[:cap]
    elif profile == "test_full":
        if max_n > 0:
            pool = pool[:max_n]

    out_dir = Path(os.getenv("WEBQA_SLICE_DIR", str(base / "result" / run_id / "webqa_slice")))
    out_dir.mkdir(parents=True, exist_ok=True)

    questions: list[dict] = []
    texts_map: dict[str, dict] = {}
    images_map: dict[str, dict] = {}

    for rec in pool:
        qrow, trows, irows = _build_rows_for_record(rec, test_mode)
        questions.append(qrow)
        for tr in trows:
            texts_map[str(tr["id"])] = tr
        for ir in irows:
            images_map[str(ir["id"])] = ir

    q_path = out_dir / "webqa_questions.jsonl"
    t_path = out_dir / "webqa_texts.jsonl"
    i_path = out_dir / "webqa_images.jsonl"
    tab_path = out_dir / "webqa_tables.jsonl"
    meta_path = out_dir / "webqa_export_meta.json"

    with q_path.open("w", encoding="utf-8") as fq:
        for row in questions:
            fq.write(json.dumps(row, ensure_ascii=False) + "\n")
    with t_path.open("w", encoding="utf-8") as ft:
        for row in texts_map.values():
            ft.write(json.dumps(row, ensure_ascii=False) + "\n")
    with i_path.open("w", encoding="utf-8") as fi:
        for row in images_map.values():
            fi.write(json.dumps(row, ensure_ascii=False) + "\n")
    with tab_path.open("w", encoding="utf-8") as ftab:
        ftab.write(json.dumps(_dummy_table(), ensure_ascii=False) + "\n")

    empty_gold = sum(
        1
        for q in questions
        if not (q.get("answers") and str(q["answers"][0].get("answer", "")).strip())
    )
    meta = {
        "profile": profile,
        "source_json": str(Path(json_path).resolve()),
        "n_questions": len(questions),
        "empty_gold_after_normalize": empty_gold,
        "out_dir": str(out_dir.resolve()),
        "note": "If empty_gold_after_normalize == n_questions (typical for public WebQA test), QA EM/F1 vs A is undefined.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(questions)} questions to {q_path}")
    print(f"Texts: {len(texts_map)}  Images: {len(images_map)}  -> {out_dir}")
    print(f"Empty gold strings (after normalize): {empty_gold} / {len(questions)}")


if __name__ == "__main__":
    from pathlib import Path

    from util.pipeline_session_log import run_with_session_stdio_tee

    run_with_session_stdio_tee(Path(__file__).resolve().parent, "export_webqa_slice", main)
