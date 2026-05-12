#!/usr/bin/env bash
# MultiModalQA: export -> pattern -> extraction -> construct -> inference -> QA eval.
# Uses literal env names MMGRAPHRAG_DATASET / MMGRAPHRAG_RUN_ID (must match *.py getenv strings).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"

N_QUESTIONS="${1:-5}"

# Avoid inherited typo keys or stale RUN_ID from the parent Cursor/shell env.
unset MMGRAPHRAG_RUN_ID MMGRAPHRAG_DATASET 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"

# Load Gemma in bfloat16 to halve GPU memory usage (~15 GB vs ~30 GB at fp32).
export GEMMA4_E4B_IT_TORCH_DTYPE="${GEMMA4_E4B_IT_TORCH_DTYPE:-bfloat16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export N_QUESTIONS="${N_QUESTIONS}"
# Stored under result/multimodalqa/YYYYMMDD_HHMMSS_<slug>/ (see util/result_layout.resolve_mmqa_bash_pipeline_run_id).
# Optional MMQA_RUN_ID_OVERRIDE slug; MMQA_ALLOW_LEGACY_UNSTAMPED_RUN_ID=1 keeps an unstamped path as-is.
run_id="$("$PY" -c "from util.result_layout import resolve_mmqa_bash_pipeline_run_id as r; print(r())")"

export MMGRAPHRAG_DATASET=mmqa
export MMGRAPHRAG_RUN_ID="${run_id}"

export PATTERN_MAX_SAMPLES="$N_QUESTIONS"
export EXTRACTION_MAX_QUESTIONS="$N_QUESTIONS"
export CONSTRUCT_MAX_QUESTIONS="$N_QUESTIONS"
export INFERENCE_MAX_QUESTIONS="$N_QUESTIONS"

# Force real LLM / ColEmbed paths (no dry-run).
export PATTERN_DRY_RUN=0
export EXTRACTION_DRY_RUN=0
export INFERENCE_DRY_RUN=0

echo "=== RUN_ID=${MMGRAPHRAG_RUN_ID} (n=${N_QUESTIONS}) ==="

"$PY" export_mmqa_slice.py
"$PY" pattern.py
"$PY" extraction.py
"$PY" construct.py
"$PY" inference.py

rid="${MMGRAPHRAG_RUN_ID}"
PHASE5_DIR="${ROOT}/result/${rid}/phase5_inference"
EVAL_OUT="${PHASE5_DIR}/evaluation_report.json"

echo "=== eval -> ${EVAL_OUT} ==="
"$PY" eval/evaluate_multimodal_qa.py \
  --predictions "${PHASE5_DIR}/predictions.json" \
  --gold_jsonl "${ROOT}/result/${rid}/mmqa_slice/mmqa_questions.jsonl" \
  --split_label "mmqa_dev_n${N_QUESTIONS}" \
  --retrieval_rankings_json "${PHASE5_DIR}/predictions_retrieval.json" \
  --report_json "${EVAL_OUT}" || echo "eval exited non-zero (see output)"

echo "DONE."
