# ColGraphRAG WebQA -- Query-Driven Multimodal GraphRAG for WebQA

A query-driven pipeline that builds **per-question NetworkX graphs (GraphML)**,
reranks image candidates with **ColEmbed late interaction (MaxSim)**,
and generates final answers with an **LLM (Gemma, etc.)**.

> This repository is a **clean version** of the research original (`colgraphrag_webqa`)
> with unused legacy code (Chroma, BGE-M3, async branches) removed.

---

## Notebook tutorial

Step-by-step walkthrough (env setup, one-shot `test_pipeline.py`, optional per-phase cells):

- `notebook/01_colgraphrag_webqa_pipeline_tutorial.ipynb`

Start Jupyter from the repo root, or open in VS Code / Cursor.

## Pipeline Overview

```
export_webqa_slice.py   Phase 0  JSONL corpus slice
        |
    pattern.py          Phase 2  Per-question graph pattern (LLM)
        |
   extraction.py        Phase 3  Entity/relation extraction (LLM)
        |
   construct.py         Phase 4  NetworkX GraphML construction
        |
   inference.py         Phase 5  graph_to_str + LLM answer generation
        |                         + ColEmbed MaxSim retrieval
        |
eval/evaluate_webqa_qa.py  Phase 6  QA-FL / QA-Acc / QA evaluation
```

---

## Requirements

- Python 3.10+
- CUDA 12.4 + NVIDIA GPU (A100 / H100 recommended)
- ~16 GB VRAM (when loading ColEmbed 3B + Gemma 4 E4B IT simultaneously)

```bash
pip install -r requirements.txt
```

---

## Datasets

### WebQA

| Item | Description |
|------|-------------|
| **Paper** | [WebQA: Multihop and Multimodal QA (Chang et al., 2021)](https://arxiv.org/abs/2109.00590) |
| **Homepage** | <https://webqna.github.io/> |
| **Data Download** | [WebQA GitHub -- Data](https://github.com/WebQnA/WebQA#data-download) |
| **Composition** | Questions + text snippets + images + tables (multimodal multi-hop QA) |
| **Metrics** | **QA-FL** (BARTScore fluency), **QA-Acc** (keyword overlap by Qcate), **QA** (FL x Acc) |

After download, place in the following path:

```
/workspace/data/webqa/
    WebQA_data_first_release/
        WebQA_train_val.json
        imgs/               # Images (or per-shard splits)
```

#### Toy Dataset (shard_00014)

For quick testing without full data:

```bash
python scripts/build_shard14_toy_split.py
```

Output location: `/workspace/data/webqa/WebQA_imgs_7z_chunks/webqa_shard14_toy/webqa_slice/`
(66 VAL questions, 60 images)

---

## Models

### ColEmbed VL (Image Retrieval)

| Item | Description |
|------|-------------|
| **Model** | `nvidia/llama-nemotron-colembed-vl-3b-v2` |
| **HuggingFace** | <https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2> |
| **Paper** | [Nemotron ColEmbed V2 (arXiv:2602.03992)](https://arxiv.org/abs/2602.03992) |
| **Architecture** | SigLIP2-Giant + Llama-3.2-3B, ~4.4B params |
| **Method** | Late interaction (ColBERT-style MaxSim) -- query-image token interaction |
| **License** | NVIDIA Non-Commercial License (research use) |

Download:

```bash
# HuggingFace CLI
huggingface-cli download nvidia/llama-nemotron-colembed-vl-3b-v2 \
    --local-dir /workspace/models/retriever/llama-nemotron-colembed-vl-3b-v2

# Or Python
from huggingface_hub import snapshot_download
snapshot_download(
    "nvidia/llama-nemotron-colembed-vl-3b-v2",
    local_dir="/workspace/models/retriever/llama-nemotron-colembed-vl-3b-v2",
)
```

Environment variable: `COLEMBED_MODEL_PATH=/workspace/models/retriever/llama-nemotron-colembed-vl-3b-v2`

### Gemma 4 E4B IT (Answer Generation LLM)

| Item | Description |
|------|-------------|
| **Model** | `google/gemma-4-e4b-it` |
| **HuggingFace** | <https://huggingface.co/google/gemma-4-e4b-it> |
| **Usage** | Phase 2 pattern, Phase 3 extraction, Phase 5 answer generation (in-process) |

Download:

```bash
huggingface-cli download google/gemma-4-e4b-it \
    --local-dir /workspace/models/mllm/gemma-4-e4b-it
```

Environment variable: `GEMMA4_E4B_IT_MODEL_PATH=/workspace/models/mllm/gemma-4-e4b-it`

### BART (Fluency Evaluation)

| Item | Description |
|------|-------------|
| **Model** | `facebook/bart-large-cnn` |
| **Usage** | QA-FL (BARTScore) calculation |

Auto-downloaded, but to pre-cache:

```bash
python scripts/preload_fluency_model.py
```

---

## Quick Start

### Real inference (default: GPU + Gemma + ColEmbed + toy dataset)

Omitting `-n` uses **`--n-queries` default = 5** (not all VAL questions; use `-n 66` on the toy slice if you want every question in that slice).

```bash
cd /workspace/mmgraphrag/lecture/code/colgraphrag_webqa
python tests/test_pipeline.py -n 5
```

### Run 10 queries (for Demo testing)

```bash
cd /workspace/mmgraphrag/lecture/code/colgraphrag_webqa
python tests/test_pipeline.py -n 10
```

> **Note:** A new run directory is created under `result/`. When you restart the Demo BE,
> it auto-selects the **most recent run** based on the `run_id: "latest"` setting in `paths.yaml`.
> Verify that the left-side badge shows "10 answered / 66 total".

### Dry-run (no GPU required, quick pipeline verification)

```bash
cd /workspace/mmgraphrag/lecture/code/colgraphrag_webqa
python tests/test_pipeline.py --dry-run -n 5
```

`tests/test_pipeline.py` in this clean package only runs **Hugging Face Gemma in-process** (default) or **`--dry-run`**. There is no `--http-llm` / Ollama integration on this entrypoint.

### Run all 66 toy queries

```bash
cd /workspace/mmgraphrag/lecture/code/colgraphrag_webqa
python scripts/run_full_pipeline.py           # all 66 (default)
python scripts/run_full_pipeline.py -n 20     # or specify count
```

### Full WebQA export (instead of toy dataset)

```bash
python tests/test_pipeline.py --no-toy -n 5
```

---

## Environment Variables

Repository root loads `.env` when present (`python-dotenv` in `util/request.py`). Copy **`.env.example`** → **`.env`** and set tokens locally (`.env` must not be committed).

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | _(unset)_ | Hugging Face Hub token (`huggingface_hub` / CLI downloads). Same as `HUGGING_FACE_HUB_TOKEN`. |
| `HF_USERNAME` | _(unset)_ | Optional Hub username (auth is token-based). |
| `MMGRAPHRAG_RUN_ID` | auto timestamp | Result directory name |
| `GEMMA4_E4B_IT_MODEL_PATH` | `/workspace/models/mllm/gemma-4-e4b-it` | Gemma checkpoint |
| `COLEMBED_MODEL_PATH` | `/workspace/models/retriever/llama-nemotron-colembed-vl-3b-v2` | ColEmbed checkpoint |
| `VIDORE_TEXT_LLM_BACKEND` | `hf_gemma_4_e4b_it` | Text LLM backend (HF Gemma in-process) |
| `INFERENCE_DRY_RUN` | `0` | If `1`, graph heuristic without LLM |
| `INFERENCE_COLEMBED_RETRIEVAL` | `1` | If `1`, ColEmbed MaxSim retrieval |
| `INFERENCE_COLEMBED_TOP_K` | `10` | MaxSim top k |
| `WEBQA_DATA_ROOT` | `/workspace/data/webqa` | WebQA data root |

---

## Directory Structure

```
colgraphrag_webqa/
    pattern.py              # Phase 2: Graph pattern generation
    extraction.py           # Phase 3: Entity/relation extraction
    construct.py            # Phase 4: GraphML construction
    inference.py            # Phase 5: Inference (ColEmbed + LLM)
    prompt.py               # LLM prompt constants
    export_webqa_slice.py   # JSONL slice generation
    requirements.txt
    config/
        data.yaml            # WebQA / data paths (see util/repo_config.py)
        model.yaml           # Gemma, ColEmbed, fluency defaults
    .env.example             # Secret template (HF, etc.) -> copy to .env
    util/
        request.py          # Utilities (dotenv, ColEmbed loader, etc.)
        webqa_metrics_approx.py   # QA-Acc approximation
        webqa_fluency.py    # BARTScore fluency
    eval/
        evaluate_webqa_qa.py      # QA-FL / QA-Acc / QA evaluation
        evaluate_retrieval.py     # Retrieval IR metrics
    scripts/
        run_full_pipeline.py      # Run all queries (default 66)
        preload_fluency_model.py  # Pre-cache BART for QA-FL
        build_shard14_toy_split.py  # Build toy dataset from shard_00014
    tests/
        test_pipeline.py    # End-to-end pipeline test (default 5 queries)
    result/                 # Experiment results (auto-generated)
```

---

## Demo (Web UI)

A **standalone demo** (BE + FE) for viewing pipeline results in the browser.

### Running

```bash
# 1) Backend (FastAPI, :8000)
cd demo/be
pip install -r requirements.txt
python server.py

# 2) Frontend (Vite, :5173) -- separate terminal
cd demo/fe
npm install && npm run dev
```

- **Browser:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs

### Auto-selection of latest run (`run_id: "latest"`)

In `demo/be/config/paths.yaml`:

```yaml
run:
  run_id: "latest"   # Auto-select folder with latest mtime under result/
```

- When you run the pipeline (`tests/test_pipeline.py -n 10`, etc.), a new run is created under `result/`
- **On BE restart**, the most recent run is selected (no auto-switch during runtime)
- Verify via the left-side badge **"N answered / 66 total"**

### Main Screens

| Tab | Description |
|-----|-------------|
| **Results** | Question list, predicted answers, gold answers, graph viewer |
| **Live Chat** | Real-time chat with graph context (Gemma LLM) |

More detail: `demo/README.md` (API, paths, RunPod notes).

---

## Output (result/)

Each run is created under `result/<MMGRAPHRAG_RUN_ID>/`:

| Path | Contents |
|------|----------|
| `webqa_slice/` | Input JSONL slice |
| `phase2_pattern_cache/` | Per-question pattern JSON |
| `phase3_extraction_cache/` | Extraction JSON |
| `phase4_graphs_out/` | `{qid}_graph.graphml` |
| `phase5_inference/predictions.json` | Answers |
| `phase5_inference/evaluation_report.json` | QA evaluation report |

---

## References

- Bu et al., *Query-Driven Multimodal GraphRAG: Dynamic Local Knowledge Graph Construction for Online Reasoning* (Findings of ACL 2025).
  Official implementation: <https://github.com/DMiC-Lab-HFUT/Query-Driven-Multimodal-GraphRAG>
- Chang et al., *WebQA: Multihop and Multimodal QA*, arXiv:2109.00590, 2021.
  <https://webqna.github.io/>
- Moreira et al., *Nemotron ColEmbed V2*, arXiv:2602.03992, 2026.
  <https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2>
