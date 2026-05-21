# ColGraphRAG — Query-Driven Multimodal GraphRAG

A query-driven pipeline that builds **per-question NetworkX graphs (GraphML)**,
reranks multimodal candidates with **ColEmbed late interaction (MaxSim)**,
and generates answers with an **in-process HF Gemma 4 E4B IT** (or **`--dry-run`**
without GPU).

**Paper:** [ColGraphRAG: Late-Interaction Evidence Retrieval for Multimodal GraphRAG](asset/colgraphrag.pdf)

The implementation follows **Query-Driven Multimodal GraphRAG** (Bu et al., ACL
2025 Findings). It supports **WebQA** and **MultiModalQA (MMQA)** corpora via
`--dataset webqa` / `--dataset mmqa`.

---

## Notebook tutorials

Step-by-step walkthroughs (venv + CUDA, `tests/test_pipeline.py`, manual phase
cells, optional demo). Run Jupyter from the **repository root** (so `REPO`
auto-detection finds `inference.py`), or open an `.ipynb` in your editor and
select the project **`.venv`** kernel.

| Notebook | Corpus | Language |
|----------|--------|----------|
| `notebook/colgraphrag_webqa_pipeline_tutorial_eng.ipynb` | WebQA | English |
| `notebook/colgraphrag_webqa_pipeline_tutorial_kor.ipynb` | WebQA | Korean |
| `notebook/colgraphrag_mmqa_pipeline_tutorial_eng.ipynb` | MMQA | English |
| `notebook/colgraphrag_mmqa_pipeline_tutorial_kor.ipynb` | MMQA | Korean |

The `notebook/` directory may also contain **date-stamped** copies of the MMQA
tutorial (e.g. for a specific workshop run); treat **`colgraphrag_*_tutorial_*.ipynb`**
as the canonical entry points.

---

## Pipeline overview

Phases **2–6** are shared. Phase **0** picks a corpus-specific export:

| Phase | WebQA | MMQA |
|-------|-------|------|
| 0 | `export_webqa_slice.py` → `webqa_slice/` | `export_mmqa_slice.py` → `mmqa_slice/` |
| 2 | `pattern.py` — per-question graph pattern (LLM) | same (`MMGRAPHRAG_DATASET=mmqa`) |
| 3 | `extraction.py` — entity/relation extraction (LLM) | same |
| 4 | `construct.py` — NetworkX → GraphML | same |
| 5 | `inference.py` — graph context + ColEmbed MaxSim + LLM answer | same |
| 6 | `eval/evaluate_webqa_qa.py` | `eval/evaluate_multimodal_qa.py` (list EM/F1; can stratify on `metadata.modalities`) |

Environment **`MMGRAPHRAG_DATASET`** is set to **`webqa`** or **`mmqa`** by
`tests/test_pipeline.py` and the notebooks. Pattern cache files are **query-driven
schema hints**, not gold labels: one JSON per question under `phase2_pattern_cache/{qid}.json`,
consumed by extraction as a structural constraint.

```
export_*_slice.py          Phase 0  JSONL slice (webqa_slice/ or mmqa_slice/)
        |
   pattern.py             Phase 2  Per-question graph pattern (LLM)
        |
  extraction.py            Phase 3  Entity/relation extraction (LLM)
        |
   construct.py            Phase 4  NetworkX GraphML construction
        |
   inference.py            Phase 5  graph_to_str + ColEmbed MaxSim + LLM
        |
eval/evaluate_*_qa.py      Phase 6  WebQA or MMQA metrics
```

---

## Prerequisites

Reference stack for **Linux GPU servers** (e.g. RunPod) and the **PyTorch `cu124`**
wheels pinned in `requirements.txt`. This is **not** the only valid setup:
`--dry-run` smoke tests and single-model phases need far less VRAM.

| Item | Details |
|------|---------|
| **OS** | **Linux x86_64** (recommended). Best match for published PyTorch CUDA wheels and typical cloud GPU images. |
| **Python** | **3.10+** (`requirements.txt`); **3.11** is recommended for notebooks / CI parity. |
| **Virtual env** | `python3.11 -m venv .venv` → `source .venv/bin/activate` → `pip install -r requirements.txt` |
| **GPU** | NVIDIA GPU. **A6000 48GB-class** is a practical target for development / PoC; **A100 / H100-class** helps throughput on full runs. |
| **NVIDIA Driver** | Must support **CUDA 12.x** for the shipped PyTorch **`+cu124`** wheels. `nvidia-smi` may show a newer CUDA version (e.g. 12.6–12.8); that is usually still compatible with the cu124 wheel. NVIDIA documents packaged Linux drivers for CUDA 12.4 from **550.54** onward; for CUDA 12.x minor compatibility, **525.60.13+** is cited as a lower bound — in practice aim for **driver 550+** to reduce surprises. See the [CUDA 12.4 Toolkit release notes (PDF)](https://docs.nvidia.com/cuda/archive/12.4.1/pdf/CUDA_Toolkit_Release_Notes.pdf). |
| **CUDA runtime** | **No full system CUDA toolkit** is required for the default setup: `pip` installs the PyTorch CUDA wheel, which pulls the needed runtime libraries. If you **compile** custom CUDA extensions, you may still need a toolkit / `nvcc`. |
| **PyTorch** | Pinned: **torch 2.6.0+cu124**, **torchvision 0.21.0+cu124**, **torchaudio 2.6.0+cu124** with `--index-url https://download.pytorch.org/whl/cu124`. See also [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/). |
| **VRAM** | **At least ~24 GB** recommended if **ColEmbed 3B** and **Gemma 4 E4B IT** appear in the same heavy phase (KV cache, activations, fragmentation). **48 GB+** is recommended for comfortable end-to-end work; treat **~16 GB** only as a **smoke-test** or strictly **sequential** loading lower bound, not a stable dual-model budget. |
| **RAM / vCPU** | **~50 GB+ RAM** and **9+ vCPU** are reasonable when JSONL I/O, caches, GraphML, evaluation, and the optional FastAPI demo all run on the same machine — CPU RAM can become a bottleneck too. |
| **Hardware** | Throughput favors **A100 / H100**; **A6000 48 GB** remains a good fit for development, demos, and toy slices. |
| **Model weights** | **Not included in the repo.** Download **Gemma**, **ColEmbed**, and (for WebQA **QA-FL**) **BART** under `models/` or your network volume — see [Models](#models) before non-`--dry-run` runs. |

> **One-line summary:** Use **Linux x86_64**, a **Python 3.11** venv, and **PyTorch 2.6.0 + cu124** from the pinned index. Prefer a **CUDA 12.x–capable NVIDIA driver (550+)** rather than installing a full CUDA toolkit; rely on the PyTorch wheel for runtime libraries. For **ColEmbed + Gemma** in realistic workloads, plan **24 GB+ VRAM** and treat **48 GB-class** GPUs as the comfortable default.

### Install

From the **repository root**:

```bash
python3.11 -m venv .venv       # or: python3 -m venv .venv  (Python 3.10+)
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the PyTorch **cu124** wheel index so CUDA builds resolve correctly.

**Model weights are separate:** **`pip install -r requirements.txt` does not download Gemma, ColEmbed, or BART.** For any real (non-`--dry-run`) run you must pull those checkpoints first — see **[Models](#models)** (`util/download_models.py`).

---

## Datasets

### WebQA

| Item | Description |
|------|-------------|
| **Paper** | [WebQA: Multihop and Multimodal QA (Chang et al., 2021)](https://arxiv.org/abs/2109.00590) |
| **Site** | <https://webqna.github.io/> |
| **Download** | [WebQA GitHub — Data](https://github.com/WebQnA/WebQA#data-download) |
| **Metrics** | **QA-FL** (BARTScore fluency), **QA-Acc** (keyword overlap by Qcate), **QA** (FL × Acc) |

Typical layout (you can also keep data **under the repo** as `data/webqa/`;

see `config/data.yaml` and `data/README.md`):

```
/path/to/webqa/
    WebQA_data_first_release/
        WebQA_train_val.json
        imgs/               # or per-shard layouts used by your export script
```

#### Toy dataset (shard 14)

For quick tests without the full corpus:

```bash
python scripts/build_shard14_toy_split.py
```

Default toy slice: `data/webqa/webqa_shard14_toy/webqa_slice/` (or under
`WebQA_imgs_7z_chunks/` depending on layout — see `data/README.md`).

### MultiModalQA (MMQA)

| Item | Description |
|------|-------------|
| **Paper** | [MultiModalQA: Complex Question Answering over Text, Tables and Images](https://arxiv.org/abs/2104.06039) (Talmor et al., ICLR 2021) |
| **Data & format** | Official release and file layout: [allenai/multimodalqa](https://github.com/allenai/multimodalqa) (`dataset/` question & context JSONL, `images.zip`, etc.) |
| **Pipeline** | `export_mmqa_slice.py` reads `MMQA_<split>_n*.jsonl`, `MMQA_texts_n*.jsonl`, etc. under **`MMQA_DATA_DIR`** (default: `data/multimodalqa/dataset`) and writes **`mmqa_questions.jsonl`**, **`mmqa_texts.jsonl`**, **`mmqa_images.jsonl`**, **`mmqa_tables.jsonl`** into **`mmqa_slice/`** for the current run. |
| **Images** | Usually `data/multimodalqa/final_dataset_images/` (paths referenced from MMQA JSONL). |

A small **dev toy bundle** (e.g. `MMQA_*_n100.jsonl`) may ship under
`data/multimodalqa/dataset/` so you can run `--dataset mmqa` without a full
MMQA download. For the **full** corpus, follow **[allenai/multimodalqa](https://github.com/allenai/multimodalqa)** and place shards (and image files) under `data/multimodalqa/` (or point **`MMQA_DATA_DIR`** at your checkout).

---

## Models

**Large checkpoints are not included in this repository.** You must download them
(e.g. **`python util/download_models.py`**) before running pattern,
extraction, inference, or ColEmbed retrieval on GPU. Only **`--dry-run`**
pipelines skip loading those weights (they still execute the rest of the wiring).

Hub ids and default target directories are in **`config/model.yaml`**. Recommended
layout: **`models/`** at the repo root (`models/mllm/…`, `models/retriever/…`,
`models/eval/…`).

### Batch download (`util/download_models.py`)

Reads `config/model.yaml` and pulls snapshots with **`huggingface_hub.snapshot_download`**
(resume-friendly). For gated Gemma, put **`HF_TOKEN`** (or **`HUGGING_FACE_HUB_TOKEN`**)
and optional **`HF_USERNAME`** in repo-root **`.env`** (see **`.env.example`**).

```bash
pip install huggingface_hub python-dotenv pyyaml   # or rely on requirements.txt
python util/download_models.py
python util/download_models.py --only gemma colembed
python util/download_models.py --dry-run
```

| Component | Hub id | Default local dir |
|-----------|--------|-------------------|
| Gemma | `google/gemma-4-E4B-it` | `models/mllm/gemma-4-E4B-it` |
| ColEmbed | `nvidia/llama-nemotron-colembed-vl-3b-v2` | `models/retriever/llama-nemotron-colembed-vl-3b-v2` |
| Fluency (WebQA QA-FL) | `facebook/bart-large-cnn` | `models/eval/bart-large-cnn` |

Optional: **`python scripts/preload_fluency_model.py`** to pre-cache BART.

### ColEmbed VL (image / multimodal late interaction)

| Item | Description |
|------|-------------|
| **Model** | `nvidia/llama-nemotron-colembed-vl-3b-v2` |
| **Paper** | [Nemotron ColEmbed V2 (arXiv:2602.03992)](https://arxiv.org/abs/2602.03992) |

Set **`COLEMBED_MODEL_PATH`** to the downloaded directory (default under `models/retriever/`).

### Gemma 4 E4B IT (pattern, extraction, answer)

Used in-process when **`VIDORE_TEXT_LLM_BACKEND=hf_gemma_4_e4b_it`** (default in
tests/notebooks). Set **`GEMMA4_E4B_IT_MODEL_PATH`** to the snapshot directory
(default: **`models/mllm/gemma-4-E4B-it`**).

---

## Quick start

Run everything from the **repository root** with the venv activated. For
**real LLM + retrieval** runs, finish **[Models](#models)** (Gemma + ColEmbed;
BART optional for WebQA QA-FL) first, or use **`--dry-run`** to validate the
pipeline without those weights.

### WebQA (default: toy slice when available)

Omitting `-n` uses the script default for query count (see `tests/test_pipeline.py --help`).
Use a larger **`-n`** if you want more questions from the slice.

```bash
python tests/test_pipeline.py -n 5
```

### MMQA

```bash
python tests/test_pipeline.py --dataset mmqa -n 5
```

### Dry-run (no GPU LLM; quick wiring check)

```bash
python tests/test_pipeline.py --dry-run -n 5
python tests/test_pipeline.py --dataset mmqa --dry-run -n 5
```

`tests/test_pipeline.py` uses **HF Gemma in-process** for real runs or synthetic
outputs when **`PATTERN_DRY_RUN` / `EXTRACTION_DRY_RUN`** are set via **`--dry-run`**.

### WebQA export (no toy)

```bash
python tests/test_pipeline.py --no-toy -n 5
```

### Many queries (helper script)

```bash
python scripts/run_full_pipeline.py --help
```

### Demo: latest run

After `tests/test_pipeline.py`, the **demo** backend can resolve **`run_id: "latest"`**
in `demo/be/config/paths.yaml` (restart BE to pick the newest `result/` run). See
**`demo/README.md`**.

---

## Environment variables

The repo loads **`.env`** when present (`python-dotenv`). Copy **`.env.example`**
→ **`.env`** locally (do not commit secrets).

| Variable | Role |
|----------|------|
| **`MMGRAPHRAG_DATASET`** | `webqa` or `mmqa` (set by `test_pipeline.py` / notebooks). |
| **`MMGRAPHRAG_RUN_ID`** | Run folder id (e.g. `webqa/<stamp>_…`, `multimodalqa/<stamp>_…`); stamped automatically if unset in drivers. |
| **`PIPELINE_WEBQA_DATA_DIR`** / **`WEBQA_DATA_ROOT`** | WebQA tree (default: `data/webqa` under repo). |
| **`MMQA_DATA_DIR`** | Directory with `MMQA_*_n*.jsonl` shards for export (default: `data/multimodalqa/dataset`). |
| **`HF_TOKEN`**, **`HUGGING_FACE_HUB_TOKEN`**, **`HF_USERNAME`** | Hugging Face Hub auth. |
| **`GEMMA4_E4B_IT_MODEL_PATH`** | Gemma 4 E4B IT weights directory. |
| **`COLEMBED_MODEL_PATH`** | ColEmbed checkpoint directory. |
| **`VIDORE_TEXT_LLM_BACKEND`** | Default **`hf_gemma_4_e4b_it`** for in-process Gemma. |
| **`PATTERN_DRY_RUN`**, **`EXTRACTION_DRY_RUN`** | `1` for LLM-free pattern/extraction stubs (see `--dry-run`). |
| **`INFERENCE_DRY_RUN`**, **`INFERENCE_COLEMBED_RETRIEVAL`**, **`INFERENCE_COLEMBED_TOP_K`** | Inference behaviour (see `inference.py` / env docs in code). |

Phase-specific paths (**`PATTERN_JSON_FILE_PATH`**, **`PATTERN_CACHE_DIR`**, etc.)
are normally set by `tests/test_pipeline.py` or the notebooks.

---

## Directory structure (repository root)

```
├── pattern.py / extraction.py / construct.py / inference.py
├── prompt.py
├── export_webqa_slice.py
├── export_mmqa_slice.py
├── requirements.txt
├── config/
│   ├── data.yaml          # WebQA + multimodalqa paths
│   └── model.yaml         # Gemma, ColEmbed, fluency
├── util/
├── eval/
│   ├── evaluate_webqa_qa.py
│   ├── evaluate_multimodal_qa.py
│   └── …
├── scripts/
├── tests/
│   └── test_pipeline.py   # End-to-end: --dataset webqa | mmqa
├── notebook/              # tutorials (see table above)
├── demo/                  # optional Web UI (see demo/README.md)
├── data/                  # local corpora (webqa, multimodalqa, …)
└── result/                # runs (auto-created)
```

---

## Demo (Web UI)

Optional **FastAPI + Vite** app to browse **`predictions.json`**, graphs, and chat.
See **`demo/README.md`** for ports, **`paths.yaml`**, and MMQA vs WebQA path
resolution.

---

## Output (`result/`)

Runs are rooted at **`result/<dataset-namespace>/<run-id>/`** (e.g.
`result/webqa/…`, `result/multimodalqa/…`).

| Path | Contents |
|------|----------|
| **`webqa_slice/`** or **`mmqa_slice/`** | Normalized JSONL slice for the run (`*_questions.jsonl`, texts, images, tables as applicable). |
| **`phase2_pattern_cache/`** | One `*.json` per question (LLM graph pattern). |
| **`phase3_extraction_cache/`** | Extraction JSON per question. |
| **`phase4_graphs_out/`** | `{qid}_graph.graphml`. |
| **`phase5_inference/predictions.json`** | Model answers. |
| **`phase5_inference/evaluation_report.json`** | Eval summary when Phase 6 runs. |

---

## References

- Bu et al., *Query-Driven Multimodal GraphRAG: Dynamic Local Knowledge Graph
  Construction for Online Reasoning* (Findings of ACL, 2025).
  Anthology: <https://aclanthology.org/2025.findings-acl.1100/> ·
  Related code: <https://github.com/DMiC-Lab-HFUT/Query-Driven-Multimodal-GraphRAG>
- Talmor et al., *MultiModalQA: Complex Question Answering over Text, Tables and Images*,
  ICLR 2021. [arXiv:2104.06039](https://arxiv.org/abs/2104.06039) — data:
  [allenai/multimodalqa](https://github.com/allenai/multimodalqa)
- Chang et al., *WebQA: Multihop and Multimodal QA*, arXiv:2109.00590, 2021.
  <https://webqna.github.io/>
- Moreira et al., *Nemotron ColEmbed V2*, arXiv:2602.03992, 2026.
  <https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2>
