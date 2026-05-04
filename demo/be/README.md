# ColGraphRAG WebQA Demo Backend

FastAPI server that serves **pipeline outputs** under **`result/<run_id>/`** (questions, graphs, predictions, scores) plus an optional **`/api/chat`** path that runs **local Gemma** over graph context — no external hosted search APIs.

## Quick Start

Use the **repository root `.venv`** (same stack as `requirements.txt`), or install only BE deps:

```bash
cd demo/be
pip install -r requirements.txt
python server.py
```

Defaults: **`http://127.0.0.1:8000`**.

RunPod / LAN (listen on all interfaces):

```bash
python server.py --host 0.0.0.0 --port 8000
```

CLI flags: **`--host`**, **`--port`**, **`--reload`** (dev).

- **Swagger:** http://127.0.0.1:8000/docs  
- **Health:** http://127.0.0.1:8000/health  

## Logging

Each process start writes a session file under **`logs/`**:

- Filename pattern: **`YYYYMMDD_HHMMSS.log`** (local time when `server.py` / `uvicorn server:app` loads)
- **`uvicorn.access`**, **`uvicorn.error`**, and **`INFO`** logs from the **`server`** logger (including resolved **`paths.*`** from `paths.yaml`) go to **file + stderr**

See **`services/logging_setup.py`**.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status + `run_id` |
| `/api/run/info` | GET | Run metadata (run_id, counts, paths) |
| `/api/run/scores` | GET | QA-FL / QA-Acc / QA + Qcate breakdown |
| `/api/questions` | GET | Question list with predictions |
| `/api/questions/{qid}` | GET | Question detail (gold, prediction, retrieval) |
| `/api/graphs/{qid}` | GET | Graph nodes/edges for visualization |
| `/api/graphs/{qid}/graphml` | GET | Download raw GraphML |
| `/api/images/{image_id}` | GET | Serve PNG image |
| `/api/images` | GET | List available image IDs |
| `/api/chat` | POST | Live-style answer (JSON body `{ "question": "..." }`); uses local Gemma + graph context |

## Configuration

Primary file: **`config/paths.yaml`**. Copy from **`config/paths.example.yaml`** when needed (root **`.gitignore`** may omit a local **`paths.yaml`**).

| Area | YAML keys | Notes |
|------|-----------|--------|
| Repo root | `repo_root` | Empty string ⇒ auto-resolve to clone root (**`demo/be/config/ →` parents[3]**). Override if layout differs. |
| Run selection | `run.run_id` | **`"latest"`** → subdirectory of **`<repo>/result/`** with newest mtime. Otherwise exact **`MMGRAPHRAG_RUN_ID`** folder name. |
| Run override | `run.explicit_result_dir` | Optional path (**repo-relative or absolute**) to force a **`result`** run directory. |
| WebQA fixtures | `webqa.shard14_slice`, `webqa.imgs_dir` | Default toy slice + shard-14 images under **`data/webqa/`** |
| Model dirs | `models.*` | Used by chat / loaders (paths under **`models/`** repo-local) |

Subpaths inside a run (e.g. **`webqa_slice`**, **`phase4_graphs_out`**) live under **`result_layout`**; see **`config/resolve_demo_paths.py`**.

**Refresh “latest”:** restart the backend after a new **`result/<id>/`** is written (selection is done at lifespan startup).

## Tests

From **`demo/be/`**:

```bash
pytest tests/ -v
```

## Directory layout

```
demo/be/
  server.py                  # Entry; wires logging then FastAPI lifespan
  requirements.txt
  logs/                     # Session logs (*.gitignored via root *.log)
  config/
    paths.yaml              # Active config (copy from paths.example.yaml)
    paths.example.yaml
    resolve_demo_paths.py    # Resolve repo + latest run + model paths
  routers/                  # FastAPI routers (questions, graphs, chat, ...)
  schemas/                   # Pydantic models
  services/                  # loaders, graph/image/run + llm helpers
    logging_setup.py         # logs/YYYYMMDD_HHMMSS.log
  tests/
```

See also repo **`demo/README.md`** and **`demo/fe/`** (Vite proxies **`/api`** to this backend).
