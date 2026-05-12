# ColGraphRAG Demo

Chat-style viewer for ColGraphRAG pipeline results. Supports **WebQA** and **MultimodalQA** datasets.

## Prerequisites

- Python 3.10+ with the repo-root virtualenv (`.venv/`) already set up
- Node.js 18+ (required for Vite 6)

## Quick Start

### Option A — convenience scripts (from repo root)

```bash
# Terminal 1: backend
bash demo/run-be.sh

# Terminal 2: frontend
bash demo/run-fe.sh
```

### Option B — manual

**Terminal 1 — Backend**

```bash
cd demo/be
source ../../.venv/bin/activate   # or: pip install -r requirements.txt
python server.py                  # 127.0.0.1:8000
```

**Terminal 2 — Frontend**

```bash
cd demo/fe
npm install
npm run dev                       # 0.0.0.0:5173
```

### Open in browser

```
http://localhost:5173
```

API docs (Swagger): http://localhost:8000/docs

## Architecture

```
demo/
  be/           # FastAPI backend  (port 8000)
  fe/           # React/Vite frontend  (port 5173)
  run-be.sh     # Start BE using repo-root .venv
  run-fe.sh     # Start FE dev server
```

Vite proxies `/api` and `/health` to `http://127.0.0.1:8000`, so the browser only talks to port 5173.

## Features

- Chat-style Q&A interface
- Score cards (QA-FL / QA-Acc / QA) with per-category breakdown
- Question list with category filtering
- Gold vs. predicted answer comparison
- Retrieval ranking display
- Interactive knowledge graph visualization
- Evidence image viewer
- WebQA and MultimodalQA dataset switching

## API Endpoints

All data endpoints accept an optional `?dataset=webqa|mmqa` query parameter (default: `webqa`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status + active run ID |
| `/api/datasets` | GET | Available datasets and their run IDs |
| `/api/run/info` | GET | Run metadata (counts, paths) |
| `/api/run/scores` | GET | QA-FL / QA-Acc / QA scores |
| `/api/questions` | GET | Question list with predictions |
| `/api/questions/{qid}` | GET | Question detail (gold, prediction, retrieval) |
| `/api/graphs/{qid}` | GET | Graph nodes/edges for visualization |
| `/api/graphs/{qid}/graphml` | GET | Download raw GraphML |
| `/api/images/{image_id}` | GET | Serve image file |
| `/api/images` | GET | List available image IDs |
| `/api/chat` | POST | Live answer via local Gemma + graph context |

## Configuration

Edit `be/config/paths.yaml` (copy from `be/config/paths.example.yaml` if missing).

| Key | Default | Description |
|-----|---------|-------------|
| `repo_root` | `""` (auto) | Override repo root path |
| `run.run_id` | `"latest"` | Specific run ID or `"latest"` (newest mtime under `result/webqa/`) |
| `run.explicit_result_dir` | `""` | Force an absolute result directory |
| `mmqa.run_id` | `"latest"` | MMQA run ID or `"latest"` (under `result/multimodalqa/`) |
| `models.gemma` | `models/mllm/gemma-4-E4B-it` | Chat model path |

After changing `run_id`, restart the backend — run selection happens at startup.

## Remote / Headless Server

To expose both servers on all interfaces (e.g. SSH port-forward or GPU server):

```bash
# Backend
python demo/be/server.py --host 0.0.0.0 --port 8000

# Frontend
cd demo/fe && npm run dev -- --host 0.0.0.0 --port 5173
```

Then port-forward from your local machine:

```bash
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 user@server
```

## Logs

Backend writes a per-session log to `demo/be/logs/YYYYMMDD_HHMMSS.log` on each startup.
