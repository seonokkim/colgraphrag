# ColGraphRAG WebQA Demo

Chat-style demo for viewing WebQA GraphRAG pipeline results.

## Quick Start

### 1. Backend (Terminal 1)

```bash
cd demo/be
pip install -r requirements.txt
python server.py
```

Backend runs at http://127.0.0.1:8000

### 2. Frontend (Terminal 2)

```bash
cd demo/fe
npm install
npm run dev
```

Frontend runs at http://localhost:5173

### 3. Open Browser

Navigate to http://localhost:5173

## Architecture

```
demo/
  be/       # FastAPI backend (port 8000)
  fe/       # React frontend (port 5173)
```

## Features

- Chat-style Q&A interface
- Score cards (QA-FL / QA-Acc / QA)
- Question list with category filtering
- Gold vs Predicted answer comparison
- Retrieval ranking display
- Interactive knowledge graph visualization
- Evidence image viewer

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Server health |
| `/api/run/info` | Run metadata |
| `/api/run/scores` | QA scores |
| `/api/questions` | Question list |
| `/api/questions/{qid}` | Question detail |
| `/api/graphs/{qid}` | Graph data |
| `/api/images/{id}` | Serve image |

## Configuration

Edit `be/config/paths.yaml` to change:
- `run.run_id`: `"latest"` or specific run ID
- Data and model paths
