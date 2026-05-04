# ColGraphRAG WebQA Demo Backend

FastAPI server for viewing WebQA pipeline results. Read-only — no external search or LLM APIs needed.

## Quick Start

```bash
cd demo/be
pip install -r requirements.txt
python server.py
```

Default: http://127.0.0.1:8000

RunPod / LAN (expose port 8000):

```bash
python server.py --host 0.0.0.0 --port 8000
```

- API Docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status + run_id |
| `/api/run/info` | GET | Run metadata (run_id, counts, paths) |
| `/api/run/scores` | GET | QA-FL / QA-Acc / QA + Qcate breakdown |
| `/api/questions` | GET | Question list with predictions |
| `/api/questions/{qid}` | GET | Question detail (gold, prediction, retrieval) |
| `/api/graphs/{qid}` | GET | Graph nodes/edges for visualization |
| `/api/graphs/{qid}/graphml` | GET | Download raw GraphML file |
| `/api/images/{image_id}` | GET | Serve PNG image |
| `/api/images` | GET | List available image IDs |

## Configuration

Edit `config/paths.yaml` to set:
- `run.run_id`: `"latest"` (auto-select newest) or specific run ID
- `webqa.imgs_dir`: Path to WebQA images

## Testing

```bash
cd demo/be
pytest tests/ -v
```

## Directory Structure

```
demo/be/
  server.py           # FastAPI app entry point
  requirements.txt    # Dependencies
  config/
    paths.yaml        # Path configuration
    resolve_demo_paths.py  # Path resolver
  routers/            # API endpoints
  schemas/            # Pydantic models
  services/           # Data loading logic
  tests/              # API tests
```
