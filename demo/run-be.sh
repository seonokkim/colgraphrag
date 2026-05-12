#!/usr/bin/env bash
# Start demo FastAPI from repo root venv; bind 0.0.0.0 for remote access.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/demo/be"
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate"

# Load Gemma in bfloat16 so it fits in GPU VRAM (~8 GB vs ~15 GB for float32).
export GEMMA4_E4B_IT_TORCH_DTYPE="${GEMMA4_E4B_IT_TORCH_DTYPE:-bfloat16}"

exec python server.py --host 0.0.0.0 --port 8000
