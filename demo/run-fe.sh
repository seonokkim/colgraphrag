#!/usr/bin/env bash
# Start Vite dev server; explicit host/port (same as package.json dev script).
# Prefers Linux Node via nvm over any Windows Node on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Load nvm if available (no-op when already on the right node)
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
# shellcheck source=/dev/null
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

cd "$ROOT/demo/fe"
exec npm run dev -- --host 0.0.0.0 --port 5173
