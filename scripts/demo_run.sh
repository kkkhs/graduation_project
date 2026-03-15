#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install -r backend/requirements.txt
./scripts/init_db.sh

echo "启动后端: http://localhost:8000"
echo "启动前端: http://localhost:5173"

./scripts/start_backend.sh &
BACK_PID=$!

sleep 2
./scripts/start_frontend.sh &
FRONT_PID=$!

cleanup() {
  kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait
