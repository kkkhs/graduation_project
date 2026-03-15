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

(
  ./scripts/start_backend.sh
) &
BACK_PID=$!

(
  ./scripts/start_frontend.sh
) &
FRONT_PID=$!

cleanup() {
  kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Backend PID: $BACK_PID"
echo "Frontend PID: $FRONT_PID"
wait
