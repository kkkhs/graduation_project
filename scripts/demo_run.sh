#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

if ! python - <<'PY'
import importlib.util
import sys

required = [
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "alembic",
    "pydantic",
    "yaml",
    "PIL",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("MISSING_DEPS:" + ",".join(missing))
    sys.exit(1)
sys.exit(0)
PY
then
  echo "检测到关键依赖缺失，尝试安装 backend/requirements.txt ..."
  if ! pip install -r backend/requirements.txt; then
    echo "依赖安装失败（可能是当前网络不可用）。"
    echo "请先联网后执行：source .venv/bin/activate && pip install -r backend/requirements.txt"
    exit 1
  fi
fi

./scripts/init_db.sh

BACK_HOST="${APP_HOST:-127.0.0.1}"
BACK_PORT="${APP_PORT:-8000}"
WEB_HOST="${FRONTEND_HOST:-127.0.0.1}"
WEB_PORT="${FRONTEND_PORT:-5173}"

echo "启动后端: http://${BACK_HOST}:${BACK_PORT}"
echo "启动前端: http://${WEB_HOST}:${WEB_PORT}"

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
