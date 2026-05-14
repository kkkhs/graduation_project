#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export APP_DB_PATH="${APP_DB_PATH:-$ROOT_DIR/app.db}"

if command -v uvicorn >/dev/null 2>&1; then
  UVICORN_BIN="uvicorn"
elif [ -x "$ROOT_DIR/.venv/bin/uvicorn" ]; then
  UVICORN_BIN="$ROOT_DIR/.venv/bin/uvicorn"
else
  echo "uvicorn not found. Please install dependencies or create .venv." >&2
  exit 1
fi

RELOAD_FLAG="${APP_RELOAD:-0}"
APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8000}"
if [ "$RELOAD_FLAG" = "1" ]; then
  "$UVICORN_BIN" backend.app.main:app --host "$APP_HOST" --port "$APP_PORT" --reload
else
  "$UVICORN_BIN" backend.app.main:app --host "$APP_HOST" --port "$APP_PORT"
fi
