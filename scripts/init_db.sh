#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export APP_DB_PATH="${APP_DB_PATH:-$ROOT_DIR/app.db}"

DB_ACTION="$(python - <<'PY'
import os
import sqlite3
from pathlib import Path

db_path = Path(os.environ["APP_DB_PATH"])
if not db_path.exists():
    print("upgrade")
    raise SystemExit(0)

conn = sqlite3.connect(str(db_path))
try:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
finally:
    conn.close()

tables = {name for (name,) in rows}
schema_ready = {"models", "tasks", "results", "task_files"}.issubset(tables)
if "alembic_version" in tables:
    conn = sqlite3.connect(str(db_path))
    try:
        version_rows = conn.execute("SELECT version_num FROM alembic_version").fetchall()
    finally:
        conn.close()
    if not version_rows and schema_ready:
        print("stamp")
    else:
        print("upgrade")
elif schema_ready:
    print("stamp")
else:
    print("upgrade")
PY
)"

if [ "$DB_ACTION" = "stamp" ]; then
  python -m alembic -c backend/alembic.ini stamp head
else
  python -m alembic -c backend/alembic.ini upgrade head
fi
python - <<'PY'
from backend.app.core.settings import load_settings
from backend.app.db.models import Base
from backend.app.db.session import build_engine, build_session_factory
from backend.app.services.model_registry import sync_models

settings = load_settings()
engine = build_engine(str(settings.db_path))
Base.metadata.create_all(bind=engine)
Session = build_session_factory(engine)
with Session() as session:
    sync_models(session, settings.config_path)
print(f"DB initialized at {settings.db_path}")
PY
