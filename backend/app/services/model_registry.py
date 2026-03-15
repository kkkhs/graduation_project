from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from backend.app.db.models import ModelEntity



def load_model_entries(config_path: Path) -> list[dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"models config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    entries = []
    for item in raw.get("models", []):
        key = str(item.get("model_name", "")).strip()
        if not key:
            continue
        entries.append(
            {
                "name": key,
                "key": key,
                "weight_path": str(item.get("weight_path", "")),
                "is_enabled": bool(item.get("enabled", True)),
            }
        )
    return entries



def sync_models(session: Session, config_path: Path) -> None:
    desired = {entry["key"]: entry for entry in load_model_entries(config_path)}
    existing = {row.key: row for row in session.query(ModelEntity).all()}

    for key, entry in desired.items():
        row = existing.get(key)
        if row is None:
            session.add(
                ModelEntity(
                    name=entry["name"],
                    key=entry["key"],
                    weight_path=entry["weight_path"],
                    is_enabled=entry["is_enabled"],
                )
            )
        else:
            row.name = entry["name"]
            row.weight_path = entry["weight_path"]
            # Do not force overwrite is_enabled on every startup to preserve user toggle in DB.

    session.commit()
