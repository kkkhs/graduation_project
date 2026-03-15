from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    project_root: Path
    config_path: Path
    db_path: Path
    outputs_root: Path
    task_output_root: Path
    mock_inference: bool
    max_workers: int



def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}



def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[3]

    config_path = Path(os.getenv("APP_CONFIG_PATH", str(project_root / "configs" / "models.yaml"))).resolve()
    db_path = Path(os.getenv("APP_DB_PATH", str(project_root / "app.db"))).resolve()
    outputs_root = Path(os.getenv("APP_OUTPUTS_ROOT", str(project_root / "outputs"))).resolve()
    task_output_root = Path(os.getenv("APP_TASK_OUTPUT_ROOT", str(outputs_root / "tasks"))).resolve()
    max_workers = int(os.getenv("APP_MAX_WORKERS", "1"))

    return Settings(
        project_root=project_root,
        config_path=config_path,
        db_path=db_path,
        outputs_root=outputs_root,
        task_output_root=task_output_root,
        mock_inference=_as_bool(os.getenv("APP_MOCK_INFERENCE"), default=False),
        max_workers=max(1, max_workers),
    )
