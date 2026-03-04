from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.application.dto import GlobalRuntimeConfig, ModelRuntimeConfig, RuntimeConfig


def _to_model_config(data: Dict[str, Any]) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        model_name=data["model_name"],
        framework_type=data["framework_type"],
        weight_path=data.get("weight_path", ""),
        config_path=data.get("config_path", ""),
        class_names=data.get("class_names", []),
        input_size=data.get("input_size", [640, 640]),
        default_conf_threshold=float(data.get("default_conf_threshold", 0.25)),
        default_iou_threshold=float(data.get("default_iou_threshold", 0.50)),
    )


def load_runtime_config(config_path: str) -> RuntimeConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    global_raw = raw.get("global", {})

    global_config = GlobalRuntimeConfig(
        device=global_raw.get("device", "cuda:0"),
        default_conf_threshold=float(global_raw.get("default_conf_threshold", 0.25)),
        default_iou_threshold=float(global_raw.get("default_iou_threshold", 0.50)),
        output_dir=global_raw.get("output_dir", "outputs/predictions"),
        visualization_dir=global_raw.get("visualization_dir", "outputs/visualizations"),
    )

    models = {}
    for entry in raw.get("models", []):
        model = _to_model_config(entry)
        models[model.model_name] = model

    return RuntimeConfig(global_config=global_config, models=models)
