from __future__ import annotations

from src.application.predict_service import PredictService
from src.infrastructure.config_loader import load_runtime_config


def build_predict_service(config_path: str) -> PredictService:
    runtime = load_runtime_config(config_path)
    return PredictService(runtime)
