from __future__ import annotations

import threading
from typing import Dict

from src.application.dto import ModelRuntimeConfig
from src.domain.interfaces import DetectorAdapter
from src.infrastructure.adapters.drenet_adapter import DRENetAdapter
from src.infrastructure.adapters.mmdet_adapter import MMDetAdapter
from src.infrastructure.adapters.yolo_adapter import YOLOAdapter


class AdapterFactory:
    def __init__(self) -> None:
        self._instances: Dict[str, DetectorAdapter] = {}
        self._lock = threading.Lock()

    def get_or_create(self, model_config: ModelRuntimeConfig) -> DetectorAdapter:
        name = model_config.model_name
        with self._lock:
            if name in self._instances:
                return self._instances[name]

            adapter = self._build_adapter(model_config)
            self._instances[name] = adapter
            return adapter

    def preload_all(self, model_configs: Dict[str, ModelRuntimeConfig]) -> None:
        """Load all model weights into memory without running inference."""
        for name, config in model_configs.items():
            adapter = self.get_or_create(config)
            adapter.ensure_loaded()

    @staticmethod
    def _build_adapter(model_config: ModelRuntimeConfig) -> DetectorAdapter:
        framework = model_config.framework_type.lower()
        if framework == "drenet":
            return DRENetAdapter(model_config)
        if framework == "mmdetection":
            return MMDetAdapter(model_config)
        if framework == "ultralytics":
            return YOLOAdapter(model_config)
        raise ValueError(f"unsupported framework_type: {model_config.framework_type}")
