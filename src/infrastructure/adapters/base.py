from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from src.application.dto import ModelRuntimeConfig
from src.domain.interfaces import DetectorAdapter


class BaseAdapter(DetectorAdapter, ABC):
    """Shared adapter behavior for framework-specific detectors."""

    def __init__(self, model_config: ModelRuntimeConfig) -> None:
        self.model_config = model_config
        self._loaded = False

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_model()
            self._loaded = True

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def infer(
        self,
        image_path: str,
        conf_threshold: float,
        iou_threshold: float,
    ) -> List[Dict[str, Any]]:
        pass

    def validate_image_path(self, image_path: str) -> None:
        if not Path(image_path).exists():
            raise ValueError(f"invalid image_path: {image_path}")
