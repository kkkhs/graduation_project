from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class DetectorAdapter(ABC):
    @abstractmethod
    def ensure_loaded(self) -> None:
        pass

    @abstractmethod
    def infer(
        self,
        image_path: str,
        conf_threshold: float,
        iou_threshold: float,
        override_imgsz: int | None = None,
    ) -> List[Dict[str, Any]]:
        pass
