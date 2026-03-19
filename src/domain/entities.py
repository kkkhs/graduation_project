from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PredictRequest:
    image_path: str
    model_name: str
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    override_imgsz: Optional[int] = None


@dataclass(frozen=True)
class PredictionResult:
    image_id: str
    model_name: str
    bbox: List[float]
    score: float
    category_id: int
    inference_time: float
