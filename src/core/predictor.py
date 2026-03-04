"""Compatibility wrapper for legacy src.core.predictor.UnifiedPredictor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.application.dto import RuntimeConfig
from src.application.predict_service import PredictService


class UnifiedPredictor:
    """Backward-compatible wrapper around application PredictService."""

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self._service = PredictService(runtime_config)

    def predict(
        self,
        image_path: str,
        model_name: str,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return self._service.predict(
            image_path=image_path,
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    def predict_ensemble(
        self,
        image_path: str,
        model_names: Optional[List[str]] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        fusion_iou_threshold: float = 0.55,
        min_votes: int = 1,
    ) -> List[Dict[str, Any]]:
        return self._service.predict_ensemble(
            image_path=image_path,
            model_names=model_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            fusion_iou_threshold=fusion_iou_threshold,
            min_votes=min_votes,
        )
