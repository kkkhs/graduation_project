from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.application.fusion import fuse_predictions
from src.application.dto import RuntimeConfig
from src.domain.entities import PredictRequest, PredictionResult
from src.infrastructure.adapter_factory import AdapterFactory


class PredictService:
    """Application-layer orchestrator for unified inference."""

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self.adapter_factory = AdapterFactory()

    def available_models(self) -> List[str]:
        return sorted(list(self.runtime_config.models.keys()))

    def predict_ensemble(
        self,
        image_path: str,
        model_names: Optional[Sequence[str]] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        fusion_iou_threshold: float = 0.55,
        min_votes: int = 1,
    ) -> List[Dict[str, Any]]:
        if fusion_iou_threshold < 0 or fusion_iou_threshold > 1:
            raise ValueError("fusion_iou_threshold out of range")
        if min_votes < 1:
            raise ValueError("min_votes must be >= 1")

        selected_models = self._resolve_models(model_names)
        all_records: List[Dict[str, Any]] = []
        for name in selected_models:
            records = self.predict(
                image_path=image_path,
                model_name=name,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            all_records.extend(records)
        return fuse_predictions(
            records=all_records,
            iou_threshold=fusion_iou_threshold,
            min_votes=min_votes,
        )

    def predict(
        self,
        image_path: str,
        model_name: str,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        request = PredictRequest(
            image_path=image_path,
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        results = self.predict_by_request(request)
        return [
            {
                "image_id": item.image_id,
                "model_name": item.model_name,
                "bbox": item.bbox,
                "score": item.score,
                "category_id": item.category_id,
                "inference_time": item.inference_time,
            }
            for item in results
        ]

    def predict_by_request(self, request: PredictRequest) -> List[PredictionResult]:
        if request.model_name not in self.runtime_config.models:
            raise ValueError(f"unknown model_name: {request.model_name}")

        model = self.runtime_config.models[request.model_name]
        conf = self._resolve_threshold(request.conf_threshold, model.default_conf_threshold)
        iou = self._resolve_threshold(request.iou_threshold, model.default_iou_threshold)

        adapter = self.adapter_factory.get_or_create(model)
        adapter.ensure_loaded()

        t0 = time.perf_counter()
        preds = adapter.infer(image_path=request.image_path, conf_threshold=conf, iou_threshold=iou)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        image_id = Path(request.image_path).name
        return [self._normalize_row(image_id, request.model_name, row, dt_ms) for row in preds]

    @staticmethod
    def _resolve_threshold(value: Optional[float], default: float) -> float:
        output = default if value is None else float(value)
        if output < 0 or output > 1:
            raise ValueError("threshold out of range")
        return output

    @staticmethod
    def _normalize_row(
        image_id: str,
        model_name: str,
        row: Dict[str, Any],
        inference_time_ms: float,
    ) -> PredictionResult:
        return PredictionResult(
            image_id=image_id,
            model_name=model_name,
            bbox=row.get("bbox", [0, 0, 0, 0]),
            score=float(row.get("score", 0.0)),
            category_id=int(row.get("category_id", 0)),
            inference_time=float(inference_time_ms),
        )

    def _resolve_models(self, model_names: Optional[Sequence[str]]) -> List[str]:
        if model_names is None:
            return self.available_models()

        names = [x.strip() for x in model_names if x and x.strip()]
        if not names:
            raise ValueError("model_names is empty")

        unknown = [x for x in names if x not in self.runtime_config.models]
        if unknown:
            raise ValueError(f"unknown model_names: {unknown}")
        return names
