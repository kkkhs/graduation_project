from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

from backend.app.core.settings import Settings
from src.application.bootstrap import build_predict_service
from src.application.fusion import fuse_predictions


class InferenceRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._service = build_predict_service(str(settings.config_path))

    def available_models(self) -> list[str]:
        return self._service.available_models()

    def predict_single(self, image_path: str, model_key: str, score_thr: float) -> list[dict[str, Any]]:
        if self.settings.mock_inference:
            return self._mock_predict(image_path, model_key, score_thr)
        return self._service.predict(
            image_path=image_path,
            model_name=model_key,
            conf_threshold=score_thr,
            iou_threshold=None,
        )

    def predict_ensemble(self, image_path: str, model_keys: list[str], score_thr: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        per_model: list[dict[str, Any]] = []
        for model_key in model_keys:
            per_model.extend(self.predict_single(image_path=image_path, model_key=model_key, score_thr=score_thr))

        fused = fuse_predictions(records=per_model, iou_threshold=0.55, min_votes=1)
        return per_model, fused

    def _mock_predict(self, image_path: str, model_key: str, score_thr: float) -> list[dict[str, Any]]:
        image_name = Path(image_path).name
        with Image.open(image_path) as image:
            width, height = image.size

        key = f"{image_name}:{model_key}".encode("utf-8")
        digest = hashlib.md5(key).hexdigest()

        base_x = int(digest[:2], 16) % max(1, width // 2)
        base_y = int(digest[2:4], 16) % max(1, height // 2)
        box_w = max(24, width // 6)
        box_h = max(24, height // 6)

        score = max(score_thr, 0.55)

        return [
            {
                "image_id": image_name,
                "model_name": model_key,
                "bbox": [float(base_x), float(base_y), float(box_w), float(box_h)],
                "score": float(min(score + 0.1, 0.99)),
                "category_id": 0,
                "inference_time": 5.0,
            }
        ]
