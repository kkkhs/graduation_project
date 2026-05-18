from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

from backend.app.core.settings import Settings
from src.application.bootstrap import build_predict_service
from src.application.fusion import fuse_predictions

logger = logging.getLogger(__name__)


class InferenceRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._service = build_predict_service(str(settings.config_path))
        self._parallel_pool: ThreadPoolExecutor | None = None

    def available_models(self) -> list[str]:
        return self._service.available_models()

    def preload_all(self) -> None:
        """Pre-load all configured model weights into memory before any inference."""
        self._service.preload_all()
        logger.info("all models preloaded: %s", self.available_models())

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
        max_parallel = self.settings.max_parallel_models
        if max_parallel <= 1 or len(model_keys) <= 1:
            # Serial fallback for low parallelism setting or single model
            per_model: list[dict[str, Any]] = []
            for model_key in model_keys:
                per_model.extend(self.predict_single(image_path=image_path, model_key=model_key, score_thr=score_thr))
        else:
            # Parallel inference across models
            per_model = self._parallel_predict(image_path, model_keys, score_thr, max_parallel)

        fused = fuse_predictions(records=per_model, iou_threshold=0.55, min_votes=1)
        return per_model, fused

    def _parallel_predict(
        self,
        image_path: str,
        model_keys: list[str],
        score_thr: float,
        max_workers: int,
    ) -> list[dict[str, Any]]:
        """Run inference for multiple models in parallel using ThreadPoolExecutor."""
        workers = min(max_workers, len(model_keys))
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="infer") as pool:
            future_map = {
                pool.submit(self.predict_single, image_path, key, score_thr): key
                for key in model_keys
            }
            for future in as_completed(future_map):
                model_key = future_map[future]
                try:
                    results.extend(future.result())
                except Exception:
                    logger.error("parallel inference failed for model: %s", model_key, exc_info=True)

        return results

    def shutdown(self) -> None:
        """Clean up parallel pool if it exists."""
        if self._parallel_pool is not None:
            self._parallel_pool.shutdown(wait=False)
            self._parallel_pool = None

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
