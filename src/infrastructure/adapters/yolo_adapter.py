from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.infrastructure.adapters.base import BaseAdapter
from src.infrastructure.adapters.parsing import rows_from_ultralytics_results


class YOLOAdapter(BaseAdapter):
    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        self._model = None

    def load_model(self) -> None:
        weight_path = Path(self.model_config.weight_path)
        if not self.model_config.weight_path:
            raise FileNotFoundError("YOLO weight_path is empty")
        if not weight_path.exists():
            raise FileNotFoundError(f"YOLO weight file not found: {weight_path}")

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        try:
            self._model = YOLO(str(weight_path))
        except Exception as exc:
            raise RuntimeError(f"failed to load YOLO model from {weight_path}") from exc

    def infer(
        self,
        image_path: str,
        conf_threshold: float,
        iou_threshold: float,
        override_imgsz: int | None = None,
    ) -> List[Dict[str, Any]]:
        self.validate_image_path(image_path)
        if self._model is None:
            raise RuntimeError("YOLO model is not loaded")

        imgsz = override_imgsz or (
            int(self.model_config.input_size[0]) if self.model_config.input_size else 640
        )

        try:
            results = self._model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed for {image_path}") from exc

        return rows_from_ultralytics_results(results, conf_threshold)
