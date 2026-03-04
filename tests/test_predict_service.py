from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Dict, List

from src.application.dto import GlobalRuntimeConfig, ModelRuntimeConfig, RuntimeConfig
from src.application.predict_service import PredictService


@dataclass
class _FakeAdapter:
    rows: List[Dict[str, Any]]

    def ensure_loaded(self) -> None:
        return None

    def infer(self, image_path: str, conf_threshold: float, iou_threshold: float) -> List[Dict[str, Any]]:
        del image_path, conf_threshold, iou_threshold
        return self.rows


class _FakeFactory:
    def __init__(self, mapping: Dict[str, _FakeAdapter]) -> None:
        self._mapping = mapping

    def get_or_create(self, model_config: ModelRuntimeConfig):
        return self._mapping[model_config.model_name]


class PredictServiceTests(unittest.TestCase):
    def _make_service(self) -> PredictService:
        runtime = RuntimeConfig(
            global_config=GlobalRuntimeConfig(
                device="cpu",
                default_conf_threshold=0.25,
                default_iou_threshold=0.5,
                output_dir="outputs/predictions",
                visualization_dir="outputs/visualizations",
            ),
            models={
                "drenet": ModelRuntimeConfig(
                    model_name="drenet",
                    framework_type="drenet",
                    weight_path="/tmp/a.pth",
                    config_path="/tmp/a.py",
                    class_names=["ship"],
                    input_size=[640, 640],
                    default_conf_threshold=0.25,
                    default_iou_threshold=0.5,
                ),
                "yolo": ModelRuntimeConfig(
                    model_name="yolo",
                    framework_type="ultralytics",
                    weight_path="/tmp/b.pt",
                    config_path="",
                    class_names=["ship"],
                    input_size=[640, 640],
                    default_conf_threshold=0.25,
                    default_iou_threshold=0.5,
                ),
            },
        )
        service = PredictService(runtime)
        service.adapter_factory = _FakeFactory(
            {
                "drenet": _FakeAdapter(
                    rows=[{"bbox": [10, 10, 20, 20], "score": 0.9, "category_id": 0}]
                ),
                "yolo": _FakeAdapter(
                    rows=[{"bbox": [11, 10, 20, 20], "score": 0.8, "category_id": 0}]
                ),
            }
        )
        return service

    def test_predict_unknown_model(self) -> None:
        service = self._make_service()
        with self.assertRaises(ValueError):
            service.predict("a.jpg", "unknown")

    def test_predict_threshold_out_of_range(self) -> None:
        service = self._make_service()
        with self.assertRaises(ValueError):
            service.predict("a.jpg", "drenet", conf_threshold=1.2)

    def test_predict_ensemble_and_output_model_name(self) -> None:
        service = self._make_service()
        rows = service.predict_ensemble("a.jpg", model_names=["drenet", "yolo"], fusion_iou_threshold=0.3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_name"], "ensemble")

    def test_predict_ensemble_invalid_params(self) -> None:
        service = self._make_service()
        with self.assertRaises(ValueError):
            service.predict_ensemble("a.jpg", fusion_iou_threshold=1.2)
        with self.assertRaises(ValueError):
            service.predict_ensemble("a.jpg", min_votes=0)


if __name__ == "__main__":
    unittest.main()
