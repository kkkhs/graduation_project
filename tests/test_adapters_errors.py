from __future__ import annotations

import unittest

from src.application.dto import ModelRuntimeConfig
from src.infrastructure.adapters.drenet_adapter import DRENetAdapter
from src.infrastructure.adapters.mmdet_adapter import MMDetAdapter
from src.infrastructure.adapters.yolo_adapter import YOLOAdapter


def _cfg(model_name: str, framework: str, weight: str, config: str) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        model_name=model_name,
        framework_type=framework,
        weight_path=weight,
        config_path=config,
        class_names=["ship"],
        input_size=[640, 640],
        default_conf_threshold=0.25,
        default_iou_threshold=0.5,
    )


class AdapterErrorTests(unittest.TestCase):
    def test_yolo_empty_weight(self) -> None:
        adapter = YOLOAdapter(_cfg("yolo", "ultralytics", "", ""))
        with self.assertRaises(FileNotFoundError):
            adapter.load_model()

    def test_mmdet_empty_config(self) -> None:
        adapter = MMDetAdapter(_cfg("mmdet", "mmdetection", "/tmp/a.pth", ""))
        with self.assertRaises(FileNotFoundError):
            adapter.load_model()

    def test_drenet_missing_plugin_file(self) -> None:
        adapter = DRENetAdapter(
            _cfg("drenet", "drenet", "/tmp/a.pth", "/tmp/not_found_plugin.py:build_predictor")
        )
        with self.assertRaises(FileNotFoundError):
            adapter.load_model()


if __name__ == "__main__":
    unittest.main()
