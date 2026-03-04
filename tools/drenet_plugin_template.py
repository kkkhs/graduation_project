"""DRENet custom predictor plugin template.

Usage in configs/models.yaml:
config_path: "/absolute/path/to/tools/drenet_plugin_template.py:build_predictor"
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_predictor(weight_path: str, config_path: str, device: str):
    """Return a callable predictor(image_path, conf_threshold, iou_threshold).

    Replace this function with real DRENet loading and inference code.
    """

    # TODO: load your DRENet model here.
    # Example:
    # model = load_drenet_model(weight_path=weight_path, device=device)

    def predictor(image_path: str, conf_threshold: float, iou_threshold: float) -> List[Dict[str, Any]]:
        del config_path, iou_threshold

        # TODO: run model inference and convert raw output.
        # Return unified rows:
        # [
        #   {
        #     "bbox": [x, y, w, h],
        #     "score": 0.95,
        #     "category_id": 0,
        #   }
        # ]

        # Placeholder empty output.
        _ = (weight_path, device, image_path, conf_threshold)
        return []

    return predictor
