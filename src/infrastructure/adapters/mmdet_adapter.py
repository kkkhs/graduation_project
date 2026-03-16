from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.infrastructure.adapters.base import BaseAdapter
from src.infrastructure.adapters.parsing import rows_from_mmdet_result


class MMDetAdapter(BaseAdapter):
    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        self._model = None
        self._inference_detector = None

    def load_model(self) -> None:
        config_path = Path(self.model_config.config_path)
        weight_path = Path(self.model_config.weight_path)

        if not self.model_config.config_path:
            raise FileNotFoundError("MMDetection config_path is empty")
        if not self.model_config.weight_path:
            raise FileNotFoundError("MMDetection weight_path is empty")
        if not config_path.exists():
            raise FileNotFoundError(f"MMDetection config file not found: {config_path}")
        if not weight_path.exists():
            raise FileNotFoundError(f"MMDetection weight file not found: {weight_path}")

        try:
            import os

            # Torch 2.6 defaults weights_only=True which breaks MMDet checkpoints.
            # We trust local checkpoints; allow full load for inference.
            os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")

            import torch
            try:
                import numpy as np
                from mmengine.logging import HistoryBuffer  # type: ignore
                from numpy.core.multiarray import _reconstruct
                from torch.serialization import add_safe_globals

                add_safe_globals([HistoryBuffer, _reconstruct, np.ndarray])
            except Exception:
                pass
            _orig_torch_load = torch.load

            def _safe_torch_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_torch_load(*args, **kwargs)

            torch.load = _safe_torch_load
            from mmdet.apis import inference_detector, init_detector  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "MMDetection is not installed. Install mmdet/mmcv/mmengine in your training env."
            ) from exc

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            self._model = init_detector(str(config_path), str(weight_path), device=device)
            self._inference_detector = inference_detector
        except Exception as exc:
            raise RuntimeError(
                f"failed to initialize MMDetection model with config={config_path}, weight={weight_path}"
            ) from exc

    def infer(
        self,
        image_path: str,
        conf_threshold: float,
        iou_threshold: float,
    ) -> List[Dict[str, Any]]:
        del iou_threshold  # NMS IoU is typically handled inside model config for MMDetection.
        self.validate_image_path(image_path)
        if self._model is None or self._inference_detector is None:
            raise RuntimeError("MMDetection model is not loaded")

        try:
            result = self._inference_detector(self._model, image_path)
        except Exception as exc:
            raise RuntimeError(f"MMDetection inference failed for {image_path}") from exc

        return rows_from_mmdet_result(result, conf_threshold)
