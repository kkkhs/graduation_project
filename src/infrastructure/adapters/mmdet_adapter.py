from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import copy

from src.infrastructure.adapters.base import BaseAdapter
from src.infrastructure.adapters.parsing import rows_from_mmdet_result


class MMDetAdapter(BaseAdapter):
    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        self._model = None
        self._inference_detector = None
        self._base_config_path = None
        self._base_weight_path = None
        self._current_imgsz = None

    def load_model(self) -> None:
        config_path = Path(self.model_config.config_path)
        weight_path = Path(self.model_config.weight_path)
        self._base_config_path = config_path
        self._base_weight_path = weight_path

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
            self._current_imgsz = None
        except Exception as exc:
            raise RuntimeError(
                f"failed to initialize MMDetection model with config={config_path}, weight={weight_path}"
            ) from exc

    def infer(
        self,
        image_path: str,
        conf_threshold: float,
        iou_threshold: float,
        override_imgsz: int | None = None,
    ) -> List[Dict[str, Any]]:
        del iou_threshold  # NMS IoU is typically handled inside model config for MMDetection.
        self.validate_image_path(image_path)
        if self._model is None or self._inference_detector is None:
            raise RuntimeError("MMDetection model is not loaded")

        if override_imgsz is not None:
            self._ensure_resize_override(override_imgsz)

        try:
            result = self._inference_detector(self._model, image_path)
        except Exception as exc:
            raise RuntimeError(f"MMDetection inference failed for {image_path}") from exc

        return rows_from_mmdet_result(result, conf_threshold)

    def _ensure_resize_override(self, override_imgsz: int) -> None:
        if self._current_imgsz == override_imgsz:
            return
        if self._base_config_path is None or self._base_weight_path is None:
            raise RuntimeError("MMDetection base config/weight is not initialized")
        if self._inference_detector is None:
            raise RuntimeError("MMDetection inference api is not loaded")

        try:
            from mmengine.config import Config  # type: ignore
            from mmdet.apis import init_detector  # type: ignore
            import torch
        except Exception as exc:
            raise RuntimeError("MMDetection resize override requires mmdet/mmengine") from exc

        cfg = Config.fromfile(str(self._base_config_path))
        scale = (override_imgsz, override_imgsz)

        for attr in ("test_dataloader", "val_dataloader"):
            dataloader = getattr(cfg, attr, None)
            if dataloader and dataloader.get("dataset"):
                pipeline = dataloader["dataset"].get("pipeline", [])
                self._patch_resize_pipeline(pipeline, scale)

        if hasattr(cfg, "test_pipeline"):
            self._patch_resize_pipeline(cfg.test_pipeline, scale)
        if hasattr(cfg, "val_pipeline"):
            self._patch_resize_pipeline(cfg.val_pipeline, scale)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            self._model = init_detector(cfg, str(self._base_weight_path), device=device)
            self._current_imgsz = override_imgsz
        except Exception as exc:
            raise RuntimeError(f"failed to reinitialize MMDetection model for imgsz={override_imgsz}") from exc

    @staticmethod
    def _patch_resize_pipeline(pipeline: List[Dict[str, Any]], scale: tuple[int, int]) -> None:
        for step in pipeline:
            if not isinstance(step, dict):
                continue
            if step.get("type") == "Resize":
                step["scale"] = scale
