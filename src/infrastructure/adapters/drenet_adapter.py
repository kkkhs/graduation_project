from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional

from src.infrastructure.adapters.base import BaseAdapter
from src.infrastructure.adapters.parsing import normalize_raw_predictions, rows_from_mmdet_result


class DRENetAdapter(BaseAdapter):
    """Adapter for DRENet inference.

    Supported integration modes:
    1) MMDetection-compatible mode
       - config_path: path to mmdet-style config file
       - weight_path: path to model checkpoint

    2) Custom plugin mode
       - config_path: "/abs/path/to/plugin.py:factory_name"
       - factory signature:
           factory(weight_path: str, config_path: str, device: str) -> predictor
       - predictor signature:
           predictor(image_path: str, conf_threshold: float, iou_threshold: float) -> raw_predictions
    """

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        self._mode: Optional[str] = None
        self._mmdet_model = None
        self._mmdet_infer = None
        self._plugin_predictor: Optional[Callable[[str, float, float], Any]] = None

    def load_model(self) -> None:
        if not self.model_config.weight_path:
            raise FileNotFoundError("DRENet weight_path is empty")

        weight_path = Path(self.model_config.weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"DRENet weight file not found: {weight_path}")

        if ":" in self.model_config.config_path:
            self._load_plugin_mode()
            return

        self._load_mmdet_mode()

    def infer(
        self,
        image_path: str,
        conf_threshold: float,
        iou_threshold: float,
        override_imgsz: int | None = None,
    ) -> List[Dict[str, Any]]:
        self.validate_image_path(image_path)

        if self._mode == "mmdet":
            if self._mmdet_model is None or self._mmdet_infer is None:
                raise RuntimeError("DRENet MMDetection backend is not loaded")
            try:
                result = self._mmdet_infer(self._mmdet_model, image_path)
            except Exception as exc:
                raise RuntimeError(f"DRENet inference failed for {image_path}") from exc
            return rows_from_mmdet_result(result, conf_threshold)

        if self._mode == "plugin":
            if self._plugin_predictor is None:
                raise RuntimeError("DRENet plugin predictor is not loaded")
            try:
                if override_imgsz is None:
                    raw = self._plugin_predictor(image_path, conf_threshold, iou_threshold)
                else:
                    raw = self._plugin_predictor(image_path, conf_threshold, iou_threshold, override_imgsz)
            except Exception as exc:
                raise RuntimeError(f"DRENet plugin inference failed for {image_path}") from exc
            return normalize_raw_predictions(raw, conf_threshold)

        raise RuntimeError("DRENet adapter is not initialized correctly")

    def _load_mmdet_mode(self) -> None:
        if not self.model_config.config_path:
            raise FileNotFoundError(
                "DRENet config_path is empty. Use mmdet config file path, or plugin format '/path/plugin.py:factory'."
            )

        config_path = Path(self.model_config.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"DRENet config file not found: {config_path}")

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
                "DRENet mmdet mode requires mmdet/mmcv/mmengine. "
                "If your DRENet is custom, set config_path as '/path/plugin.py:factory'."
            ) from exc

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            self._mmdet_model = init_detector(
                str(config_path),
                str(self.model_config.weight_path),
                device=device,
            )
            self._mmdet_infer = inference_detector
            self._mode = "mmdet"
        except Exception as exc:
            raise RuntimeError(
                f"failed to initialize DRENet (mmdet mode) with config={config_path}, "
                f"weight={self.model_config.weight_path}"
            ) from exc

    def _load_plugin_mode(self) -> None:
        plugin_path_str, factory_name = self.model_config.config_path.split(":", 1)
        plugin_path = Path(plugin_path_str)
        if not plugin_path.exists():
            raise FileNotFoundError(f"DRENet plugin file not found: {plugin_path}")

        module = self._load_module_from_file(plugin_path)
        factory = getattr(module, factory_name, None)
        if factory is None or not callable(factory):
            raise RuntimeError(
                f"DRENet plugin factory '{factory_name}' not found or not callable in {plugin_path}"
            )

        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        try:
            predictor = factory(
                weight_path=str(self.model_config.weight_path),
                config_path=str(plugin_path),
                device=device,
            )
        except Exception as exc:
            raise RuntimeError("failed to build DRENet plugin predictor") from exc

        if predictor is None or not callable(predictor):
            raise RuntimeError(
                "DRENet plugin factory must return a callable predictor(image_path, conf, iou)"
            )

        self._plugin_predictor = predictor
        self._mode = "plugin"

    @staticmethod
    def _load_module_from_file(plugin_path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location("drenet_plugin", str(plugin_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to create module spec for {plugin_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
