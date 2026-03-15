"""DRENet local plugin backed by the original DRENet repo code.

Usage in configs/models.yaml:
config_path: "/Users/khs/codes/graduation_project/tools/drenet_local_plugin.py:build_predictor"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


DRENET_REPO = Path("/Users/khs/codes/graduation_project/experiments/drenet/DRENet")



def _ensure_repo_importable() -> None:
    if not DRENET_REPO.exists():
        raise FileNotFoundError(f"DRENet repo not found: {DRENET_REPO}")
    repo_str = str(DRENET_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)



def build_predictor(weight_path: str, config_path: str, device: str):
    del config_path  # not used by plugin mode

    _ensure_repo_importable()

    # PyTorch 2.6 changed torch.load default to weights_only=True, which breaks legacy YOLOv5 ckpt loading.
    _orig_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _compat_torch_load

    from models.experimental import attempt_load  # type: ignore
    from utils.general import check_img_size, non_max_suppression, scale_coords  # type: ignore
    from utils.torch_utils import select_device  # type: ignore
    from utils.datasets import LoadImages  # type: ignore

    # DRENet code supports cpu/cuda with YOLOv5-style device selector.
    requested = "cpu" if device.startswith("cpu") else device.replace("cuda:", "")
    runtime_device = select_device(requested)

    model = attempt_load(weight_path, map_location=runtime_device)
    model.eval()

    stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    imgsz = check_img_size(512, s=stride)
    use_half = runtime_device.type != "cpu"
    if use_half:
        model.half()

    @torch.no_grad()
    def predictor(image_path: str, conf_threshold: float, iou_threshold: float) -> List[Dict[str, Any]]:
        dataset = LoadImages(image_path, img_size=imgsz)
        rows: List[Dict[str, Any]] = []

        for _, img, im0s, _ in dataset:
            tensor = torch.from_numpy(img).to(runtime_device)
            tensor = tensor.half() if use_half else tensor.float()
            tensor /= 255.0
            if tensor.ndimension() == 3:
                tensor = tensor.unsqueeze(0)

            pred = model(tensor, augment=False)[0][0]
            pred = non_max_suppression(pred, conf_threshold, iou_threshold)

            det = pred[0] if pred else None
            if det is None or len(det) == 0:
                continue

            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], im0s.shape).round()

            for *xyxy, conf, cls in det.tolist():
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                rows.append(
                    {
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(conf),
                        "category_id": int(cls),
                    }
                )

        return rows

    return predictor
