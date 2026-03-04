from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelRuntimeConfig:
    model_name: str
    framework_type: str
    weight_path: str
    config_path: str
    class_names: List[str]
    input_size: List[int]
    default_conf_threshold: float
    default_iou_threshold: float


@dataclass(frozen=True)
class GlobalRuntimeConfig:
    device: str
    default_conf_threshold: float
    default_iou_threshold: float
    output_dir: str
    visualization_dir: str


@dataclass(frozen=True)
class RuntimeConfig:
    global_config: GlobalRuntimeConfig
    models: Dict[str, ModelRuntimeConfig]
