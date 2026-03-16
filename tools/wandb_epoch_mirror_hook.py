from __future__ import annotations

import re
from pathlib import Path

import wandb
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

AR100_RE = re.compile(
    r"Average Recall\s+\(AR\) @\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([\-0-9.]+)"
)
AR1000_RE = re.compile(
    r"Average Recall\s+\(AR\) @\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=1000\s*\]\s*=\s*([\-0-9.]+)"
)


@HOOKS.register_module()
class WandbEpochMirrorHook(Hook):
    def __init__(self, epoch_offset: int = 0) -> None:
        self.epoch_offset = epoch_offset

    def before_train(self, runner) -> None:
        if wandb.run is None:
            return
        wandb.define_metric("progress/epoch")
        wandb.define_metric("metrics/*", step_metric="progress/epoch")

    def _latest_ar_from_log(self, log_file: Path) -> tuple[float | None, float | None]:
        if not log_file.exists():
            return None, None

        text = log_file.read_text(errors="ignore")
        tail = "\n".join(text.splitlines()[-500:])
        ar100 = None
        ar1000 = None

        matches = list(AR100_RE.finditer(tail))
        if matches:
            ar100 = float(matches[-1].group(1))

        matches = list(AR1000_RE.finditer(tail))
        if matches:
            ar1000 = float(matches[-1].group(1))

        return ar100, ar1000

    def after_val_epoch(self, runner, metrics=None) -> None:
        if wandb.run is None or not isinstance(metrics, dict):
            return

        payload = {"progress/epoch": int(runner.epoch) + self.epoch_offset}

        m50 = metrics.get("coco/bbox_mAP_50")
        if m50 is not None:
            payload["metrics/mAP50"] = float(m50)

        m5095 = metrics.get("coco/bbox_mAP")
        if m5095 is not None:
            payload["metrics/mAP50-95"] = float(m5095)

        log_file = Path(runner.work_dir).with_suffix(".log")
        ar100, ar1000 = self._latest_ar_from_log(log_file)
        if ar100 is not None:
            payload["metrics/recall"] = float(ar100)
        if ar1000 is not None:
            payload["metrics/recall@1000"] = float(ar1000)

        wandb.log(payload)
