from __future__ import annotations

import argparse
import os

# Torch 2.6 defaults weights_only=True which breaks MMDet checkpoints.
# Set early before any torch import happens.
os.environ.setdefault("TORCH_LOAD_WEIGHTS_ONLY", "0")
import json

from src.application.bootstrap import build_predict_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified prediction entrypoint")
    parser.add_argument("--config", required=True, help="path to models yaml")
    parser.add_argument("--image", required=True, help="image path")
    parser.add_argument("--model", default="yolo", help="registered model name (single mode)")
    parser.add_argument(
        "--mode",
        choices=["single", "ensemble"],
        default="ensemble",
        help="single: run one model, ensemble: run all/selected models",
    )
    parser.add_argument(
        "--models",
        default="",
        help="comma-separated model names for ensemble mode, empty means all",
    )
    parser.add_argument("--conf", type=float, default=None, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="iou threshold")
    parser.add_argument("--fusion-iou", type=float, default=0.55, help="IoU threshold for fusion")
    parser.add_argument("--min-votes", type=int, default=1, help="min votes for fused result")
    args = parser.parse_args()

    predictor = build_predict_service(args.config)

    if args.mode == "ensemble":
        model_names = [x.strip() for x in args.models.split(",") if x.strip()] if args.models else None
        results = predictor.predict_ensemble(
            image_path=args.image,
            model_names=model_names,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            fusion_iou_threshold=args.fusion_iou,
            min_votes=args.min_votes,
        )
    else:
        results = predictor.predict(
            image_path=args.image,
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
