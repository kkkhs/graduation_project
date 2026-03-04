from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.application.bootstrap import build_predict_service
from src.infrastructure.visualization import render_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure Python visualization entrypoint")
    parser.add_argument("--config", required=True, help="path to models yaml")
    parser.add_argument("--image", required=True, help="path to input image")
    parser.add_argument("--model", default="yolo", help="registered model name (single mode)")
    parser.add_argument(
        "--mode",
        choices=["single", "ensemble"],
        default="ensemble",
        help="single: run one model, ensemble: run multi-model fusion",
    )
    parser.add_argument(
        "--models",
        default="",
        help="comma-separated model list for ensemble mode; empty means all models",
    )
    parser.add_argument("--conf", type=float, default=None, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="iou threshold")
    parser.add_argument(
        "--fusion-iou",
        type=float,
        default=0.55,
        help="IoU threshold for ensemble fusion clustering",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help="minimum matched detections required to keep fused box",
    )
    parser.add_argument(
        "--vis-out",
        default="outputs/visualizations/vis_result.jpg",
        help="output path for visualized image",
    )
    parser.add_argument(
        "--json-out",
        default="outputs/predictions/pred_result.json",
        help="output path for json predictions",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display visualization using matplotlib window",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise ValueError(f"image not found: {args.image}")

    service = build_predict_service(args.config)
    if args.mode == "ensemble":
        model_names = [x.strip() for x in args.models.split(",") if x.strip()] if args.models else None
        records = service.predict_ensemble(
            image_path=str(image_path),
            model_names=model_names,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            fusion_iou_threshold=args.fusion_iou,
            min_votes=args.min_votes,
        )
        title_mode = "ensemble"
    else:
        records = service.predict(
            image_path=str(image_path),
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
        title_mode = args.model

    vis_out = Path(args.vis_out)
    vis_out.parent.mkdir(parents=True, exist_ok=True)
    saved_vis_path = render_detections(str(image_path), records, str(vis_out))

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"visualization saved: {saved_vis_path}")
    print(f"json saved: {json_out}")
    print(f"detections: {len(records)}")

    if args.show:
        import matplotlib.pyplot as plt
        from PIL import Image

        img = Image.open(saved_vis_path).convert("RGB")
        plt.figure(figsize=(8, 6))
        plt.title(f"Model: {title_mode} | Detections: {len(records)}")
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
