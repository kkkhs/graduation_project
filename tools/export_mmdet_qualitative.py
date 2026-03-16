#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MMDetection qualitative examples with GT and predictions."
    )
    parser.add_argument("--config", required=True, help="MMDetection config path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--ann", required=True, help="COCO annotation json")
    parser.add_argument("--image-root", required=True, help="Image root directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--score-thr", type=float, default=0.25, help="Prediction score threshold"
    )
    parser.add_argument(
        "--iou-thr", type=float, default=0.5, help="IoU threshold for TP/FP/FN matching"
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2,
        help="Maximum number of exported examples for each class",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Maximum number of images to scan from the annotation file",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Optional specific image file name(s) to prioritize",
    )
    return parser.parse_args()


def load_coco(ann_path: Path) -> tuple[list[dict[str, Any]], dict[int, list[list[float]]]]:
    data = json.loads(ann_path.read_text())
    images = data["images"]
    anns_by_image: dict[int, list[list[float]]] = defaultdict(list)
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        anns_by_image[ann["image_id"]].append([x, y, x + w, y + h])
    return images, anns_by_image


def iou(box1: list[float], box2: list[float]) -> float:
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    denom = area1 + area2 - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def classify(gt_boxes: list[list[float]], pred_boxes: list[list[float]], iou_thr: float) -> tuple[str, int, int, int]:
    matched_gt: set[int] = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_iou >= iou_thr and best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    if len(gt_boxes) > 0 and tp > 0 and fp == 0 and fn == 0:
        return "success", tp, fp, fn
    if fn > 0:
        return "miss", tp, fp, fn
    if fp > 0:
        return "false_positive", tp, fp, fn
    return "other", tp, fp, fn


def extract_pred_boxes(result: Any, score_thr: float) -> list[list[float]]:
    pred_instances = getattr(result, "pred_instances", None)
    if pred_instances is None:
        return []
    boxes = pred_instances.bboxes.cpu().numpy().tolist()
    scores = pred_instances.scores.cpu().numpy().tolist()
    labels = pred_instances.labels.cpu().numpy().tolist()
    kept: list[list[float]] = []
    for box, score, label in zip(boxes, scores, labels):
        if int(label) != 0:
            continue
        if float(score) < score_thr:
            continue
        kept.append([float(v) for v in box] + [float(score)])
    return kept


def draw_boxes(image_path: Path, gt_boxes: list[list[float]], pred_boxes: list[list[float]], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for gt in gt_boxes:
        draw.rectangle(gt, outline="red", width=2)

    for pred in pred_boxes:
        x1, y1, x2, y2, score = pred
        draw.rectangle([x1, y1, x2, y2], outline="deepskyblue", width=2)
        draw.text((x1, max(0, y1 - 12)), f"ship {score:.2f}", fill="deepskyblue")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> int:
    args = parse_args()

    from mmdet.apis import inference_detector, init_detector  # type: ignore

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    ann_path = Path(args.ann)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images, anns_by_image = load_coco(ann_path)
    if args.image:
        preferred = set(args.image)
        chosen_images = [img for img in images if img["file_name"] in preferred]
        remaining = [img for img in images if img["file_name"] not in preferred]
        images = chosen_images + remaining

    model = init_detector(str(config_path), str(checkpoint_path), device="cuda:0")

    exported_count = {"success": 0, "miss": 0, "false_positive": 0}
    records: list[dict[str, Any]] = []

    for image_info in images[: args.limit]:
        if all(v >= args.max_per_class for v in exported_count.values()):
            break

        file_name = image_info["file_name"]
        image_path = image_root / file_name
        if not image_path.exists():
            continue

        gt_boxes = anns_by_image.get(image_info["id"], [])
        result = inference_detector(model, str(image_path))
        pred_boxes = extract_pred_boxes(result, args.score_thr)
        category, tp, fp, fn = classify(gt_boxes, [p[:4] for p in pred_boxes], args.iou_thr)

        if category not in exported_count:
            continue
        if exported_count[category] >= args.max_per_class:
            continue

        out_name = f"{Path(file_name).stem}_{category}.jpg"
        out_path = output_dir / category / out_name
        draw_boxes(image_path, gt_boxes, pred_boxes, out_path)
        exported_count[category] += 1
        records.append(
            {
                "file_name": file_name,
                "category": category,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "output_path": str(out_path),
            }
        )

    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "ann": str(ann_path),
        "image_root": str(image_root),
        "score_thr": args.score_thr,
        "iou_thr": args.iou_thr,
        "exported_count": exported_count,
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
