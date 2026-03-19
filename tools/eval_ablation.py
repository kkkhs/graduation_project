from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from src.application.bootstrap import build_predict_service


def iou_xywh(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1, y1, w1, h1 = [float(v) for v in box1]
    x2, y2, w2, h2 = [float(v) for v in box2]
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = max(0.0, w1 * h1) + max(0.0, w2 * h2) - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_yolo_gt(label_path: Path, image_size: tuple[int, int]) -> List[List[float]]:
    width, height = image_size
    rows: List[List[float]] = []
    if not label_path.exists():
        return rows
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return rows
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = parts[:5]
        xc = float(xc) * width
        yc = float(yc) * height
        bw = float(w) * width
        bh = float(h) * height
        x1 = xc - bw / 2.0
        y1 = yc - bh / 2.0
        rows.append([x1, y1, bw, bh])
    return rows


def image_size_from_png_or_jpg(image_path: Path) -> tuple[int, int]:
    import imghdr
    import struct

    kind = imghdr.what(image_path)
    if kind == "png":
        with image_path.open("rb") as f:
            f.read(8)
            header = f.read(25)
            width, height = struct.unpack(">LL", header[8:16])
            return int(width), int(height)
    if kind in {"jpeg", "jpg"}:
        with image_path.open("rb") as f:
            f.read(2)
            while True:
                marker, code = f.read(1), f.read(1)
                if not marker or not code:
                    break
                while code == b"\xff":
                    code = f.read(1)
                if code in {b"\xc0", b"\xc2"}:
                    f.read(3)
                    h, w = struct.unpack(">HH", f.read(4))
                    return int(w), int(h)
                seg_len = struct.unpack(">H", f.read(2))[0]
                f.read(seg_len - 2)
    raise RuntimeError(f"unsupported image format for size parsing: {image_path}")


@dataclass
class MetricSummary:
    model: str
    setting: str
    ap50: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    num_images: int


def evaluate_predictions(
    predictions_by_image: Dict[str, List[dict]],
    labels_dir: Path,
    images_dir: Path,
    iou_match: float = 0.5,
) -> MetricSummary:
    tp = 0
    fp = 0
    fn = 0
    scored_pairs: List[tuple[float, int]] = []

    image_names = sorted(predictions_by_image.keys())
    for image_name in image_names:
        image_path = images_dir / image_name
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        image_size = image_size_from_png_or_jpg(image_path)
        gts = load_yolo_gt(label_path, image_size)
        preds = sorted(predictions_by_image[image_name], key=lambda x: float(x["score"]), reverse=True)

        matched_gt = [False] * len(gts)
        for pred in preds:
            pred_box = pred["bbox"]
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt_box in enumerate(gts):
                if matched_gt[idx]:
                    continue
                score = iou_xywh(pred_box, gt_box)
                if score > best_iou:
                    best_iou = score
                    best_gt_idx = idx
            if best_gt_idx >= 0 and best_iou >= iou_match:
                matched_gt[best_gt_idx] = True
                tp += 1
                scored_pairs.append((float(pred["score"]), 1))
            else:
                fp += 1
                scored_pairs.append((float(pred["score"]), 0))

        fn += sum(1 for matched in matched_gt if not matched)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ap50 = compute_ap(scored_pairs, total_gt=tp + fn)
    return MetricSummary(
        model="",
        setting="",
        ap50=ap50,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        num_images=len(image_names),
    )


def compute_ap(scored_pairs: List[tuple[float, int]], total_gt: int) -> float:
    if total_gt <= 0 or not scored_pairs:
        return 0.0
    scored_pairs = sorted(scored_pairs, key=lambda x: x[0], reverse=True)
    tp_cum = 0
    fp_cum = 0
    recalls: List[float] = []
    precisions: List[float] = []
    for _, is_tp in scored_pairs:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        recalls.append(tp_cum / total_gt)
        precisions.append(tp_cum / (tp_cum + fp_cum))

    ap = 0.0
    for threshold in [i / 10.0 for i in range(11)]:
        prec_at_recall = [p for r, p in zip(recalls, precisions) if r >= threshold]
        ap += max(prec_at_recall) if prec_at_recall else 0.0
    return ap / 11.0


def collect_predictions(
    service,
    images_dir: Path,
    mode: str,
    model_name: str | None,
    conf: float,
    iou: float,
    imgsz: int | None,
    limit: int | None,
) -> Dict[str, List[dict]]:
    output: Dict[str, List[dict]] = {}
    image_paths = [path for path in sorted(images_dir.iterdir()) if path.is_file()]
    if limit is not None:
        image_paths = image_paths[:limit]

    for image_path in image_paths:
        if mode == "single":
            records = service.predict(
                image_path=str(image_path),
                model_name=model_name or "",
                conf_threshold=conf,
                iou_threshold=iou,
                override_imgsz=imgsz,
            )
        else:
            raise ValueError(f"unsupported mode: {mode}")
        output[image_path.name] = records
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ablation settings on LEVIR-Ship test split")
    parser.add_argument("--config", required=True, help="path to models yaml")
    parser.add_argument("--images-dir", required=True, help="test images directory")
    parser.add_argument("--labels-dir", required=True, help="test labels directory in YOLO format")
    parser.add_argument("--model", required=True, help="registered model name")
    parser.add_argument("--conf", type=float, required=True, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--imgsz", type=int, default=None, help="override inference image size")
    parser.add_argument("--group", required=True, help="ablation group name")
    parser.add_argument("--setting", required=True, help="setting identifier")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--limit", type=int, default=None, help="optional number of test images to evaluate")
    args = parser.parse_args()

    service = build_predict_service(args.config)
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = collect_predictions(
        service=service,
        images_dir=images_dir,
        mode="single",
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        limit=args.limit,
    )
    summary = evaluate_predictions(
        predictions_by_image=predictions,
        labels_dir=labels_dir,
        images_dir=images_dir,
        iou_match=0.5,
    )
    summary.model = args.model
    summary.setting = args.setting

    (out_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
