from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def _xywh_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def _xyxy_to_xywh(bbox: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _weighted_box(items: List[Dict[str, Any]]) -> List[float]:
    total = sum(float(max(1e-6, x.get("score", 0.0))) for x in items)
    if total <= 0:
        return list(items[0].get("bbox", [0, 0, 0, 0]))

    x1 = y1 = x2 = y2 = 0.0
    for item in items:
        w = float(max(1e-6, item.get("score", 0.0)))
        bx1, by1, bx2, by2 = _xywh_to_xyxy(item.get("bbox", [0, 0, 0, 0]))
        x1 += bx1 * w
        y1 += by1 * w
        x2 += bx2 * w
        y2 += by2 * w

    return _xyxy_to_xywh([x1 / total, y1 / total, x2 / total, y2 / total])


def fuse_predictions(
    records: List[Dict[str, Any]],
    iou_threshold: float = 0.55,
    min_votes: int = 1,
) -> List[Dict[str, Any]]:
    """Fuse multi-model predictions with IoU-based weighted clustering."""
    if not records:
        return []

    candidates = sorted(records, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    clusters: List[List[Dict[str, Any]]] = []

    for det in candidates:
        matched = False
        cat = int(det.get("category_id", 0))
        bbox = det.get("bbox", [0, 0, 0, 0])

        for cluster in clusters:
            ref = cluster[0]
            if int(ref.get("category_id", 0)) != cat:
                continue
            ref_box = _weighted_box(cluster)
            if iou_xywh(bbox, ref_box) >= iou_threshold:
                cluster.append(det)
                matched = True
                break

        if not matched:
            clusters.append([det])

    fused: List[Dict[str, Any]] = []
    for cluster in clusters:
        if len(cluster) < min_votes:
            continue
        image_id = str(cluster[0].get("image_id", ""))
        bbox = _weighted_box(cluster)
        score = sum(float(x.get("score", 0.0)) for x in cluster) / len(cluster)
        inference_time = max(float(x.get("inference_time", 0.0)) for x in cluster)
        fused.append(
            {
                "image_id": image_id,
                "model_name": "ensemble",
                "bbox": bbox,
                "score": float(score),
                "category_id": int(cluster[0].get("category_id", 0)),
                "inference_time": float(inference_time),
            }
        )

    fused.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return fused
