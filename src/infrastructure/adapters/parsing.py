from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clamp_score(score: Any) -> float:
    value = float(score)
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def normalize_record(record: Dict[str, Any], conf_threshold: float) -> Dict[str, Any] | None:
    bbox = record.get("bbox")
    if not bbox or len(bbox) != 4:
        return None

    score = clamp_score(record.get("score", 0.0))
    if score < conf_threshold:
        return None

    return {
        "bbox": [float(v) for v in bbox],
        "score": score,
        "category_id": int(record.get("category_id", 0)),
    }


def normalize_raw_predictions(raw: Any, conf_threshold: float) -> List[Dict[str, Any]]:
    """Normalize custom predictor output into unified detection rows.

    Accepted raw formats:
    1) list[dict] with keys bbox/score/category_id
    2) dict with key 'detections' -> list[dict]
    3) list[list|tuple] each as [x1,y1,x2,y2,score,label(optional)]
    """
    if raw is None:
        return []

    if isinstance(raw, dict) and "detections" in raw:
        raw = raw["detections"]

    detections: List[Dict[str, Any]] = []

    if isinstance(raw, list):
        if len(raw) == 0:
            return detections

        if isinstance(raw[0], dict):
            for item in raw:
                normalized = normalize_record(item, conf_threshold)
                if normalized is not None:
                    detections.append(normalized)
            return detections

        if isinstance(raw[0], (list, tuple)):
            for item in raw:
                if len(item) < 5:
                    continue
                bbox = xyxy_to_xywh(item[0:4])
                score = clamp_score(item[4])
                if score < conf_threshold:
                    continue
                category_id = int(item[5]) if len(item) > 5 else 0
                detections.append(
                    {
                        "bbox": bbox,
                        "score": score,
                        "category_id": category_id,
                    }
                )
            return detections

    raise ValueError("unsupported raw prediction format from custom predictor")


def rows_from_mmdet_result(result: Any, conf_threshold: float) -> List[Dict[str, Any]]:
    """Parse MMDetection inference result into unified rows."""
    sample = result[0] if isinstance(result, list) else result
    pred_instances = getattr(sample, "pred_instances", None)
    if pred_instances is None:
        return []

    bboxes = pred_instances.bboxes.detach().cpu().numpy()
    scores = pred_instances.scores.detach().cpu().numpy()
    labels = pred_instances.labels.detach().cpu().numpy()

    rows: List[Dict[str, Any]] = []
    for bbox_xyxy, score, label in zip(bboxes, scores, labels):
        score_value = clamp_score(score)
        if score_value < conf_threshold:
            continue
        rows.append(
            {
                "bbox": xyxy_to_xywh(bbox_xyxy),
                "score": score_value,
                "category_id": int(label),
            }
        )
    return rows


def rows_from_ultralytics_results(results: Iterable[Any], conf_threshold: float) -> List[Dict[str, Any]]:
    """Parse Ultralytics YOLO results into unified rows."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy()

        for box_xyxy, score, cls_id in zip(xyxy, conf, cls):
            score_value = clamp_score(score)
            if score_value < conf_threshold:
                continue
            rows.append(
                {
                    "bbox": xyxy_to_xywh(box_xyxy),
                    "score": score_value,
                    "category_id": int(cls_id),
                }
            )
    return rows
