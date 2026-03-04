from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw


def render_detections(image_path: str, predictions: List[Dict], output_path: str) -> str:
    src = Path(image_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(image)

    for item in predictions:
        bbox = item.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        score = item.get("score", 0.0)
        label = f"ship {score:.3f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(255, 0, 0))

    image.save(out)
    return str(out)
