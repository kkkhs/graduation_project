from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


def _safe_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size=size)
    except OSError:
        return ImageFont.load_default()


def _color_by_model(model_name: str) -> tuple[int, int, int]:
    digest = hashlib.md5(model_name.encode('utf-8')).hexdigest()
    r = 55 + int(digest[0:2], 16) % 170
    g = 55 + int(digest[2:4], 16) % 170
    b = 55 + int(digest[4:6], 16) % 170
    return r, g, b


def render_detections(image_path: str, predictions: List[Dict], output_path: str) -> str:
    src = Path(image_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(src).convert('RGB')
    width, height = image.size
    draw = ImageDraw.Draw(image)

    line_width = max(2, int(round(max(width, height) / 640)))
    font_size = max(13, int(round(max(width, height) / 70)))
    font = _safe_font(font_size)

    for item in predictions:
        bbox = item.get('bbox', [0, 0, 0, 0])
        if len(bbox) != 4:
            continue

        x, y, w, h = bbox
        x1 = max(0.0, float(x))
        y1 = max(0.0, float(y))
        x2 = min(float(width - 1), x1 + max(0.0, float(w)))
        y2 = min(float(height - 1), y1 + max(0.0, float(h)))
        if x2 <= x1 or y2 <= y1:
            continue

        score = float(item.get('score', 0.0))
        model_name = str(item.get('model_name') or item.get('source_model') or 'model')
        label = f'{model_name} {score:.3f}'
        color = _color_by_model(model_name)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad_x = max(3, line_width + 1)
        pad_y = max(2, line_width)

        label_x1 = x1
        label_y2 = y1
        label_y1 = max(0, label_y2 - text_h - pad_y * 2)
        label_x2 = min(float(width - 1), label_x1 + text_w + pad_x * 2)
        if label_x2 <= label_x1 + 6:
            label_x2 = min(float(width - 1), label_x1 + 6)

        draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=color)
        draw.text((label_x1 + pad_x, label_y1 + pad_y), label, fill=(255, 255, 255), font=font)

    if out.suffix.lower() == '.png':
        image.save(out, format='PNG', optimize=True)
    else:
        image.save(out, quality=95, subsampling=0, optimize=True)
    return str(out)
