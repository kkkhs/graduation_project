from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO txt labels to COCO json")
    parser.add_argument("--images", required=True, help="image directory")
    parser.add_argument("--labels", required=True, help="label directory")
    parser.add_argument("--output", required=True, help="output coco json path")
    parser.add_argument("--category-name", default="ship", help="category name")
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
        help="comma-separated image extensions",
    )
    return parser.parse_args()


def list_images(images_dir: Path, exts: List[str]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(images_dir.rglob(f"*{ext}"))
    files = sorted(set(files))
    return files


def yolo_to_coco_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    w = width * img_w
    h = height * img_h
    x = (x_center * img_w) - (w / 2.0)
    y = (y_center * img_h) - (h / 2.0)
    return x, y, w, h


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exts = [x.strip().lower() for x in args.extensions.split(",") if x.strip()]
    image_files = list_images(images_dir, exts)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": args.category_name}],
    }

    ann_id = 1
    img_id = 1

    for img_path in image_files:
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        coco["images"].append(
            {
                "id": img_id,
                "file_name": str(img_path.relative_to(images_dir)).replace("\\", "/"),
                "width": img_w,
                "height": img_h,
            }
        )

        label_file = labels_dir / f"{img_path.stem}.txt"
        if label_file.exists():
            for line in label_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, xc, yc, w, h = parts[:5]
                x, y, bw, bh = yolo_to_coco_bbox(
                    float(xc), float(yc), float(w), float(h), img_w, img_h
                )
                if bw <= 0 or bh <= 0:
                    continue
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(float(cls)) + 1,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        img_id += 1

    output_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"images={len(coco['images'])}, annotations={len(coco['annotations'])}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
