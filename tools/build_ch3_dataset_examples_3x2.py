from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/Users/khs/codes/graduation_project')
IMG_DIR = ROOT / 'experiment_assets/datasets/LEVIR-Ship/test/images'
LBL_DIR = ROOT / 'experiment_assets/datasets/LEVIR-Ship/test/labels'
OUT = ROOT / 'thesis_overleaf/figures/generated/ch3_dataset_examples.png'

# fixed size for readability in thesis
TILE_W = 420
TILE_H = 420
GAP_X = 16
GAP_Y = 16

# choose 6 representative samples with 1-3 ships
SAMPLE_FILES = [
    'GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_0_6144.png',
    'GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_1536_2560.png',
    'GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_2048_2048.png',
    'GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_3584_11776.png',
    'GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_4096_11264.png',
    'GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_4608_11264.png',
]

BOX_COLOR = (80, 220, 80)
FONT_PATH = Path('/System/Library/Fonts/Supplemental/Times New Roman.ttf')
# Figure-internal annotation should be smaller than thesis body text.
FONT_SIZE_BADGE = 26


def load_font(size):
    try:
        return ImageFont.truetype(str(FONT_PATH), size=size)
    except OSError:
        return ImageFont.load_default()


def text_size(text, font):
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1), (0, 0, 0)))
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def yolo_to_xyxy(line, w, h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, xc, yc, bw, bh = map(float, parts[:5])
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h
    return [x1, y1, x2, y2]


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def get_gt_boxes(file_name):
    p = LBL_DIR / file_name.replace('.png', '.txt')
    boxes = []
    if not p.exists():
        return boxes
    w = 512
    h = 512
    for ln in p.read_text(encoding='utf-8').splitlines():
        ln = ln.strip()
        if not ln:
            continue
        b = yolo_to_xyxy(ln, w, h)
        if b:
            boxes.append(b)
    return boxes


def draw_boxes(img, boxes, color):
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        x1 = max(0, min(out.shape[1] - 1, x1))
        x2 = max(0, min(out.shape[1] - 1, x2))
        y1 = max(0, min(out.shape[0] - 1, y1))
        y2 = max(0, min(out.shape[0] - 1, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def add_badge(img, text):
    pil = Image.fromarray(img).convert('RGBA')
    overlay = Image.new('RGBA', pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(FONT_SIZE_BADGE)
    bx0, by0, _, _ = draw.textbbox((0, 0), text, font=font)
    x0, y0 = 14, 14
    tx = int(round(x0 - bx0))
    ty = int(round(y0 - by0))
    draw.text((tx, ty), text, font=font, fill=(245, 245, 245, 255))
    out = Image.alpha_composite(pil, overlay).convert('RGB')
    return np.array(out)


def resize(img):
    return cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_CUBIC)


def main():
    canvas_w = TILE_W * 3 + GAP_X * 2
    canvas_h = TILE_H * 2 + GAP_Y
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for i, fn in enumerate(SAMPLE_FILES):
        base = read_rgb(IMG_DIR / fn)
        gt = get_gt_boxes(fn)
        panel = draw_boxes(base, gt, BOX_COLOR)
        panel = resize(panel)
        # Draw text after resize to avoid antialias blur from resampling.
        panel = add_badge(panel, f'GT: {len(gt)}')

        r = i // 3
        c = i % 3
        x = c * (TILE_W + GAP_X)
        y = r * (TILE_H + GAP_Y)
        canvas[y:y + TILE_H, x:x + TILE_W] = panel

    write_rgb(OUT, canvas)
    print(f'WROTE {OUT}')


if __name__ == '__main__':
    main()
