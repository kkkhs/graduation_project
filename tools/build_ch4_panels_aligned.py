from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/Users/khs/codes/graduation_project')
IMG_DIR = ROOT / 'experiment_assets/datasets/LEVIR-Ship/test/images'
LBL_DIR = ROOT / 'experiment_assets/datasets/LEVIR-Ship/test/labels'
GEN = ROOT / 'thesis_overleaf/figures/generated'

PRED_PATHS = {
    'DRENet': ROOT / 'experiment_assets/ablation/threshold/drenet/conf025/predictions.json',
    'YOLO26': ROOT / 'experiment_assets/ablation/threshold/yolo/conf025/predictions.json',
    'FCOS': ROOT / 'experiment_assets/ablation/threshold/mmdet_fcos/conf025_native/predictions.json',
}

MODEL_ORDER = ['DRENet', 'YOLO26', 'FCOS']
CATS = ['success', 'miss', 'false_positive']

TILE_W = 520
TILE_H = 520
LEFT_W = 130
TOP_H = 74
GAP_X = 20
GAP_Y = 16

COLORS = {
    'Label': (80, 220, 80),
    'DRENet': (255, 90, 90),
    'YOLO26': (70, 170, 255),
    'FCOS': (255, 200, 70),
}

FONT_PATH = Path('/System/Library/Fonts/Supplemental/Times New Roman.ttf')
# Keep figure text smaller and visually consistent with thesis figure style.
FONT_SIZE_ROW = 48
FONT_SIZE_COL = 44


def load_font(size):
    try:
        return ImageFont.truetype(str(FONT_PATH), size=size)
    except OSError:
        return ImageFont.load_default()


def text_size(text, font):
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1), (0, 0, 0)))
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def draw_text_on_rgb(rgb, pos, text, font, fill):
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=fill)
    return np.array(pil)


def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_rgb(p):
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(p, rgb):
    p.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(p), bgr)


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


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def get_gt_boxes(file_name):
    p = LBL_DIR / file_name.replace('.png', '.txt')
    boxes = []
    if not p.exists():
        return boxes
    # all tiles are 512 in this dataset split
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


def get_pred_boxes(pred_dict, file_name):
    arr = pred_dict.get(file_name, [])
    out = []
    for r in arr:
        b = r.get('bbox')
        if not b or len(b) != 4:
            continue
        x, y, w, h = map(float, b)
        out.append([x, y, x + w, y + h, float(r.get('score', 0.0))])
    out.sort(key=lambda z: z[4], reverse=True)
    return out


def eval_one(gt_boxes, pred_boxes):
    used = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    for pb in pred_boxes:
        bb = pb[:4]
        best_i = -1
        best = 0.0
        for i, gb in enumerate(gt_boxes):
            if used[i]:
                continue
            v = iou(bb, gb)
            if v > best:
                best = v
                best_i = i
        if best >= 0.5 and best_i >= 0:
            used[best_i] = True
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def add_badge(img, text):
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    bw, bh = tw + 16, th + 14
    cv2.rectangle(out, (8, 8), (8 + bw, 8 + bh), (0, 0, 0), -1)
    cv2.rectangle(out, (8, 8), (8 + bw, 8 + bh), (255, 255, 255), 1)
    cv2.putText(out, text, (16, 8 + bh - 6), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def draw_boxes(img, boxes, color):
    out = img.copy()
    t = 2
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        x1 = max(0, min(out.shape[1] - 1, x1))
        x2 = max(0, min(out.shape[1] - 1, x2))
        y1 = max(0, min(out.shape[0] - 1, y1))
        y2 = max(0, min(out.shape[0] - 1, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
    return out


def resize(img):
    return cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_CUBIC)


def choose_case(rows, cat, used_files):
    # rows: list[(file_name, tp, fp, fn, gt, pred_n)] for one model
    if cat == 'success':
        # strict success first: no fp/fn, at least one tp
        cand = [r for r in rows if r[1] > 0 and r[2] == 0 and r[3] == 0]
        if not cand:
            cand = [r for r in rows if r[1] > 0 and r[3] == 0]
        # prefer clearer panels with moderate object count (avoid too dense mosaics)
        cand.sort(key=lambda r: (abs(r[4] - 2), abs(r[5] - 2), -r[1], r[0]))
    elif cat == 'miss':
        cand = [r for r in rows if r[3] > 0 and r[4] > 0]
        # prefer complete misses first (tp=0), then fewer confounders (fp)
        cand.sort(key=lambda r: (0 if r[1] == 0 else 1, abs(r[3] - 1), r[2], abs(r[4] - 2), r[0]))
    else:
        cand = [r for r in rows if r[2] > 0]
        # prefer pure fp (gt=0), then larger fp
        cand.sort(key=lambda r: (0 if r[4] == 0 else 1, 0 if r[3] == 0 else 1, abs(r[2] - 2), r[0]))

    if not cand:
        return None
    for r in cand:
        if r[0] not in used_files:
            return r
    return cand[0]


def build_samples(preds):
    all_files = sorted(set(preds['DRENet'].keys()) | set(preds['YOLO26'].keys()) | set(preds['FCOS'].keys()))
    per_model_rows = {m: [] for m in MODEL_ORDER}

    for fn in all_files:
        gt = get_gt_boxes(fn)
        for m in MODEL_ORDER:
            pb = get_pred_boxes(preds[m], fn)
            tp, fp, fnn = eval_one(gt, pb)
            per_model_rows[m].append((fn, tp, fp, fnn, len(gt), len(pb)))

    selected = {cat: {} for cat in CATS}
    for cat in CATS:
        used_files = set()
        for m in MODEL_ORDER:
            rows = per_model_rows[m]
            pick = choose_case(rows, cat, used_files)
            if pick is None:
                continue
            used_files.add(pick[0])
            selected[cat][m] = {
                'file': pick[0],
                'tp': pick[1],
                'fp': pick[2],
                'fn': pick[3],
                'gt': pick[4],
                'pred_n': pick[5],
            }

    return selected


def render_case_figure(cat, selected_info, preds):
    # 3x2 grid:
    #   columns = model names
    #   rows    = Label / Prediction
    row_font = load_font(FONT_SIZE_ROW)
    col_font = load_font(FONT_SIZE_COL)
    row_names = ['Label', 'Prediction']
    row_widths = [text_size(n, row_font)[0] for n in row_names]
    left_w = max(LEFT_W, max(row_widths) + 24)

    total_w = left_w + TILE_W * 3 + GAP_X * 2
    total_h = TOP_H + TILE_H * 2 + GAP_Y
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    # row titles on the left
    for row_name, row_y in [('Label', TOP_H + TILE_H // 2), ('Prediction', TOP_H + TILE_H + GAP_Y + TILE_H // 2)]:
        tw, th = text_size(row_name, row_font)
        tx = (left_w - tw) // 2
        ty = row_y - th // 2 - 1
        canvas = draw_text_on_rgb(canvas, (tx, ty), row_name, row_font, fill=(25, 25, 25))

    for i, m in enumerate(MODEL_ORDER):
        info = selected_info[m]
        fn = info['file']
        base = read_rgb(IMG_DIR / fn)
        gt = get_gt_boxes(fn)
        pb = get_pred_boxes(preds[m], fn)
        label_img = resize(draw_boxes(base, gt, COLORS['Label']))
        pred_img = resize(draw_boxes(base, pb, COLORS[m]))

        x = left_w + i * (TILE_W + GAP_X)
        y_label = TOP_H
        y_pred = TOP_H + TILE_H + GAP_Y
        canvas[y_label:y_label + TILE_H, x:x + TILE_W] = label_img
        canvas[y_pred:y_pred + TILE_H, x:x + TILE_W] = pred_img

        # column title (model name)
        tw, th = text_size(m, col_font)
        tx = x + (TILE_W - tw) // 2
        ty = (TOP_H - th) // 2 + 1
        canvas = draw_text_on_rgb(canvas, (tx, ty), m, col_font, fill=(20, 20, 20))

    out = GEN / f'ch4_{cat}_cases.png'
    write_rgb(out, canvas)
    print(f'WROTE {out}')


def main():
    preds = {k: load_json(v) for k, v in PRED_PATHS.items()}
    selected = build_samples(preds)

    print('Selected samples:')
    for cat in CATS:
        print(cat, selected[cat])
        if all(m in selected[cat] for m in MODEL_ORDER):
            render_case_figure(cat, selected[cat], preds)
        else:
            print(f'SKIP {cat}: missing model selections')


if __name__ == '__main__':
    main()
