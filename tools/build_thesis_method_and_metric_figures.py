#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "thesis_overleaf" / "img"

MODEL_COLORS = {
    "DRENet": "#8C8C8C",
    "YOLO26": "#B0B0B0",
    "FCOS": "#D0D0D0",
}
METRIC_COLORS = {
    "Precision": "#566D7E",
    "Recall": "#6E7F63",
    "F1": "#8A6A46",
    "AP50": "#4F4F4F",
    "AP50-95": "#8C8C8C",
}
BG = "#FFFFFF"
TEXT = "#222222"
MUTED = "#555555"
LINE = "#222222"
BOX_FILL = "#F7F7F7"
GROUP_FILL = "#FBFBFB"
ACCENT_FILL = "#EFEFEF"

MAIN_RESULTS = {
    "DRENet": {"AP50": 0.7949, "AP50-95": 0.2919, "Precision": 0.4927, "Recall": 0.8511, "F1": 0.6241},
    "FCOS": {"AP50": 0.7389, "AP50-95": 0.2880, "Precision": 0.7621, "Recall": 0.8297, "F1": 0.7944},
    "YOLO26": {"AP50": 0.7950, "AP50-95": 0.3170, "Precision": 0.8430, "Recall": 0.7220, "F1": 0.7778},
}

THRESHOLD_RESULTS = {
    "DRENet": {
        "x": [0.15, 0.25, 0.35],
        "Precision": [0.6441, 0.7331, 0.7778],
        "Recall": [0.7935, 0.7663, 0.7355],
        "F1": [0.7110, 0.7493, 0.7561],
    },
    "FCOS": {
        "x": [0.15, 0.25, 0.35],
        "Precision": [0.7283, 0.7621, 0.7761],
        "Recall": [0.8351, 0.8297, 0.8225],
        "F1": [0.7781, 0.7944, 0.7986],
    },
    "YOLO26": {
        "x": [0.15, 0.25, 0.35],
        "Precision": [0.6412, 0.7527, 0.8248],
        "Recall": [0.7899, 0.7609, 0.6993],
        "F1": [0.7078, 0.7568, 0.7569],
    },
}

SIZE_RESULTS = {
    "FCOS": {
        "x": [512, 640, 800],
        "AP50": [0.4490, 0.6452, 0.7389],
        "Recall": [0.6395, 0.7808, 0.8297],
        "F1": [0.6139, 0.7601, 0.7944],
    },
    "YOLO26": {
        "x": [512, 640, 800],
        "AP50": [0.6661, 0.6737, 0.6564],
        "Recall": [0.7609, 0.7953, 0.7989],
        "F1": [0.7568, 0.7743, 0.7362],
    },
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Songti SC",
                "STSong",
                "SimSun",
                "STIX Two Text",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.facecolor": "white",
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.titleweight": "normal",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    fc: str = BOX_FILL,
    ec: str = LINE,
    lw: float = 1.2,
    dashed: bool = False,
    rounding: float = 0.001,
    fontsize: int = 11,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center", fontsize=fontsize)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=max(fontsize - 2, 8), color=TEXT)


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    text: str | None = None,
    color: str = LINE,
    dashed: bool = False,
    text_offset: tuple[float, float] = (0, 0),
    shrink_a: float = 0.5,
    shrink_b: float = 3.0,
    connection_style: str = "arc3,rad=0",
) -> None:
    """
    添加带箭头的连接线

    参数改进：
    - shrink_a/shrink_b 减小，使箭头更贴近框边界
    - 添加 connection_style 控制路径形状，默认直线
    """
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        linewidth=1.2,
        color=color,
        mutation_scale=16,
        linestyle="--" if dashed else "-",
        alpha=1.0,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        connectionstyle=connection_style,
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=9.0, color=TEXT, ha="center", va="center", wrap=True)


def add_polyline_with_final_arrow(
    ax,
    points: list[tuple[float, float]],
    color: str = LINE,
    lw: float = 1.2,
    final_shrink_a: float = 0.0,
    final_shrink_b: float = 6.0,
) -> None:
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw)
    add_arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        shrink_a=final_shrink_a,
        shrink_b=final_shrink_b,
    )


def add_pill(ax, x: float, y: float, w: float, h: float, text: str, fc: str, color: str = TEXT) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=0,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=color)


def polish_axes(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def add_entity_box(ax, x: float, y: float, w: float, h: float, title: str, fields: list[str], fontsize: int = 10) -> None:
    add_box(ax, x, y, w, h, "", fc=BOX_FILL, ec=LINE, fontsize=fontsize)
    header_h = 0.11 * h
    header = FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle="round,pad=0.012,rounding_size=0.001",
        linewidth=1.0,
        edgecolor=LINE,
        facecolor=ACCENT_FILL,
    )
    ax.add_patch(header)
    ax.text(x + w / 2, y + h - header_h / 2, title, ha="center", va="center", fontsize=11)
    top_text_y = y + h - header_h - 0.03
    line_gap = (h - header_h - 0.06) / max(len(fields), 1)
    for idx, field in enumerate(fields):
        ax.text(x + 0.04 * w, top_text_y - idx * line_gap, field, ha="left", va="top", fontsize=fontsize)


def build_drenet_overview(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    polish_axes(ax)
    top_y = 0.60
    top_h = 0.14
    xs = [0.05, 0.25, 0.45, 0.65, 0.85]
    ws = [0.14, 0.14, 0.14, 0.14, 0.10]
    labels = [
        ("Input image", "medium-resolution RS image"),
        ("Backbone", "multi-scale feature extraction"),
        ("CRMA block", "cross-stage attention"),
        ("Detection head", "classification and box regression"),
        ("Output", "ship detections"),
    ]
    for x, w, (title, subtitle) in zip(xs, ws, labels):
        add_box(ax, x, top_y, w, top_h, title, subtitle)

    for i in range(4):
        add_arrow(ax, (xs[i] + ws[i], top_y + top_h / 2), (xs[i + 1], top_y + top_h / 2))

    group_x, group_y, group_w, group_h = 0.22, 0.17, 0.70, 0.28
    add_box(ax, group_x, group_y, group_w, group_h, "", fc=GROUP_FILL, ec=LINE, dashed=True, fontsize=10)
    ax.text(group_x + group_w / 2, group_y + group_h - 0.03, "Train-only degraded reconstruction branch", ha="center", va="center", fontsize=10)

    sub_y = 0.24
    sub_h = 0.10
    sub_xs = [0.29, 0.53, 0.77]
    sub_w = 0.14
    sub_labels = [
        ("DRE branch", "degraded reconstruction"),
        ("Reconstructed map", "object-aware degraded image"),
        ("Supervision target", "degraded-image constraint"),
    ]
    for x, (title, subtitle) in zip(sub_xs, sub_labels):
        add_box(ax, x, sub_y, sub_w, sub_h, title, subtitle, fontsize=10)

    backbone_center_x = xs[1] + ws[1] / 2
    add_arrow(ax, (backbone_center_x, top_y), (backbone_center_x, sub_y + sub_h), text="shared features", text_offset=(0.08, 0.0))
    add_arrow(ax, (sub_xs[0] + sub_w, sub_y + sub_h / 2), (sub_xs[1], sub_y + sub_h / 2))
    add_arrow(ax, (sub_xs[1] + sub_w, sub_y + sub_h / 2), (sub_xs[2], sub_y + sub_h / 2), text="reconstruction loss", text_offset=(0.0, 0.04))
    add_arrow(ax, (sub_xs[1] + sub_w * 0.15, sub_y + sub_h + 0.02), (xs[2] + ws[2] * 0.15, top_y - 0.01), text="feature guidance", dashed=True, text_offset=(0.01, 0.035))

    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_fcos_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    polish_axes(ax)
    top_y = 0.58
    top_h = 0.14
    xs = [0.05, 0.23, 0.41, 0.59, 0.77]
    ws = [0.13, 0.13, 0.13, 0.13, 0.13]
    labels = [
        ("Input image", ""),
        ("Backbone", "feature extraction"),
        ("FPN", "multi-level features"),
        ("FCOS head", "shared conv tower"),
        ("Decoded boxes", "anchor-free outputs"),
    ]
    for x, w, (title, subtitle) in zip(xs, ws, labels):
        add_box(ax, x, top_y, w, top_h, title, subtitle)
    for i in range(4):
        add_arrow(ax, (xs[i] + ws[i], top_y + top_h / 2), (xs[i + 1], top_y + top_h / 2))

    branch_y = 0.24
    branch_h = 0.10
    branch_xs = [0.60, 0.73, 0.86]
    branch_w = 0.10
    branch_labels = [
        ("Classification", "class scores"),
        ("Regression", "l, t, r, b"),
        ("Centerness", "quality prior"),
    ]
    for x, (title, subtitle) in zip(branch_xs, branch_labels):
        add_box(ax, x, branch_y, branch_w, branch_h, title, subtitle, fontsize=10)

    head_anchor_x = xs[3] + ws[3] / 2
    head_bottom_y = top_y - 0.008
    target_centers = [x + branch_w / 2 for x in branch_xs]
    for tc in target_centers:
        add_arrow(ax, (head_anchor_x, head_bottom_y), (tc, branch_y + branch_h), shrink_a=0.0, shrink_b=6.0)

    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_yolo_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    polish_axes(ax)
    top_y = 0.58
    top_h = 0.14
    xs = [0.05, 0.23, 0.41, 0.59, 0.77]
    ws = [0.13, 0.13, 0.13, 0.13, 0.13]
    labels = [
        ("Input image", "512 main setting"),
        ("Backbone", "feature extraction"),
        ("Neck", "multi-scale fusion"),
        ("Detection head", "one-stage prediction"),
        ("Output", "NMS result"),
    ]
    for x, w, (title, subtitle) in zip(xs, ws, labels):
        add_box(ax, x, top_y, w, top_h, title, subtitle)
    for i in range(4):
        add_arrow(ax, (xs[i] + ws[i], top_y + top_h / 2), (xs[i + 1], top_y + top_h / 2))

    branch_y = 0.24
    branch_h = 0.10
    branch_xs = [0.60, 0.73, 0.86]
    branch_w = 0.10
    branch_labels = [
        ("Small scale", "fine targets"),
        ("Medium scale", "balanced scale"),
        ("Large scale", "context support"),
    ]
    for x, (title, subtitle) in zip(branch_xs, branch_labels):
        add_box(ax, x, branch_y, branch_w, branch_h, title, subtitle, fontsize=10)

    head_anchor_x = xs[3] + ws[3] / 2
    head_bottom_y = top_y - 0.008
    for tc in [x + branch_w / 2 for x in branch_xs]:
        add_arrow(ax, (head_anchor_x, head_bottom_y), (tc, branch_y + branch_h), shrink_a=0.0, shrink_b=6.0)

    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_ch1_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.3))
    polish_axes(ax)

    add_box(ax, 0.05, 0.68, 0.16, 0.13, "Data preparation", "dataset check and unified format")
    add_box(ax, 0.27, 0.68, 0.20, 0.13, "Model training and inference", "DRENet / YOLO26 / FCOS")
    add_box(ax, 0.53, 0.68, 0.16, 0.13, "Result analysis", "quantitative and qualitative study")
    add_box(ax, 0.75, 0.68, 0.18, 0.13, "System integration", "web deployment and demo")

    add_arrow(ax, (0.21, 0.745), (0.27, 0.745))
    add_arrow(ax, (0.47, 0.745), (0.53, 0.745))
    add_arrow(ax, (0.69, 0.745), (0.75, 0.745))

    add_box(ax, 0.12, 0.24, 0.20, 0.14, "Unified protocol", "data split / metrics / thresholds")
    add_box(ax, 0.40, 0.24, 0.20, 0.14, "Unified inference API", "single-model / ensemble")
    add_box(ax, 0.68, 0.24, 0.20, 0.14, "System evidence chain", "task replay / result review / history")

    add_arrow(ax, (0.29, 0.68), (0.22, 0.38), text="constrains training and evaluation", text_offset=(-0.02, 0.03))
    add_arrow(ax, (0.61, 0.68), (0.50, 0.38), text="feeds into unified interface", text_offset=(-0.02, 0.03))
    add_arrow(ax, (0.84, 0.68), (0.78, 0.38), text="service-based presentation", text_offset=(0.02, 0.03))
    add_arrow(ax, (0.32, 0.31), (0.40, 0.31))
    add_arrow(ax, (0.60, 0.31), (0.68, 0.31))

    add_box(ax, 0.24, 0.06, 0.52, 0.10, "Paper results, system demos and task history share the same protocol", fc=ACCENT_FILL, ec=LINE, fontsize=10)
    add_arrow(ax, (0.50, 0.24), (0.50, 0.16))

    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_ch5_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    polish_axes(ax)

    def add_group(ax, x: float, y: float, w: float, h: float, label: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.001",
            linewidth=1.2,
            edgecolor=LINE,
            facecolor=GROUP_FILL,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.01,
            y + h + 0.020,
            label,
            ha="left",
            va="bottom",
            fontsize=11,
            color=TEXT,
        )

    add_group(ax, 0.05, 0.72, 0.90, 0.14, "Frontend layer")
    frontend_xs = [0.09, 0.31, 0.53, 0.75]
    frontend_labels = [
        ("React + Vite + AntD", ""),
        ("Submit / Tasks", ""),
        ("Detail / Models", ""),
        ("Polling / Visualization", ""),
    ]
    for x, (title, subtitle) in zip(frontend_xs, frontend_labels):
        add_box(ax, x, 0.755, 0.16, 0.065, title, subtitle, fontsize=10)

    add_group(ax, 0.05, 0.43, 0.90, 0.18, "Backend layer")
    backend_xs = [0.09, 0.31, 0.53, 0.75]
    backend_labels = [
        ("FastAPI router", "/api/v1"),
        ("Task executor", "ThreadPoolExecutor"),
        ("Inference runtime", "unified predictor"),
        ("SQLAlchemy service", "task / result persistence"),
    ]
    for x, (title, subtitle) in zip(backend_xs, backend_labels):
        add_box(ax, x, 0.485, 0.16, 0.075, title, subtitle, fontsize=10)
    for i in range(3):
        add_arrow(ax, (backend_xs[i] + 0.16, 0.522), (backend_xs[i + 1], 0.522))
    add_arrow(ax, (0.50, 0.72), (0.50, 0.56))

    add_group(ax, 0.05, 0.12, 0.40, 0.16, "Model layer")
    model_xs = [0.10, 0.22, 0.34]
    for x, name in zip(model_xs, ["DRENet", "YOLO26", "FCOS"]):
        add_box(ax, x, 0.165, 0.08, 0.065, name, "", fontsize=10)

    add_group(ax, 0.53, 0.12, 0.42, 0.16, "Data and output layer")
    add_box(ax, 0.58, 0.165, 0.12, 0.065, "SQLite", "models / tasks / results", fontsize=10)
    add_box(ax, 0.76, 0.165, 0.14, 0.065, "outputs/tasks/<id>", "raw / vis / json artifacts", fontsize=10)

    runtime_center_x = backend_xs[2] + 0.08
    runtime_bottom_y = 0.485
    model_mid_x = 0.25
    storage_mid_x = 0.64
    storage_bottom_mid_x = 0.83

    route_y = 0.39
    add_polyline_with_final_arrow(
        ax,
        [
            (runtime_center_x, runtime_bottom_y),
            (runtime_center_x, route_y),
            (model_mid_x, route_y),
            (model_mid_x, 0.23),
        ],
    )

    add_polyline_with_final_arrow(
        ax,
        [
            (runtime_center_x, runtime_bottom_y),
            (runtime_center_x, route_y),
            (storage_mid_x, route_y),
            (storage_mid_x, 0.23),
        ],
    )

    service_center_x = backend_xs[3] + 0.08
    add_arrow(ax, (service_center_x, runtime_bottom_y), (service_center_x, 0.23))

    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_ch5_flow(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    polish_axes(ax)

    x = 0.30
    w = 0.40
    h = 0.07
    ys = [0.86, 0.74, 0.62, 0.50, 0.38, 0.26, 0.14]
    steps = [
        "1. User submits images and parameters",
        "2. FastAPI validates request and creates task row",
        "3. TaskExecutor dispatches local inference job",
        "4. Runtime invokes DRENet / YOLO26 / FCOS",
        "5. Results are written to DB and output directory",
        "6. Frontend polls task status and result endpoints",
        "7. User reviews progress, visualizations and results",
    ]
    for y, step in zip(ys, steps):
        add_box(ax, x, y, w, h, step, "", fontsize=10)
    for i in range(len(ys) - 1):
        add_arrow(ax, (x + w / 2, ys[i]), (x + w / 2, ys[i + 1] + h))

    add_box(ax, 0.05, 0.47, 0.18, 0.13, "State machine", "queued -> running\nrunning -> done\nrunning -> failed", fontsize=10)
    add_arrow(ax, (0.23, 0.53), (x, 0.535))

    add_box(ax, 0.77, 0.46, 0.18, 0.14, "Artifacts", "raw image\nvisualized image\njson + result rows", fontsize=10)
    add_arrow(ax, (x + w, 0.535), (0.77, 0.53))

    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_ch5_er(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    polish_axes(ax)

    add_entity_box(ax, 0.05, 0.55, 0.25, 0.28, "models", [
        "id (PK)",
        "name",
        "key",
        "weight_path",
        "is_enabled",
        "created_at",
    ])
    add_entity_box(ax, 0.36, 0.48, 0.25, 0.35, "tasks", [
        "id (PK)",
        "type",
        "status",
        "mode",
        "model_key",
        "score_thr",
        "input_count / done_count",
        "error_code / error_message",
        "created_at / started_at / finished_at",
    ], fontsize=9)
    add_entity_box(ax, 0.67, 0.55, 0.25, 0.28, "results", [
        "id (PK)",
        "task_id (FK)",
        "image_name",
        "source_model / is_fused",
        "bbox_x1 ... bbox_y2",
        "score",
        "category_id",
    ])
    add_entity_box(ax, 0.51, 0.12, 0.28, 0.22, "task_files", [
        "id (PK)",
        "task_id (FK)",
        "kind (input / output / vis)",
        "path",
    ])

    add_arrow(ax, (0.30, 0.69), (0.36, 0.69), text="config source", text_offset=(0.0, 0.04))
    add_arrow(ax, (0.61, 0.64), (0.67, 0.64), text="1 to many", text_offset=(0.0, 0.04))
    add_arrow(ax, (0.49, 0.48), (0.60, 0.34), text="1 to many", text_offset=(0.03, 0.04))

    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_main_comparison_chart(out_path: Path) -> None:
    metrics = ["AP50", "AP50-95", "Precision", "Recall", "F1"]
    models = list(MAIN_RESULTS.keys())
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("white")

    for idx, model in enumerate(models):
        offsets = x + (idx - 1) * width
        vals = [MAIN_RESULTS[model][m] for m in metrics]
        bars = ax.bar(offsets, vals, width=width, label=model, color=MODEL_COLORS[model], edgecolor="white", linewidth=1.0)
        for bar, raw in zip(bars, vals):
            cx = bar.get_x() + bar.get_width() / 2
            label = f"{raw:.3f}"
            ax.text(cx, raw + 0.018, label, ha="center", va="bottom", fontsize=8.5, color=TEXT, rotation=90)

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


LINE_LABEL_OFFSETS = {
    "Precision": (-14, 10),
    "Recall": (12, 10),
    "F1": (0, -14),
    "AP50": (-14, 10),
}


THRESHOLD_LABEL_OFFSETS = {
    "DRENet": {
        "Precision": [(-14, 10), (-16, 12), (8, 10)],
        "Recall": [(0, 12), (14, 14), (12, 6)],
        "F1": [(0, -14), (0, -18), (0, -14)],
    },
    "FCOS": {
        "Precision": [(-14, 10), (-18, 16), (-8, 10)],
        "Recall": [(12, 12), (16, 18), (12, 12)],
        "F1": [(-4, -16), (8, -18), (8, -14)],
    },
    "YOLO26": {
        "Precision": [(-18, 10), (-16, 12), (-12, 10)],
        "Recall": [(10, 12), (18, 18), (12, 10)],
        "F1": [(0, -16), (0, -18), (0, -16)],
    },
}


def _plot_metric_lines(ax, xs: list[float], data: dict[str, list[float]], title: str, xlabel: str, ylabel: str, label_offsets: dict[str, list[tuple[int, int]]] | None = None) -> None:
    for metric in ["Precision", "Recall", "F1"]:
        ax.plot(xs, data[metric], marker="o", markersize=6, linewidth=2.2, color=METRIC_COLORS[metric], label=metric)
        for idx, (x, y) in enumerate(zip(xs, data[metric])):
            if label_offsets is None:
                dx, dy = LINE_LABEL_OFFSETS[metric]
            else:
                dx, dy = label_offsets[metric][idx]
            ax.annotate(
                f"{y:.3f}",
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.0,
                ha="center",
                va="center",
                color=METRIC_COLORS[metric],
                clip_on=False,
                bbox={"facecolor": BG, "edgecolor": "none", "pad": 0.12, "alpha": 0.95},
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.60, 0.88)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_threshold_trend(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, model in zip(axes, ["DRENet", "FCOS", "YOLO26"]):
        ax.set_facecolor("white")
        _plot_metric_lines(
            ax,
            THRESHOLD_RESULTS[model]["x"],
            THRESHOLD_RESULTS[model],
            model,
            "Confidence threshold",
            "Score",
            label_offsets=THRESHOLD_LABEL_OFFSETS[model],
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.00), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_size_trend(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, model in zip(axes, ["FCOS", "YOLO26"]):
        ax.set_facecolor("white")
        xs = SIZE_RESULTS[model]["x"]
        for metric in ["AP50", "Recall", "F1"]:
            ax.plot(xs, SIZE_RESULTS[model][metric], marker="o", markersize=6, linewidth=2.3, color=METRIC_COLORS[metric], label=metric)
            for x, y in zip(xs, SIZE_RESULTS[model][metric]):
                dx, dy = LINE_LABEL_OFFSETS[metric]
                ax.annotate(
                    f"{y:.3f}",
                    (x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.2,
                    ha="center",
                    va="center",
                    color=METRIC_COLORS[metric],
                    clip_on=False,
                    bbox={"facecolor": BG, "edgecolor": "none", "pad": 0.12, "alpha": 0.95},
                )
        ax.set_title(model)
        ax.set_xlabel("Inference image size")
        ax.set_xticks(xs)
        ax.set_ylim(0.40, 0.88)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.00), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


EFFICIENCY_RESULTS = {
    "DRENet": {"FPS": 121.90, "Params(M)": 4.79, "FLOPs(G)": 4.21},
    "FCOS": {"FPS": 45.96, "Params(M)": 32.17, "FLOPs(G)": 123.0},
    "YOLO26": {"FPS": 69.06, "Params(M)": 2.50, "FLOPs(G)": 3.69},
}


def build_ch2_research_map(out_path: Path) -> None:
    """第二章方法脉络图：三条研究路线汇聚到本文切入点。"""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    polish_axes(ax)

    top_y = 0.72
    top_h = 0.12
    add_box(ax, 0.04, top_y, 0.28, top_h, "General object detection", "two-stage / one-stage / anchor-free / transformer")
    add_box(ax, 0.36, top_y, 0.28, top_h, "Small object detection", "multi-scale fusion / loss re-weighting / post-processing")
    add_box(ax, 0.68, top_y, 0.28, top_h, "Remote sensing ship detection", "LEVIR-Ship / specialized methods (DRENet)")

    mid_y = 0.48
    mid_h = 0.10
    gen_xs = [0.06, 0.16, 0.26]
    gen_labels = [("Faster R-CNN", ""), ("YOLO / RetinaNet", ""), ("FCOS / DETR", "")]
    for x, (t, s) in zip(gen_xs, gen_labels):
        add_box(ax, x, mid_y, 0.08, mid_h, t, s, fontsize=9)

    small_xs = [0.38, 0.48, 0.58]
    small_labels = [("FPN / SCRDet", ""), ("Focal Loss", ""), ("SNIP / NMS", "")]
    for x, (t, s) in zip(small_xs, small_labels):
        add_box(ax, x, mid_y, 0.08, mid_h, t, s, fontsize=9)

    ship_xs = [0.70, 0.80]
    ship_labels = [("LEVIR-Ship", "dataset"), ("DRENet", "specialized method")]
    for x, (t, s) in zip(ship_xs, ship_labels):
        add_box(ax, x, mid_y, 0.08, mid_h, t, s, fontsize=9)

    for x in gen_xs:
        add_arrow(ax, (x + 0.04, top_y), (x + 0.04, mid_y + mid_h), shrink_a=2, shrink_b=4)
    for x in small_xs:
        add_arrow(ax, (x + 0.04, top_y), (x + 0.04, mid_y + mid_h), shrink_a=2, shrink_b=4)
    for x in ship_xs:
        add_arrow(ax, (x + 0.04, top_y), (x + 0.04, mid_y + mid_h), shrink_a=2, shrink_b=4)

    bot_y = 0.18
    bot_h = 0.14
    add_box(ax, 0.20, bot_y, 0.60, bot_h, "Our entry point", "multi-model comparison under unified protocol + system implementation", fc=ACCENT_FILL, ec=LINE)

    add_arrow(ax, (0.16, mid_y), (0.35, bot_y + bot_h), text="cross-paradigm comparison basis", text_offset=(-0.04, 0.03))
    add_arrow(ax, (0.48, mid_y), (0.50, bot_y + bot_h), text="performance gap analysis", text_offset=(0.0, 0.03))
    add_arrow(ax, (0.78, mid_y), (0.65, bot_y + bot_h), text="tiny ship scenario focus", text_offset=(0.04, 0.03))

    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_ch3_preprocess_flow(out_path: Path) -> None:
    """第三章数据预处理与统一格式流程图。"""
    fig, ax = plt.subplots(figsize=(12, 4.0))
    polish_axes(ax)

    top_y = 0.55
    top_h = 0.16
    xs = [0.04, 0.22, 0.40, 0.58, 0.76]
    ws = [0.14, 0.14, 0.14, 0.14, 0.14]
    labels = [
        ("Verify images & labels", "one-to-one check\nempty labels / abnormal coords"),
        ("COCO intermediate format", "unified annotation format\ncross-framework sharing"),
        ("Fixed data split", "train / val / test\nfixed sample lists"),
        ("Unified output structure", "image_id / bbox\nscore / category_id"),
        ("System-callable", "inference API / database\nvisualization / replay"),
    ]
    for x, w, (title, subtitle) in zip(xs, ws, labels):
        add_box(ax, x, top_y, w, top_h, title, subtitle, fontsize=10)

    for i in range(4):
        add_arrow(ax, (xs[i] + ws[i], top_y + top_h / 2), (xs[i + 1], top_y + top_h / 2))

    add_box(ax, 0.15, 0.12, 0.70, 0.10, "Unified protocol constraint: differences come from detection paradigms and implementation strategies, not from data organization or script details", fc=ACCENT_FILL, ec=LINE, fontsize=10)
    add_arrow(ax, (0.50, top_y), (0.50, 0.22))

    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_ch4_efficiency_chart(out_path: Path) -> None:
    """第四章效率与复杂度对比柱状图。"""
    metrics = ["FPS", "Params(M)", "FLOPs(G)"]
    models = list(EFFICIENCY_RESULTS.keys())
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("white")

    for idx, model in enumerate(models):
        offsets = x + (idx - 1) * width
        vals = [EFFICIENCY_RESULTS[model][m] for m in metrics]
        bars = ax.bar(offsets, vals, width=width, label=model, color=MODEL_COLORS[model], edgecolor="white", linewidth=1.0)
        for bar, raw in zip(bars, vals):
            cx = bar.get_x() + bar.get_width() / 2
            label = f"{raw:.1f}" if raw >= 10 else f"{raw:.2f}"
            ax.text(cx, raw + 1.5, label, ha="center", va="bottom", fontsize=8.5, color=TEXT)

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_ch4_radar_chart(out_path: Path) -> None:
    """第四章多维度雷达图：综合展示三模型在精度、效率、部署上的表现。"""
    radar_metrics = ["AP50", "AP50-95", "Precision", "Recall", "F1", "FPS", "Params eff", "FLOPs eff"]
    models = list(MAIN_RESULTS.keys())

    # 归一化参数：效率指标用逆指标（越小越优），映射到 [0.1, 1.0] 避免最差模型为 0
    max_params = max(eff["Params(M)"] for eff in EFFICIENCY_RESULTS.values())
    min_params = min(eff["Params(M)"] for eff in EFFICIENCY_RESULTS.values())
    max_flops = max(eff["FLOPs(G)"] for eff in EFFICIENCY_RESULTS.values())
    min_flops = min(eff["FLOPs(G)"] for eff in EFFICIENCY_RESULTS.values())
    max_fps = max(eff["FPS"] for eff in EFFICIENCY_RESULTS.values())

    radar_data = {}
    for m in models:
        main = MAIN_RESULTS[m]
        eff = EFFICIENCY_RESULTS[m]
        precision_val = main["Precision"]
        f1_val = main["F1"]
        fps_norm = eff["FPS"] / max_fps
        # 逆指标归一化到 [0.1, 1.0]：最优=1.0，最差=0.1
        params_range = max_params - min_params if max_params != min_params else 1.0
        flops_range = max_flops - min_flops if max_flops != min_flops else 1.0
        params_eff = 0.1 + 0.9 * (1.0 - (eff["Params(M)"] - min_params) / params_range)
        flops_eff = 0.1 + 0.9 * (1.0 - (eff["FLOPs(G)"] - min_flops) / flops_range)
        radar_data[m] = [
            main["AP50"], main["AP50-95"], precision_val, main["Recall"], f1_val,
            fps_norm, params_eff, flops_eff,
        ]

    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG)

    for model in models:
        values = radar_data[model]
        values += values[:1]
        ax.plot(angles, values, linewidth=2.0, color=MODEL_COLORS[model], label=model)
        ax.fill(angles, values, alpha=0.15, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color=MUTED)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1.10))

    fig.tight_layout(pad=2)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_matplotlib()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    build_ch1_pipeline(OUTDIR / "ch1_pipeline.png")
    build_drenet_overview(OUTDIR / "ch3_drenet_overview.png")
    build_fcos_pipeline(OUTDIR / "ch3_fcos_pipeline.png")
    build_yolo_pipeline(OUTDIR / "ch3_yolo_pipeline.png")
    build_main_comparison_chart(OUTDIR / "ch4_main_comparison_chart.png")
    build_threshold_trend(OUTDIR / "ch4_threshold_trend.png")
    build_size_trend(OUTDIR / "ch4_size_trend.png")
    build_ch5_architecture(OUTDIR / "ch5_architecture.png")
    build_ch5_flow(OUTDIR / "ch5_flow.png")
    build_ch5_er(OUTDIR / "ch5_er.png")
    # New figures for enriching thesis
    build_ch2_research_map(OUTDIR / "ch2_research_map.png")
    build_ch3_preprocess_flow(OUTDIR / "ch3_preprocess_flow.png")
    build_ch4_efficiency_chart(OUTDIR / "ch4_efficiency_chart.png")
    build_ch4_radar_chart(OUTDIR / "ch4_radar_chart.png")
    print("Generated thesis method and metric figures in", OUTDIR)


if __name__ == "__main__":
    main()
