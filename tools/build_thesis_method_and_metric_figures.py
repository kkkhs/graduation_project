#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "thesis_overleaf" / "figures" / "generated"

MODEL_COLORS = {
    "DRENet": "#E98B8B",
    "YOLO26": "#7AA7FF",
    "FCOS": "#7BC8A4",
}
METRIC_COLORS = {
    "Precision": "#3B82F6",
    "Recall": "#10B981",
    "F1": "#F59E0B",
    "AP50": "#7C3AED",
    "AP50-95": "#EC4899",
}
BG = "#F7F9FC"
TEXT = "#1F2937"
MUTED = "#6B7280"
LINE = "#475569"

MAIN_RESULTS = {
    "DRENet": {"AP50": 0.7949, "AP50-95": 0.2919, "Precision": 0.4927, "Recall": 0.8511, "F1": 0.6241},
    "FCOS": {"AP50": 0.7700, "AP50-95": 0.2850, "Precision": None, "Recall": 0.4050, "F1": None},
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
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def add_box(ax, x: float, y: float, w: float, h: float, title: str, subtitle: str = "", fc: str = "#FFFFFF", ec: str = "#CBD5E1", lw: float = 1.6, dashed: bool = False, rounding: float = 0.025, fontsize: int = 12) -> None:
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
    ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center", fontsize=fontsize, fontweight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=max(fontsize - 2, 8), color=MUTED)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], text: str | None = None, color: str = LINE, dashed: bool = False, text_offset: tuple[float, float] = (0, 0)) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        linewidth=1.5,
        color=color,
        mutation_scale=18,
        linestyle="--" if dashed else "-",
        alpha=0.95,
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=9.5, color=MUTED, ha="center", va="center")


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
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, fontweight="bold", color=color)


def polish_axes(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def build_drenet_overview(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    polish_axes(ax)
    ax.text(0.5, 0.94, "DRENet overview for tiny ship detection", ha="center", fontsize=20, fontweight="bold")
    add_pill(ax, 0.10, 0.84, 0.16, 0.06, "Inference path", "#DBEAFE", color="#1D4ED8")
    add_pill(ax, 0.10, 0.16, 0.15, 0.06, "Train-only branch", "#FDE68A", color="#92400E")

    add_box(ax, 0.05, 0.58, 0.15, 0.17, "MR RS image", "tiny ships + complex sea", fc="#FFFFFF")
    add_box(ax, 0.26, 0.58, 0.17, 0.17, "Backbone", "shared multi-scale features", fc="#FCE7F3", ec="#F9A8D4")
    add_box(ax, 0.49, 0.58, 0.16, 0.17, "CRMA block", "cross-stage attention", fc="#E0F2FE", ec="#7DD3FC")
    add_box(ax, 0.71, 0.58, 0.14, 0.17, "Detector", "classification + box regression", fc="#DCFCE7", ec="#86EFAC")
    add_box(ax, 0.88, 0.58, 0.08, 0.17, "Output", "ship boxes", fc="#FFFFFF")

    add_arrow(ax, (0.20, 0.665), (0.26, 0.665))
    add_arrow(ax, (0.43, 0.665), (0.49, 0.665))
    add_arrow(ax, (0.65, 0.665), (0.71, 0.665))
    add_arrow(ax, (0.85, 0.665), (0.88, 0.665))

    add_box(ax, 0.27, 0.26, 0.17, 0.15, "DRE branch", "reconstruction enhancer", fc="#FEF3C7", ec="#F59E0B")
    add_box(ax, 0.52, 0.26, 0.16, 0.15, "Reconstructed map", "object-aware degraded image", fc="#FFF7ED", ec="#FDBA74")
    add_box(ax, 0.76, 0.26, 0.15, 0.15, "Degraded target", "selective blur supervision", fc="#FFFFFF", ec="#F59E0B", dashed=True)

    add_arrow(ax, (0.345, 0.58), (0.345, 0.41), text="shared features", text_offset=(0.06, 0.0))
    add_arrow(ax, (0.44, 0.335), (0.52, 0.335))
    add_arrow(ax, (0.68, 0.335), (0.76, 0.335), text="reconstruction loss", text_offset=(0.00, 0.05))
    add_arrow(ax, (0.52, 0.44), (0.43, 0.54), text="feature guidance", dashed=True, text_offset=(0.01, 0.05))

    ax.text(0.5, 0.08, "The reconstruction branch is used only during training; inference keeps the detector path lightweight.", ha="center", fontsize=11, color=MUTED)
    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_fcos_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    polish_axes(ax)
    ax.text(0.5, 0.94, "FCOS anchor-free detection pipeline", ha="center", fontsize=20, fontweight="bold")
    add_pill(ax, 0.08, 0.84, 0.18, 0.06, "Reference baseline", "#D1FAE5", color="#047857")

    add_box(ax, 0.05, 0.54, 0.13, 0.18, "Input image", "single test sample", fc="#FFFFFF")
    add_box(ax, 0.23, 0.54, 0.16, 0.18, "Backbone", "feature extraction", fc="#EDE9FE", ec="#C4B5FD")
    add_box(ax, 0.44, 0.54, 0.16, 0.18, "FPN", "P3 / P4 / P5 levels", fc="#DBEAFE", ec="#93C5FD")
    add_box(ax, 0.66, 0.54, 0.14, 0.18, "FCOS head", "shared conv towers", fc="#DCFCE7", ec="#86EFAC")
    add_box(ax, 0.84, 0.54, 0.11, 0.18, "Decode", "anchor-free boxes", fc="#FFFFFF")

    add_arrow(ax, (0.18, 0.63), (0.23, 0.63))
    add_arrow(ax, (0.39, 0.63), (0.44, 0.63))
    add_arrow(ax, (0.60, 0.63), (0.66, 0.63))
    add_arrow(ax, (0.80, 0.63), (0.84, 0.63))

    add_box(ax, 0.61, 0.24, 0.11, 0.12, "Cls branch", "class score", fc="#EFF6FF", ec="#60A5FA", fontsize=10)
    add_box(ax, 0.74, 0.24, 0.11, 0.12, "Reg branch", "l,t,r,b", fc="#F0FDF4", ec="#4ADE80", fontsize=10)
    add_box(ax, 0.87, 0.24, 0.10, 0.12, "Center-ness", "quality prior", fc="#FFF7ED", ec="#FB923C", fontsize=10)
    add_arrow(ax, (0.73, 0.54), (0.665, 0.36))
    add_arrow(ax, (0.73, 0.54), (0.795, 0.36))
    add_arrow(ax, (0.73, 0.54), (0.92, 0.36))

    ax.text(0.16, 0.30, "No anchor presets", fontsize=13, fontweight="bold", color="#047857")
    ax.text(0.16, 0.24, "Each location predicts class, box offsets and centerness directly.", fontsize=10.5, color=MUTED)
    ax.text(0.5, 0.08, "FCOS reduces anchor hyper-parameters and serves as the anchor-free comparison route in this thesis.", ha="center", fontsize=11, color=MUTED)
    fig.tight_layout(pad=1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_yolo_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    polish_axes(ax)
    ax.text(0.5, 0.94, "YOLO26 one-stage deployment pipeline", ha="center", fontsize=20, fontweight="bold")
    add_pill(ax, 0.08, 0.84, 0.22, 0.06, "Lightweight deployment route", "#DBEAFE", color="#1D4ED8")

    add_box(ax, 0.05, 0.54, 0.12, 0.18, "Input image", "512 main line", fc="#FFFFFF")
    add_box(ax, 0.22, 0.54, 0.16, 0.18, "Backbone", "compact feature extractor", fc="#E0F2FE", ec="#7DD3FC")
    add_box(ax, 0.43, 0.54, 0.17, 0.18, "Neck", "PAN / FPN fusion", fc="#EDE9FE", ec="#C4B5FD")
    add_box(ax, 0.65, 0.54, 0.15, 0.18, "Detection head", "one-stage predictions", fc="#DCFCE7", ec="#86EFAC")
    add_box(ax, 0.85, 0.54, 0.10, 0.18, "Output", "NMS result", fc="#FFFFFF")

    add_arrow(ax, (0.17, 0.63), (0.22, 0.63))
    add_arrow(ax, (0.38, 0.63), (0.43, 0.63))
    add_arrow(ax, (0.60, 0.63), (0.65, 0.63))
    add_arrow(ax, (0.80, 0.63), (0.85, 0.63))

    add_box(ax, 0.60, 0.24, 0.10, 0.12, "Small", "fine targets", fc="#EFF6FF", ec="#60A5FA", fontsize=10)
    add_box(ax, 0.72, 0.24, 0.10, 0.12, "Medium", "balanced scale", fc="#F5F3FF", ec="#A78BFA", fontsize=10)
    add_box(ax, 0.84, 0.24, 0.10, 0.12, "Large", "context support", fc="#F0FDF4", ec="#4ADE80", fontsize=10)
    add_arrow(ax, (0.725, 0.54), (0.65, 0.36))
    add_arrow(ax, (0.725, 0.54), (0.77, 0.36))
    add_arrow(ax, (0.725, 0.54), (0.89, 0.36))

    ax.text(0.12, 0.30, "One forward pass", fontsize=13, fontweight="bold", color="#1D4ED8")
    ax.text(0.12, 0.24, "The model jointly balances detection accuracy, speed and integration simplicity.", fontsize=10.5, color=MUTED)
    ax.text(0.5, 0.08, "YOLO26 is used as the one-stage representative and the strongest default-model candidate for the web system.", ha="center", fontsize=11, color=MUTED)
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
        draw_vals = [0 if v is None else v for v in vals]
        bars = ax.bar(offsets, draw_vals, width=width, label=model, color=MODEL_COLORS[model], edgecolor="white", linewidth=1.0)
        for bar, raw in zip(bars, vals):
            cx = bar.get_x() + bar.get_width() / 2
            if raw is None:
                bar.set_facecolor("#E5E7EB")
                bar.set_hatch("///")
                bar.set_edgecolor("#9CA3AF")
                ax.text(cx, 0.03, "N/A", ha="center", va="bottom", fontsize=9, color=MUTED, rotation=90)
            else:
                label = f"{raw:.3f}"
                if model == "FCOS" and metrics[list(bars).index(bar)] == "Recall":
                    label += "*"
                ax.text(cx, raw + 0.018, label, ha="center", va="bottom", fontsize=8.5, color=TEXT, rotation=90)

    ax.set_title("Main comparison across models", pad=12)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.text(0.99, -0.14, "* FCOS recall uses AR@100 from the formal run; Precision and F1 are unavailable in the current main-table record.", transform=ax.transAxes, ha="right", va="top", fontsize=9.2, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_lines(ax, xs: list[float], data: dict[str, list[float]], title: str, xlabel: str, ylabel: str) -> None:
    for metric in ["Precision", "Recall", "F1"]:
        ax.plot(xs, data[metric], marker="o", markersize=6, linewidth=2.2, color=METRIC_COLORS[metric], label=metric)
        for x, y in zip(xs, data[metric]):
            ax.text(x, y + 0.012, f"{y:.3f}", fontsize=8.5, ha="center", color=METRIC_COLORS[metric])
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
        _plot_metric_lines(ax, THRESHOLD_RESULTS[model]["x"], THRESHOLD_RESULTS[model], model, "Confidence threshold", "Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Threshold ablation trends", fontsize=18, fontweight="bold", y=1.08)
    fig.text(0.5, -0.01, "All three models show the expected trade-off: higher confidence improves Precision while Recall gradually decreases.", ha="center", fontsize=10.5, color=MUTED)
    fig.tight_layout()
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
                ax.text(x, y + 0.012, f"{y:.3f}", fontsize=8.2, ha="center", color=METRIC_COLORS[metric])
        ax.set_title(model)
        ax.set_xlabel("Inference image size")
        ax.set_xticks(xs)
        ax.set_ylim(0.40, 0.88)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Input-size sensitivity", fontsize=18, fontweight="bold", y=1.08)
    fig.text(0.5, -0.01, "FCOS benefits steadily from larger inputs, while YOLO26 reaches its most balanced point at 640 before larger inputs introduce extra false positives.", ha="center", fontsize=10.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_matplotlib()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    build_drenet_overview(OUTDIR / "ch3_drenet_overview.png")
    build_fcos_pipeline(OUTDIR / "ch3_fcos_pipeline.png")
    build_yolo_pipeline(OUTDIR / "ch3_yolo_pipeline.png")
    build_main_comparison_chart(OUTDIR / "ch4_main_comparison_chart.png")
    build_threshold_trend(OUTDIR / "ch4_threshold_trend.png")
    build_size_trend(OUTDIR / "ch4_size_trend.png")
    print("Generated thesis method and metric figures in", OUTDIR)


if __name__ == "__main__":
    main()
