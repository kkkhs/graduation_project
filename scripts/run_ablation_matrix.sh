#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/models.yaml}"
IMAGES_DIR="${IMAGES_DIR:-$ROOT_DIR/experiment_assets/datasets/LEVIR-Ship/test/images}"
LABELS_DIR="${LABELS_DIR:-$ROOT_DIR/experiment_assets/datasets/LEVIR-Ship/test/labels}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/experiment_assets/ablation}"
ABLATION_MODE="${ABLATION_MODE:-all}"

mkdir -p "$OUT_ROOT"

run_eval() {
  local group="$1"
  local model="$2"
  local setting="$3"
  local conf="$4"
  local iou="$5"
  local imgsz="$6"

  local out_dir="$OUT_ROOT/$group/$model/$setting"
  mkdir -p "$out_dir"

  echo "[ablation] group=$group model=$model setting=$setting conf=$conf iou=$iou imgsz=$imgsz"

  python3 tools/eval_ablation.py \
    --config "$CONFIG_PATH" \
    --images-dir "$IMAGES_DIR" \
    --labels-dir "$LABELS_DIR" \
    --model "$model" \
    --conf "$conf" \
    --iou "$iou" \
    --group "$group" \
    --setting "$setting" \
    --out-dir "$out_dir" \
    ${imgsz:+--imgsz "$imgsz"}
}

if [[ "$ABLATION_MODE" == "all" || "$ABLATION_MODE" == "size" ]]; then
  # A组：输入尺寸敏感性分析
  # 说明：
  # - YOLO 与 FCOS 支持推理阶段尺寸覆写，可直接做 512/640/800 对比
  # - DRENet 当前本地插件在非 512 尺寸下会触发结构 shape mismatch，因此不纳入本组批量运行
  for model in mmdet_fcos yolo; do
    run_eval size "$model" "imgsz512" "0.25" "0.50" "512"
    run_eval size "$model" "imgsz640" "0.25" "0.50" "640"
    run_eval size "$model" "imgsz800" "0.25" "0.50" "800"
  done
fi

if [[ "$ABLATION_MODE" == "all" || "$ABLATION_MODE" == "threshold" ]]; then
  # B组：阈值消融
  # 说明：
  # - DRENet / YOLO26 默认基线尺寸为 512
  # - FCOS 阈值组使用其当前默认推理尺寸（来自原配置），避免误写成统一 512
  run_eval threshold "drenet" "conf015" "0.15" "0.50" "512"
  run_eval threshold "drenet" "conf025" "0.25" "0.50" "512"
  run_eval threshold "drenet" "conf035" "0.35" "0.50" "512"

  run_eval threshold "mmdet_fcos" "conf015_native" "0.15" "0.50" ""
  run_eval threshold "mmdet_fcos" "conf025_native" "0.25" "0.50" ""
  run_eval threshold "mmdet_fcos" "conf035_native" "0.35" "0.50" ""

  run_eval threshold "yolo" "conf015" "0.15" "0.50" "512"
  run_eval threshold "yolo" "conf025" "0.25" "0.50" "512"
  run_eval threshold "yolo" "conf035" "0.35" "0.50" "512"
fi
