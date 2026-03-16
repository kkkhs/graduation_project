#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/fcos_main_fixedcfg_20260315_160824.py}"
WEIGHT_PATH="${2:-$ROOT_DIR/experiment_assets/checkpoints/mmdet/fcos_main_fixedcfg_20260315_160824_global_best_ep14.pth}"
IMAGE_PATH="${3:-}"
VIS_OUT="${4:-$ROOT_DIR/outputs/visualizations/fcos_vis_result.jpg}"
JSON_OUT="${5:-$ROOT_DIR/outputs/predictions/fcos_pred_result.json}"

if [[ -z "$IMAGE_PATH" ]]; then
  echo "usage: $0 <config_path> <weight_path> <image_path> [vis_out] [json_out]" >&2
  echo "example:" >&2
  echo "  $0 \\"
  echo "    $ROOT_DIR/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/fcos_main_fixedcfg_20260315_160824.py \\"
  echo "    $ROOT_DIR/experiment_assets/checkpoints/mmdet/fcos_main_fixedcfg_20260315_160824_global_best_ep14.pth \\"
  echo "    $ROOT_DIR/experiment_assets/datasets/LEVIR-Ship/test/images/example.png" >&2
  exit 1
fi

mkdir -p "$(dirname "$VIS_OUT")" "$(dirname "$JSON_OUT")"

echo "[info] this script expects a working MMDetection environment"
echo "[info] config: $CONFIG_PATH"
echo "[info] weight: $WEIGHT_PATH"
echo "[info] image : $IMAGE_PATH"

PYTHONPATH="$ROOT_DIR" python3 "$ROOT_DIR/tools/visualize_predict.py" \
  --config "$ROOT_DIR/configs/models.yaml" \
  --image "$IMAGE_PATH" \
  --mode single \
  --model mmdet_fcos \
  --vis-out "$VIS_OUT" \
  --json-out "$JSON_OUT"

echo "[done] vis saved to $VIS_OUT"
echo "[done] json saved to $JSON_OUT"
