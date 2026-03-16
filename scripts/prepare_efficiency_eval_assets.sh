#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/experiment_assets/benchmarks/efficiency_prep_${STAMP}"
LATEST_LINK="${ROOT_DIR}/experiment_assets/benchmarks/latest_efficiency_prep"

mkdir -p "${OUT_DIR}"

DRENET_CKPT="${ROOT_DIR}/experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_best_ep299_20260308_124629_manual_save.pt"
FCOS_CKPT="${ROOT_DIR}/experiment_assets/checkpoints/mmdet/fcos_main_fixedcfg_20260315_160824_global_best_ep14.pth"
YOLO_CKPT="${ROOT_DIR}/experiment_assets/checkpoints/yolo/yolo26_main_512_formal012_best.pt"

DRENET_OK=false
FCOS_OK=false
YOLO_OK=false

[[ -f "${DRENET_CKPT}" ]] && DRENET_OK=true
[[ -f "${FCOS_CKPT}" ]] && FCOS_OK=true
[[ -f "${YOLO_CKPT}" ]] && YOLO_OK=true

cat > "${OUT_DIR}/benchmark_manifest.json" <<EOF
{
  "protocol": {
    "imgsz": 512,
    "batch": 1,
    "warmup_iters": 50,
    "timing_iters": 200,
    "precision": "fp32_or_fp16_but_fixed_per_run",
    "same_gpu_required": true
  },
  "models": {
    "drenet": {
      "exp_id": "exp-20260308-02",
      "checkpoint": "${DRENET_CKPT}",
      "exists": ${DRENET_OK}
    },
    "fcos": {
      "exp_id": "exp-20260315-01",
      "checkpoint": "${FCOS_CKPT}",
      "exists": ${FCOS_OK}
    },
    "yolo26n": {
      "exp_id": "exp-20260308-03",
      "checkpoint": "${YOLO_CKPT}",
      "exists": ${YOLO_OK}
    }
  },
  "notes": [
    "This package is prepared in no-GPU mode.",
    "Run the commands in run_gpu_efficiency_commands.md on the GPU instance."
  ]
}
EOF

cat > "${OUT_DIR}/efficiency_results_template.json" <<'EOF'
{
  "exp-20260308-02": {
    "fps": null,
    "params_m": null,
    "flops_g": null,
    "precision_mode": "fp32",
    "device": "cuda:0",
    "source": "drenet_benchmark"
  },
  "exp-20260315-01": {
    "fps": null,
    "params_m": null,
    "flops_g": null,
    "precision_mode": "fp32",
    "device": "cuda:0",
    "source": "mmdet_benchmark"
  },
  "exp-20260308-03": {
    "fps": null,
    "params_m": null,
    "flops_g": null,
    "precision_mode": "fp32",
    "device": "cuda:0",
    "source": "ultralytics_benchmark"
  }
}
EOF

cat > "${OUT_DIR}/run_gpu_efficiency_commands.md" <<'EOF'
# GPU执行命令（三模型统一效率口径）

统一条件：
- `imgsz=512`
- `batch=1`
- `warmup=50`
- `timing=200`
- 同一GPU、同一精度模式（FP32或FP16）

## 0) 环境变量（按实际主机改）
```bash
export GP_ROOT=/workspace/graduation_project
export DATA_YAML=/data/LEVIR-Ship/ship.yaml
export TEST_IMG_DIR=/data/LEVIR-Ship/test/images
```

## 1) YOLO26n（Ultralytics）
```bash
cd /workspace/ultralytics
python - <<'PY'
from ultralytics import YOLO
m = YOLO("/workspace/graduation_project/experiment_assets/checkpoints/yolo/yolo26_main_512_formal012_best.pt")
m.info(verbose=True)
PY

# 可复用统一测速脚本（建议）
cd /workspace/graduation_project
python tools/benchmarks/benchmark_infer_fps.py \
  --framework yolo \
  --weights experiment_assets/checkpoints/yolo/yolo26_main_512_formal012_best.pt \
  --image_dir "$TEST_IMG_DIR" \
  --imgsz 512 --warmup 50 --iters 200 --device cuda:0 \
  --output experiment_assets/benchmarks/yolo26_efficiency.json
```

## 2) FCOS（MMDetection）
```bash
cd /workspace/mmdetection
python tools/analysis_tools/get_flops.py \
  /workspace/graduation_project/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/fcos_main_fixedcfg_20260315_160824.py \
  --shape 512 512

cd /workspace/graduation_project
python tools/benchmarks/benchmark_infer_fps.py \
  --framework mmdet \
  --config experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/fcos_main_fixedcfg_20260315_160824.py \
  --weights experiment_assets/checkpoints/mmdet/fcos_main_fixedcfg_20260315_160824_global_best_ep14.pth \
  --image_dir "$TEST_IMG_DIR" \
  --imgsz 512 --warmup 50 --iters 200 --device cuda:0 \
  --output experiment_assets/benchmarks/fcos_efficiency.json
```

## 3) DRENet
```bash
cd /workspace/DRENet
# 先测 params/flops（若仓库已有thop脚本就用仓库脚本；否则用下方统一脚本）

cd /workspace/graduation_project
python tools/benchmarks/benchmark_infer_fps.py \
  --framework drenet \
  --weights experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_best_ep299_20260308_124629_manual_save.pt \
  --image_dir "$TEST_IMG_DIR" \
  --imgsz 512 --warmup 50 --iters 200 --device cuda:0 \
  --output experiment_assets/benchmarks/drenet_efficiency.json
```

## 4) 汇总并回填 baselines
```bash
cd /workspace/graduation_project
python tools/benchmarks/merge_efficiency_results.py \
  --inputs experiment_assets/benchmarks/drenet_efficiency.json \
           experiment_assets/benchmarks/fcos_efficiency.json \
           experiment_assets/benchmarks/yolo26_efficiency.json \
  --output experiment_assets/benchmarks/efficiency_results_merged.json

python tools/benchmarks/update_baselines_efficiency.py \
  --baselines docs/results/baselines.md \
  --metrics experiment_assets/benchmarks/efficiency_results_merged.json
```
EOF

ln -sfn "${OUT_DIR}" "${LATEST_LINK}"

echo "[OK] no-GPU prep completed."
echo "[OUT] ${OUT_DIR}"
echo "[LINK] ${LATEST_LINK}"
