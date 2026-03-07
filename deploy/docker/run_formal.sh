#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GIT_SHA="$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo no_git)"
IMAGE_TAG_DEFAULT="drenet-train:${GIT_SHA}-cu124"

MODE="fresh"
IMAGE_TAG="${IMAGE_TAG_DEFAULT}"
EPOCHS=300
BATCH=4
WORKERS=4
IMGSZ=512
SEED=42
DATE_TAG="$(date +%Y%m%d)"
DATASET_NAME="levirship"
HOST_TAG="$(hostname -s 2>/dev/null || hostname)"

RUNS_ROOT="${RUNS_ROOT:-${PROJECT_ROOT}/.artifacts/runs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/.artifacts/checkpoints}"
DATASET_ROOT="${DATASET_ROOT:-/data/LEVIR-Ship}"
WANDB_PROJECT="${WANDB_PROJECT:-graduation-drenet}"
WANDB_RESUME_MODE="${WANDB_RESUME_MODE:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
EXTRA_TAGS="${EXTRA_TAGS:-}"

EXP_NAME="${EXP_NAME:-drenet_${DATASET_NAME}_${IMGSZ}_bs${BATCH}_s${SEED}_${DATE_TAG}}"
EXP_NAME_SET=0
RESUME_CKPT="${RESUME_CKPT:-}"
NO_BUILD=0

usage() {
  cat <<'EOF'
Usage:
  run_formal.sh [options]

Options:
  --mode fresh|strict-resume|speed-restart
  --exp-name <name>
  --dataset-root <host_path>          (default: /data/LEVIR-Ship)
  --runs-root <host_path>             (default: <repo>/.artifacts/runs)
  --checkpoints-root <host_path>      (default: <repo>/.artifacts/checkpoints)
  --resume-ckpt <host_path_to_pt>     (required for strict-resume/speed-restart)
  --epochs <n> --batch <n> --workers <n> --imgsz <n> --seed <n>
  --image-tag <tag>                   (default: drenet-train:<git_sha>-cu124)
  --wandb-project <name>              (default: graduation-drenet)
  --wandb-run-id <id>                 (optional, for same-run resume)
  --wandb-resume <mode>               (optional: allow/must/never/auto)
  --extra-tags <csv>                  (optional, append extra W&B tags)
  --no-build                          (skip docker build)
  -h|--help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; EXP_NAME_SET=1; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --checkpoints-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --resume-ckpt) RESUME_CKPT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --image-tag) IMAGE_TAG="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-run-id) WANDB_RUN_ID="$2"; shift 2 ;;
    --wandb-resume) WANDB_RESUME_MODE="$2"; shift 2 ;;
    --extra-tags) EXTRA_TAGS="$2"; shift 2 ;;
    --no-build) NO_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[ERROR] DATASET_ROOT not found: ${DATASET_ROOT}"
  exit 2
fi

if [[ ${EXP_NAME_SET} -eq 0 ]]; then
  EXP_NAME="drenet_${DATASET_NAME}_${IMGSZ}_bs${BATCH}_s${SEED}_${DATE_TAG}"
fi

if [[ "${MODE}" != "fresh" && "${MODE}" != "strict-resume" && "${MODE}" != "speed-restart" ]]; then
  echo "[ERROR] --mode must be fresh|strict-resume|speed-restart"
  exit 2
fi

if [[ "${MODE}" != "fresh" && -z "${RESUME_CKPT}" ]]; then
  echo "[ERROR] --resume-ckpt is required for ${MODE}"
  exit 2
fi

if [[ "${MODE}" != "fresh" && ! -f "${RESUME_CKPT}" ]]; then
  echo "[ERROR] resume checkpoint not found: ${RESUME_CKPT}"
  exit 2
fi

if [[ -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" && ! -f "${HOME}/_netrc" ]]; then
  echo "[ERROR] W&B credentials missing. Export WANDB_API_KEY or run 'wandb login' first."
  exit 2
fi

mkdir -p "${RUNS_ROOT}/trace" "${CHECKPOINT_ROOT}/drenet"
LOG_PATH="${RUNS_ROOT}/trace/train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

if [[ ${NO_BUILD} -eq 0 ]]; then
  docker build -f "${PROJECT_ROOT}/deploy/docker/Dockerfile.drenet" -t "${IMAGE_TAG}" "${PROJECT_ROOT}"
fi

W_TAGS="dataset=${DATASET_NAME},imgsz=${IMGSZ},bs=${BATCH},seed=${SEED},stage=formal,host=${HOST_TAG}"
if [[ -n "${EXTRA_TAGS}" ]]; then
  W_TAGS="${W_TAGS},${EXTRA_TAGS}"
fi

CKPT_IN_CONTAINER=""
if [[ -n "${RESUME_CKPT}" ]]; then
  CKPT_IN_CONTAINER="/workspace/checkpoints/$(basename "${RESUME_CKPT}")"
  cp -f "${RESUME_CKPT}" "${CHECKPOINT_ROOT}/$(basename "${RESUME_CKPT}")"
fi

TRAIN_CMD_COMMON="cd /workspace/project/experiments/drenet/DRENet && \
cat >/tmp/levir_ship_container.yaml <<'YAML'
train: /workspace/datasets/LEVIR-Ship/train/images
val: /workspace/datasets/LEVIR-Ship/val/images
test: /workspace/datasets/LEVIR-Ship/test/images
nc: 1
names: ['ship']
YAML
"

if [[ "${MODE}" == "fresh" ]]; then
  TRAIN_CMD="${TRAIN_CMD_COMMON}
python train.py \
  --data /tmp/levir_ship_container.yaml \
  --cfg models/DRENet.yaml \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH} \
  --img-size ${IMGSZ} \
  --workers ${WORKERS} \
  --device 0 \
  --project /workspace/runs \
  --name ${EXP_NAME} \
  --exist-ok"
elif [[ "${MODE}" == "strict-resume" ]]; then
  TRAIN_CMD="${TRAIN_CMD_COMMON}
python train.py --resume ${CKPT_IN_CONTAINER}"
else
  TRAIN_CMD="${TRAIN_CMD_COMMON}
python train.py \
  --data /tmp/levir_ship_container.yaml \
  --cfg models/DRENet.yaml \
  --weights ${CKPT_IN_CONTAINER} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH} \
  --img-size ${IMGSZ} \
  --workers ${WORKERS} \
  --device 0 \
  --project /workspace/runs \
  --name ${EXP_NAME} \
  --exist-ok"
fi

{
  echo "=== DRENet Docker Formal Run ==="
  echo "mode=${MODE}"
  echo "image_tag=${IMAGE_TAG}"
  echo "project_root=${PROJECT_ROOT}"
  echo "dataset_root=${DATASET_ROOT}"
  echo "runs_root=${RUNS_ROOT}"
  echo "checkpoints_root=${CHECKPOINT_ROOT}"
  echo "exp_name=${EXP_NAME}"
  echo "wandb_project=${WANDB_PROJECT}"
  echo "wandb_tags=${W_TAGS}"
  if [[ -n "${RESUME_CKPT}" ]]; then
    echo "resume_ckpt=${RESUME_CKPT}"
  fi
  echo "log_path=${LOG_PATH}"

  docker run --rm --gpus all --shm-size=16g \
    -v "${PROJECT_ROOT}:/workspace/project" \
    -v "${DATASET_ROOT}:/workspace/datasets/LEVIR-Ship:ro" \
    -v "${RUNS_ROOT}:/workspace/runs" \
    -v "${CHECKPOINT_ROOT}:/workspace/checkpoints" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_PROJECT="${WANDB_PROJECT}" \
    -e WANDB_NAME="${EXP_NAME}" \
    -e WANDB_TAGS="${W_TAGS}" \
    -e WANDB_RUN_ID="${WANDB_RUN_ID}" \
    -e WANDB_RESUME="${WANDB_RESUME_MODE}" \
    "${IMAGE_TAG}" \
    /bin/bash -lc "${TRAIN_CMD}"
} 2>&1 | tee "${LOG_PATH}"

EXIT_CODE=${PIPESTATUS[0]}
echo "TRAIN_EXIT=${EXIT_CODE}" | tee -a "${LOG_PATH}"
echo "TRACE_LOG=${LOG_PATH}" | tee -a "${LOG_PATH}"

if [[ ${EXIT_CODE} -eq 0 ]]; then
  RUN_DIR="${RUNS_ROOT}/${EXP_NAME}"
  if [[ -f "${RUN_DIR}/weights/best.pt" ]]; then
    cp -f "${RUN_DIR}/weights/best.pt" "${CHECKPOINT_ROOT}/drenet/${EXP_NAME}_best.pt"
  fi
  if [[ -f "${RUN_DIR}/weights/last.pt" ]]; then
    cp -f "${RUN_DIR}/weights/last.pt" "${CHECKPOINT_ROOT}/drenet/${EXP_NAME}_last.pt"
  fi
fi

exit "${EXIT_CODE}"
