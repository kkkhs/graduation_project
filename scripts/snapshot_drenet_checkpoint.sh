#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/snapshot_drenet_checkpoint.sh --run-name <name> [options]

Options:
  --run-name <name>                Required run name under experiment_assets/runs/
  --assets-root <path>             experiment_assets root.
                                   Default: /Users/khs/codes/graduation_project/experiment_assets
  --epoch <n>                      Optional epoch override. By default parse from results.txt tail.
  --tag <text>                     Optional suffix tag appended to output filename.
  -h, --help                       Show this help
EOF
}

ASSETS_ROOT="/Users/khs/codes/graduation_project/experiment_assets"
RUN_NAME=""
EPOCH_OVERRIDE=""
TAG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --assets-root) ASSETS_ROOT="$2"; shift 2 ;;
    --epoch) EPOCH_OVERRIDE="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$RUN_NAME" ]; then
  echo "--run-name is required." >&2
  usage
  exit 1
fi

RUN_DIR="$ASSETS_ROOT/runs/$RUN_NAME"
WEIGHTS_DIR="$RUN_DIR/weights"
RESULTS_FILE="$RUN_DIR/results.txt"
CKPT_DIR="$ASSETS_ROOT/checkpoints/drenet"

if [ ! -d "$RUN_DIR" ]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 2
fi

if [ ! -f "$WEIGHTS_DIR/last.pt" ] && [ ! -f "$WEIGHTS_DIR/best.pt" ]; then
  echo "No weights found under: $WEIGHTS_DIR" >&2
  exit 3
fi

EPOCH="$EPOCH_OVERRIDE"
if [ -z "$EPOCH" ]; then
  if [ -f "$RESULTS_FILE" ]; then
    TAIL_LINE="$(tail -n 1 "$RESULTS_FILE" | tr -d '\r')"
    TOKEN="$(printf '%s\n' "$TAIL_LINE" | awk '{print $1}')"
    EPOCH="${TOKEN%%/*}"
  fi
fi

if [ -z "$EPOCH" ] || ! printf '%s' "$EPOCH" | grep -Eq '^[0-9]+$'; then
  EPOCH="unknown"
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
TAG_SUFFIX=""
if [ -n "$TAG" ]; then
  TAG_SUFFIX="_$TAG"
fi

mkdir -p "$CKPT_DIR"

copy_if_exists() {
  local src="$1"
  local kind="$2"
  local dst="$CKPT_DIR/${RUN_NAME}_${kind}_ep${EPOCH}_${STAMP}${TAG_SUFFIX}.pt"
  if [ -f "$src" ]; then
    cp -f "$src" "$dst"
    echo "SNAPSHOT_OK $dst"
  else
    echo "SNAPSHOT_SKIP missing:$src"
  fi
}

copy_if_exists "$WEIGHTS_DIR/last.pt" "last"
copy_if_exists "$WEIGHTS_DIR/best.pt" "best"
