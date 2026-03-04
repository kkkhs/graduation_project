#!/usr/bin/env bash
set -euo pipefail

# 用法：
# bash scripts/sync_results_from_laptop.sh <laptop_user@laptop_host> <remote_project_path>
# 例：bash scripts/sync_results_from_laptop.sh khs@192.168.1.88 E:/Codes/Githubs/graduation_project

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <user@host> <remote_project_path>"
  exit 1
fi

REMOTE="$1"
REMOTE_PATH="$2"

mkdir -p docs/experiments/logs assets/figures

# 同步实验日志、结果图、结果文档（按需可扩展）
rsync -avz --progress "$REMOTE:$REMOTE_PATH/docs/experiments/logs/" "docs/experiments/logs/" || true
rsync -avz --progress "$REMOTE:$REMOTE_PATH/assets/figures/" "assets/figures/" || true
rsync -avz --progress "$REMOTE:$REMOTE_PATH/docs/results/" "docs/results/" || true

echo "Sync finished."
