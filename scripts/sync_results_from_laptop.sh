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

mkdir -p docs/experiments/logs assets/figures docs/results artifacts

sync_dir() {
  local remote_subpath="$1"
  local local_subpath="$2"
  if rsync -avz --progress "$REMOTE:$REMOTE_PATH/$remote_subpath/" "$local_subpath/"; then
    echo "Synced: $remote_subpath -> $local_subpath"
  else
    echo "Warn: skip missing or unavailable path: $remote_subpath"
  fi
}

# 同步实验日志、结果图、结果文档、训练产物与训练日志
sync_dir "docs/experiments/logs" "docs/experiments/logs"
sync_dir "assets/figures" "assets/figures"
sync_dir "docs/results" "docs/results"
sync_dir "artifacts" "artifacts"
sync_dir "runs" "artifacts/training_runs"
sync_dir "work_dirs" "artifacts/mmdet_work_dirs"

echo "Sync finished."
