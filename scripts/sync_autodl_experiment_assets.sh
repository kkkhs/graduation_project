#!/usr/bin/env bash
set -euo pipefail

# 用法示例：
# bash scripts/sync_autodl_experiment_assets.sh \
#   root@connect.westc.gpuhub.com \
#   --port 49353 \
#   --run-name drenet_levirship_512_bs4_sna_20260307_formal01 \
#   --watch-pattern 'train.py|resume_formal01_autodl.sh'

REMOTE=""
PORT="22"
REMOTE_ASSETS_ROOT="/root/autodl-tmp/experiment_assets"
LOCAL_ASSETS_ROOT="experiment_assets"
RUN_NAME=""
WATCH_PATTERN=""
INTERVAL="60"
SYNC_DATASETS="0"
SYNC_SCRIPTS="1"
SYNC_CHECKPOINTS="1"
SYNC_WANDB="0"
REMOTE_WANDB_ROOT="/root/autodl-tmp/workspace/experiments/drenet/DRENet/wandb"
SYNC_TRACE="1"
STAGING_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sync_autodl_experiment_assets.sh <user@host> [options]

Options:
  --port <port>                    SSH port. Default: 22
  --remote-assets-root <path>      Remote experiment_assets root.
                                   Default: /root/autodl-tmp/experiment_assets
  --local-assets-root <path>       Local experiment_assets root.
                                   Default: experiment_assets
  --run-name <name>                Sync only the specified run and matching checkpoints.
  --watch-pattern <pattern>        Poll remote process list. When the pattern disappears,
                                   perform one final sync and exit.
  --interval <seconds>             Poll interval for watch mode. Default: 60
  --sync-datasets                  Also sync datasets/ (off by default).
  --no-sync-scripts                Skip scripts/ sync.
  --no-sync-checkpoints            Skip checkpoints/ sync.
  --sync-wandb                     Also sync remote wandb directory into local experiment_assets/wandb/.
  --no-sync-trace                 Skip runs/trace sync.
  --remote-wandb-root <path>       Remote wandb root. Default:
                                   /root/autodl-tmp/workspace/experiments/drenet/DRENet/wandb
  --staging-dir <path>             Local temp dir for tar/scp fallback.
                                   Default: .tmp/sync_autodl_experiment_assets
Environment:
  SYNC_SSH_PASSWORD                Optional password for ssh/scp via expect.
  -h, --help                       Show this help.
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

REMOTE="$1"
shift

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --remote-assets-root)
      REMOTE_ASSETS_ROOT="$2"
      shift 2
      ;;
    --local-assets-root)
      LOCAL_ASSETS_ROOT="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --watch-pattern)
      WATCH_PATTERN="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --sync-datasets)
      SYNC_DATASETS="1"
      shift
      ;;
    --no-sync-scripts)
      SYNC_SCRIPTS="0"
      shift
      ;;
    --no-sync-checkpoints)
      SYNC_CHECKPOINTS="0"
      shift
      ;;
    --sync-wandb)
      SYNC_WANDB="1"
      shift
      ;;
    --no-sync-trace)
      SYNC_TRACE="0"
      shift
      ;;
    --remote-wandb-root)
      REMOTE_WANDB_ROOT="$2"
      shift 2
      ;;
    --staging-dir)
      STAGING_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-P "$PORT" -o StrictHostKeyChecking=accept-new)
RSYNC_RSH="ssh ${SSH_OPTS[*]}"
PASSWORD="${SYNC_SSH_PASSWORD:-}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

remote_has_cmd() {
  run_ssh "command -v $1 >/dev/null 2>&1"
}

require_cmd ssh
require_cmd scp
if [ -n "$PASSWORD" ]; then
  require_cmd expect
fi

mkdir -p "$LOCAL_ASSETS_ROOT"/runs
mkdir -p "$LOCAL_ASSETS_ROOT"/checkpoints
mkdir -p "$LOCAL_ASSETS_ROOT"/scripts
mkdir -p "$LOCAL_ASSETS_ROOT"/wandb
mkdir -p "$LOCAL_ASSETS_ROOT"/runs/trace

if [ -z "$STAGING_DIR" ]; then
  STAGING_DIR="${LOCAL_ASSETS_ROOT}/.tmp/sync_autodl_experiment_assets"
fi
mkdir -p "$STAGING_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

run_expect() {
  local prog="$1"
  shift
  expect -f - "$PASSWORD" "$prog" "$@" <<'EOF'
set timeout -1
set password [lindex $argv 0]
set prog [lindex $argv 1]
set args [lrange $argv 2 end]
log_user 1
set cmd [list $prog]
foreach arg $args {
  lappend cmd $arg
}
eval spawn $cmd
expect {
  "*yes/no*" { send "yes\r"; exp_continue }
  "*password:*" { send "$password\r"; exp_continue }
  eof
}
catch wait result
set code [lindex $result 3]
exit $code
EOF
}

run_ssh() {
  local remote_cmd="$1"
  if [ -n "$PASSWORD" ]; then
    run_expect ssh "${SSH_OPTS[@]}" "$REMOTE" "$remote_cmd"
  else
    ssh "${SSH_OPTS[@]}" "$REMOTE" "$remote_cmd"
  fi
}

run_scp() {
  if [ -n "$PASSWORD" ]; then
    run_expect scp "${SCP_OPTS[@]}" "$@"
  else
    scp "${SCP_OPTS[@]}" "$@"
  fi
}

sync_path() {
  local remote_path="$1"
  local local_path="$2"
  mkdir -p "$local_path"
  rsync -az --partial --progress -e "$RSYNC_RSH" \
    "$REMOTE:$remote_path/" "$local_path/"
}

REMOTE_SYNC_MODE="tar"
if [ -z "$PASSWORD" ] && remote_has_cmd rsync; then
  REMOTE_SYNC_MODE="rsync"
fi

sync_path_tar() {
  local remote_src="$1"
  local local_dest="$2"
  local archive_name="$3"
  local remote_archive="/tmp/${archive_name}"
  local local_archive="${STAGING_DIR}/${archive_name}"

  mkdir -p "$local_dest"
  rm -f "$local_archive"
  # Best-effort snapshot for actively-written files (e.g. weights during training).
  # Ignore file-changed races and retry once before failing.
  run_ssh "set -euo pipefail; rm -f '$remote_archive'; if [ -d '$remote_src' ]; then tar --warning=no-file-changed --ignore-failed-read -C '$remote_src' -czf '$remote_archive' . || tar --warning=no-file-changed --ignore-failed-read -C '$remote_src' -czf '$remote_archive' .; else exit 3; fi"
  run_scp "$REMOTE:$remote_archive" "$local_archive"
  tar -xzf "$local_archive" -C "$local_dest"
  run_ssh "rm -f '$remote_archive'" || true
}

sync_file_tar() {
  local remote_file="$1"
  local local_file="$2"
  local archive_name="$3"
  local remote_archive="/tmp/${archive_name}"
  local local_archive="${STAGING_DIR}/${archive_name}"
  local local_dir
  local local_name

  local_dir="$(dirname "$local_file")"
  local_name="$(basename "$local_file")"
  mkdir -p "$local_dir"
  rm -f "$local_archive"
  run_ssh "set -euo pipefail; test -f '$remote_file'; rm -f '$remote_archive'; tar --warning=no-file-changed --ignore-failed-read -C '$(dirname "$remote_file")' -czf '$remote_archive' '$local_name'"
  run_scp "$REMOTE:$remote_archive" "$local_archive"
  tar -xzf "$local_archive" -C "$local_dir"
  run_ssh "rm -f '$remote_archive'" || true
}

archive_safe_name() {
  local raw="$1"
  # Keep archive names filesystem-safe for nested run paths like detect/runs/<name>.
  raw="${raw//\//__}"
  raw="${raw// /_}"
  echo "$raw"
}

sync_run() {
  if [ -n "$RUN_NAME" ]; then
    local run_slug
    run_slug="$(archive_safe_name "$RUN_NAME")"
    log "Sync run: $RUN_NAME"
    if [ "$REMOTE_SYNC_MODE" = "rsync" ]; then
      sync_path "$REMOTE_ASSETS_ROOT/runs/$RUN_NAME" "$LOCAL_ASSETS_ROOT/runs/$RUN_NAME"
    else
      sync_path_tar "$REMOTE_ASSETS_ROOT/runs/$RUN_NAME" "$LOCAL_ASSETS_ROOT/runs/$RUN_NAME" "run_${run_slug}.tgz"
    fi
  else
    log "Sync all runs"
    if [ "$REMOTE_SYNC_MODE" = "rsync" ]; then
      sync_path "$REMOTE_ASSETS_ROOT/runs" "$LOCAL_ASSETS_ROOT/runs"
    else
      sync_path_tar "$REMOTE_ASSETS_ROOT/runs" "$LOCAL_ASSETS_ROOT/runs" "runs_all.tgz"
    fi
  fi
}

sync_checkpoints() {
  [ "$SYNC_CHECKPOINTS" = "1" ] || return 0
  mkdir -p "$LOCAL_ASSETS_ROOT/checkpoints"
  if [ -n "$RUN_NAME" ]; then
    local run_slug
    run_slug="$(archive_safe_name "$RUN_NAME")"
    log "Sync checkpoints matching run: $RUN_NAME"
    if [ "$REMOTE_SYNC_MODE" = "rsync" ]; then
      rsync -az --partial --progress -e "$RSYNC_RSH" \
        --include='*/' \
        --include="*$RUN_NAME*" \
        --exclude='*' \
        "$REMOTE:$REMOTE_ASSETS_ROOT/checkpoints/" "$LOCAL_ASSETS_ROOT/checkpoints/"
    else
      local remote_archive="/tmp/checkpoints_${run_slug}.tgz"
      local local_archive="${STAGING_DIR}/checkpoints_${run_slug}.tgz"
      rm -f "$local_archive"
      run_ssh "set -euo pipefail; rm -f '$remote_archive'; cd '$REMOTE_ASSETS_ROOT/checkpoints' && tar -czf '$remote_archive' \$(find . -type f -name '*${RUN_NAME}*' -print)"
      run_scp "$REMOTE:$remote_archive" "$local_archive"
      tar -xzf "$local_archive" -C "$LOCAL_ASSETS_ROOT/checkpoints"
      run_ssh "rm -f '$remote_archive'" || true
    fi
  else
    log "Sync all checkpoints"
    if [ "$REMOTE_SYNC_MODE" = "rsync" ]; then
      sync_path "$REMOTE_ASSETS_ROOT/checkpoints" "$LOCAL_ASSETS_ROOT/checkpoints"
    else
      sync_path_tar "$REMOTE_ASSETS_ROOT/checkpoints" "$LOCAL_ASSETS_ROOT/checkpoints" "checkpoints_all.tgz"
    fi
  fi
}

sync_scripts() {
  [ "$SYNC_SCRIPTS" = "1" ] || return 0
  log "Sync scripts"
  if [ "$REMOTE_SYNC_MODE" = "rsync" ]; then
    sync_path "$REMOTE_ASSETS_ROOT/scripts" "$LOCAL_ASSETS_ROOT/scripts"
  else
    sync_path_tar "$REMOTE_ASSETS_ROOT/scripts" "$LOCAL_ASSETS_ROOT/scripts" "scripts_all.tgz"
  fi
}

sync_datasets() {
  [ "$SYNC_DATASETS" = "1" ] || return 0
  log "Sync datasets"
  if [ "$REMOTE_SYNC_MODE" = "rsync" ]; then
    sync_path "$REMOTE_ASSETS_ROOT/datasets" "$LOCAL_ASSETS_ROOT/datasets"
  else
    sync_path_tar "$REMOTE_ASSETS_ROOT/datasets" "$LOCAL_ASSETS_ROOT/datasets" "datasets_all.tgz"
  fi
}

sync_wandb() {
  [ "$SYNC_WANDB" = "1" ] || return 0
  log "Sync wandb"
  if [ "$REMOTE_SYNC_MODE" = "rsync" ]; then
    sync_path "$REMOTE_WANDB_ROOT" "$LOCAL_ASSETS_ROOT/wandb"
  else
    sync_path_tar "$REMOTE_WANDB_ROOT" "$LOCAL_ASSETS_ROOT/wandb" "wandb_all.tgz"
  fi
}

sync_trace() {
  [ "$SYNC_TRACE" = "1" ] || return 0
  log "Sync trace"
  if [ "$REMOTE_SYNC_MODE" = "rsync" ]; then
    sync_path "$REMOTE_ASSETS_ROOT/runs/trace" "$LOCAL_ASSETS_ROOT/runs/trace"
  else
    sync_path_tar "$REMOTE_ASSETS_ROOT/runs/trace" "$LOCAL_ASSETS_ROOT/runs/trace" "runs_trace.tgz"
  fi
}

remote_pattern_active() {
  [ -n "$WATCH_PATTERN" ] || return 1
  run_ssh "pgrep -af \"$WATCH_PATTERN\" >/dev/null 2>&1"
}

sync_once() {
  sync_run
  sync_checkpoints
  sync_scripts
  sync_datasets
  sync_trace
  sync_wandb
  log "Sync pass finished"
}

log "Remote sync mode: $REMOTE_SYNC_MODE"

if [ -n "$WATCH_PATTERN" ]; then
  log "Watch mode enabled: pattern=$WATCH_PATTERN interval=${INTERVAL}s"
  while true; do
    sync_once
    if remote_pattern_active; then
      log "Remote training still active"
      sleep "$INTERVAL"
    else
      log "Remote training not found, perform final sync"
      sync_once
      break
    fi
  done
else
  sync_once
fi
