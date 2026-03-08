#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/watch_sync_then_shutdown_autodl.sh <user@host> [options]

Options:
  --port <port>                    SSH port. Default: 22
  --run-name <name>                Required run name under experiment_assets/runs/
  --watch-pattern <pattern>        Required process pattern to watch on remote host
  --local-assets-root <path>       Default: /Users/khs/codes/graduation_project/experiment_assets
  --remote-assets-root <path>      Default: /root/autodl-tmp/experiment_assets
  --interval <seconds>             Default: 120
  --sync-wandb                     Also sync wandb directory
  --shutdown-cmd <cmd>             Default: shutdown -h now
  --no-snapshot-checkpoint         Skip local checkpoint snapshot before shutdown
  --snapshot-tag <text>            Optional suffix tag for snapshot filename
  -h, --help                       Show this help

Environment:
  SYNC_SSH_PASSWORD                Optional password for ssh/scp via expect
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

REMOTE="$1"
shift
PORT="22"
RUN_NAME=""
WATCH_PATTERN=""
LOCAL_ASSETS_ROOT="/Users/khs/codes/graduation_project/experiment_assets"
REMOTE_ASSETS_ROOT="/root/autodl-tmp/experiment_assets"
INTERVAL="120"
SYNC_WANDB="0"
SHUTDOWN_CMD="shutdown -h now"
SNAPSHOT_CHECKPOINT="1"
SNAPSHOT_TAG=""
PASSWORD="${SYNC_SSH_PASSWORD:-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --watch-pattern) WATCH_PATTERN="$2"; shift 2 ;;
    --local-assets-root) LOCAL_ASSETS_ROOT="$2"; shift 2 ;;
    --remote-assets-root) REMOTE_ASSETS_ROOT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --sync-wandb) SYNC_WANDB="1"; shift ;;
    --shutdown-cmd) SHUTDOWN_CMD="$2"; shift 2 ;;
    --no-snapshot-checkpoint) SNAPSHOT_CHECKPOINT="0"; shift ;;
    --snapshot-tag) SNAPSHOT_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$RUN_NAME" ] || [ -z "$WATCH_PATTERN" ]; then
  echo "Both --run-name and --watch-pattern are required." >&2
  exit 1
fi

SYNC_SCRIPT="/Users/khs/codes/graduation_project/scripts/sync_autodl_experiment_assets.sh"
SNAPSHOT_SCRIPT="/Users/khs/codes/graduation_project/scripts/snapshot_drenet_checkpoint.sh"
RESULTS_LOCAL="$LOCAL_ASSETS_ROOT/runs/$RUN_NAME/results.txt"
SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new)

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

echo "[sync-shutdown] watch sync start"
SYNC_SSH_PASSWORD="$PASSWORD" bash "$SYNC_SCRIPT" "$REMOTE" \
  --port "$PORT" \
  --run-name "$RUN_NAME" \
  --watch-pattern "$WATCH_PATTERN" \
  --interval "$INTERVAL" \
  --local-assets-root "$LOCAL_ASSETS_ROOT" \
  --no-sync-checkpoints \
  --no-sync-scripts \
  $( [ "$SYNC_WANDB" = "1" ] && echo "--sync-wandb" )

echo "[sync-shutdown] watch exited, run final sync"
SYNC_SSH_PASSWORD="$PASSWORD" bash "$SYNC_SCRIPT" "$REMOTE" \
  --port "$PORT" \
  --run-name "$RUN_NAME" \
  --local-assets-root "$LOCAL_ASSETS_ROOT" \
  --no-sync-checkpoints \
  --no-sync-scripts \
  $( [ "$SYNC_WANDB" = "1" ] && echo "--sync-wandb" )

REMOTE_LAST_LINE="$(run_ssh "tail -n 1 '$REMOTE_ASSETS_ROOT/runs/$RUN_NAME/results.txt'" | tr -d '\r')"
LOCAL_LAST_LINE="$(tail -n 1 "$RESULTS_LOCAL" | tr -d '\r')"

echo "[sync-shutdown] remote tail: $REMOTE_LAST_LINE"
echo "[sync-shutdown] local  tail: $LOCAL_LAST_LINE"

if [ "$REMOTE_LAST_LINE" != "$LOCAL_LAST_LINE" ]; then
  echo "[sync-shutdown] mismatch after final sync, abort shutdown" >&2
  exit 2
fi

if [ "$SNAPSHOT_CHECKPOINT" = "1" ]; then
  echo "[sync-shutdown] create local checkpoint snapshot"
  SNAPSHOT_ARGS=(--run-name "$RUN_NAME" --assets-root "$LOCAL_ASSETS_ROOT")
  if [ -n "$SNAPSHOT_TAG" ]; then
    SNAPSHOT_ARGS+=(--tag "$SNAPSHOT_TAG")
  fi
  bash "$SNAPSHOT_SCRIPT" "${SNAPSHOT_ARGS[@]}"
fi

echo "[sync-shutdown] sync verified, shutdown remote host"
run_ssh "$SHUTDOWN_CMD"
echo "[sync-shutdown] shutdown command sent"
