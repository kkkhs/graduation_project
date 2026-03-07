param(
  [string]$RunDir = "E:\workspace\runs\drenet_levirship_512_bs4_sna_20260307_formal01",
  [int]$TargetEpoch = 100,
  [int]$PollSec = 60
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $RunDir)) {
  Write-Host "RunDir not found: $RunDir"
  exit 2
}

$traceDir = "E:\workspace\runs\trace"
New-Item -ItemType Directory -Force -Path $traceDir | Out-Null

$runName = Split-Path $RunDir -Leaf
$logPath = Join-Path $traceDir ("watch_checkpoint_" + $runName + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Get-LatestEpoch([string]$resultsPath) {
  if (-not (Test-Path $resultsPath)) { return -1 }
  $line = Get-Content $resultsPath -Tail 1
  if ($line -match "^\s*(\d+)\/299") { return [int]$matches[1] }
  return -1
}

$resultsPath = Join-Path $RunDir "results.txt"

"WATCH_START=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $logPath -Encoding UTF8
"RUN_DIR=$RunDir" | Out-File -FilePath $logPath -Encoding UTF8 -Append
"TARGET_EPOCH=$TargetEpoch" | Out-File -FilePath $logPath -Encoding UTF8 -Append
"RESULTS_PATH=$resultsPath" | Out-File -FilePath $logPath -Encoding UTF8 -Append

while ($true) {
  $epoch = Get-LatestEpoch $resultsPath
  $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "$now epoch=$epoch" | Out-File -FilePath $logPath -Encoding UTF8 -Append

  if ($epoch -ge $TargetEpoch) {
    $msg = "[CHECKPOINT_REACHED] run=$runName epoch=$epoch target=$TargetEpoch time=$now"
    $msg | Out-File -FilePath $logPath -Encoding UTF8 -Append
    try {
      msg * $msg | Out-Null
    } catch {
      # Best-effort notification only.
    }
    Write-Host $msg
    exit 0
  }

  Start-Sleep -Seconds $PollSec
}
