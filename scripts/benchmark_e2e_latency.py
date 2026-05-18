#!/usr/bin/env python3
"""End-to-end latency benchmark using the real system with mock_inference=True.

This tests the ACTUAL system pipeline: API → TaskExecutor → InferenceRuntime →
visualization → DB write → results endpoint → frontend polling.

Run with: .venv_pdf2zh/bin/python3 scripts/benchmark_e2e_latency.py

Prerequisites: The backend server must NOT be running (this script starts it).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_MODULE = "backend.app.main:app"
VENV_PYTHON = str(PROJECT_ROOT / ".venv_pdf2zh" / "bin" / "python3")

# ── Helpers ──────────────────────────────────────────────────────────

def _ensure_test_image() -> Path:
    """Create a small test image for e2e testing."""
    from PIL import Image, ImageDraw
    img_dir = PROJECT_ROOT / "outputs" / "bench_e2e"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "bench_test.png"
    if not img_path.exists():
        img = Image.new("RGB", (512, 512), color=(30, 60, 90))
        draw = ImageDraw.Draw(img)
        for i in range(5):
            draw.rectangle([i * 80, i * 60, i * 80 + 40, i * 60 + 30], fill=(200, 50, 50))
        img.save(img_path, format="PNG")
    return img_path


def _wait_for_server(url: str, timeout: float = 15) -> bool:
    """Wait until the server responds to health check."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(0.5)
    return False


def _api_call(method: str, path: str, data: bytes | None = None) -> dict:
    """Make an API call to the local server."""
    url = f"http://127.0.0.1:18080{path}"
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "multipart/form-data; boundary=----benchboundary")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": e.read().decode("utf-8")}


def _submit_task(image_path: Path, mode: str = "ensemble", model_key: str | None = None) -> dict:
    """Submit an inference task via multipart form upload."""
    url = f"http://127.0.0.1:18080/api/v1/tasks/infer"
    boundary = "----benchboundary"
    body = b""
    # Add type field
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\nsingle\r\n".encode()
    # Add mode field
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\n{mode}\r\n".encode()
    # Add score_thr
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"score_thr\"\r\n\r\n0.25\r\n".encode()
    # Add model_key if provided
    if model_key:
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"model_key\"\r\n\r\n{model_key}\r\n".encode()
    # Add image file
    filename = image_path.name
    with open(image_path, "rb") as f:
        file_data = f.read()
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"images\"; filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n".encode()
    body += file_data
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read().decode("utf-8"))


def _poll_until_done(task_id: int, use_progress: bool = False, interval: float = 1.5) -> tuple[str, float]:
    """Poll task status until done/failed, return (status, elapsed_seconds)."""
    start = time.perf_counter()
    while True:
        if use_progress:
            # Lightweight progress endpoint
            path = f"/api/v1/tasks/{task_id}/progress"
        else:
            # Heavy results endpoint (baseline)
            path = f"/api/v1/tasks/{task_id}"
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:18080{path}", timeout=10)
            data = json.loads(resp.read().decode("utf-8"))
            status = data.get("status", "unknown")
            if status in ("done", "failed"):
                elapsed = time.perf_counter() - start
                return status, elapsed
        except Exception:
            pass
        time.sleep(interval)


def _fetch_results(task_id: int) -> tuple[dict, float]:
    """Fetch full results and measure response time."""
    start = time.perf_counter()
    url = f"http://127.0.0.1:18080/api/v1/tasks/{task_id}/results"
    resp = urllib.request.urlopen(url, timeout=30)
    data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - start
    return data, elapsed


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   End-to-End Latency Benchmark (mock_inference=True)   ║")
    print("║   Tests the REAL system pipeline                       ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Set environment for mock inference
    env = os.environ.copy()
    env["APP_MOCK_INFERENCE"] = "true"
    env["APP_MAX_PARALLEL_MODELS"] = "2"
    env["APP_MAX_WORKERS"] = "1"
    env["APP_DB_PATH"] = str(PROJECT_ROOT / "bench_e2e_test.db")
    env["APP_OUTPUTS_ROOT"] = str(PROJECT_ROOT / "outputs" / "bench_e2e")

    # Clean up previous test artifacts
    db_path = Path(env["APP_DB_PATH"])
    if db_path.exists():
        db_path.unlink()

    # Start the backend server
    print("\n[1] Starting backend server with mock_inference=True...")
    server_proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", BACKEND_MODULE, "--host", "127.0.0.1", "--port", "18080"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
    )

    try:
        if not _wait_for_server("http://127.0.0.1:18080/api/v1/health"):
            print("FAIL: Server did not start within timeout")
            server_proc.kill()
            sys.exit(1)
        print("  Server is ready.")

        # Prepare test image
        img_path = _ensure_test_image()
        print(f"  Test image: {img_path} ({img_path.stat().st_size} bytes)")

        # ── Test A: Ensemble task with progress polling (optimized) ──
        print("\n[2] Submitting ensemble task (progress polling, 1.5s interval)...")
        task_a = _submit_task(img_path, mode="ensemble")
        task_id_a = task_a["task_id"]
        print(f"  Task ID: {task_id_a}, status: {task_a['status']}")

        status_a, poll_elapsed_a = _poll_until_done(task_id_a, use_progress=True, interval=1.5)
        print(f"  Poll completed: status={status_a}, poll_elapsed={poll_elapsed_a:.2f}s")

        # Fetch results once
        results_a, results_time_a = _fetch_results(task_id_a)
        total_a = poll_elapsed_a + results_time_a
        print(f"  Results fetch: {results_time_a:.3f}s")
        print(f"  TOTAL (progress poll + results): {total_a:.2f}s")

        # ── Test B: Ensemble task with heavy polling (baseline) ──
        # Clean DB for fresh test
        if db_path.exists():
            db_path.unlink()

        # Restart server for clean state
        server_proc.kill()
        server_proc.wait()
        time.sleep(1)

        print("\n[3] Restarting server for baseline test...")
        server_proc = subprocess.Popen(
            [VENV_PYTHON, "-m", "uvicorn", BACKEND_MODULE, "--host", "127.0.0.1", "--port", "18080"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        if not _wait_for_server("http://127.0.0.1:18080/api/v1/health"):
            print("FAIL: Server did not restart")
            server_proc.kill()
            sys.exit(1)
        print("  Server is ready.")

        print("\n[4] Submitting ensemble task (heavy task polling, 4s interval)...")
        task_b = _submit_task(img_path, mode="ensemble")
        task_id_b = task_b["task_id"]
        print(f"  Task ID: {task_id_b}, status: {task_b['status']}")

        status_b, poll_elapsed_b = _poll_until_done(task_id_b, use_progress=False, interval=4.0)
        print(f"  Poll completed: status={status_b}, poll_elapsed={poll_elapsed_b:.2f}s")

        # Fetch results
        results_b, results_time_b = _fetch_results(task_id_b)
        total_b = poll_elapsed_b + results_time_b
        print(f"  Results fetch: {results_time_b:.3f}s")
        print(f"  TOTAL (heavy poll + results): {total_b:.2f}s")

        # ── Summary ──
        print("\n" + "=" * 60)
        print("E2E Latency Comparison:")
        print(f"  Optimized (progress 1.5s):  {total_a:.2f}s")
        print(f"  Baseline   (task 4s):       {total_b:.2f}s")
        if total_b > 0:
            print(f"  Speedup:                    {total_b / total_a:.2f}x")
        print("=" * 60)

    finally:
        server_proc.kill()
        server_proc.wait()
        # Clean up
        if db_path.exists():
            db_path.unlink()
        print("\nServer stopped, test artifacts cleaned.")


if __name__ == "__main__":
    main()