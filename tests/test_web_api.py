from __future__ import annotations

import io
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.settings import Settings
from backend.app.main import create_app


class WebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="gp_webapi_")
        root = Path(self._tmp.name)
        self.config_path = root / "models.yaml"
        self.db_path = root / "app.db"
        self.outputs_root = root / "outputs"
        self.task_root = self.outputs_root / "tasks"

        self.config_path.write_text(
            """
global:
  device: "cpu"
models:
  - model_name: "drenet"
    framework_type: "drenet"
    enabled: true
    weight_path: "/tmp/drenet.pth"
    config_path: "/tmp/drenet.py"
    class_names: ["ship"]
  - model_name: "yolo"
    framework_type: "ultralytics"
    enabled: true
    weight_path: "/tmp/yolo.pt"
    config_path: ""
    class_names: ["ship"]
""".strip(),
            encoding="utf-8",
        )

        settings = Settings(
            project_root=root,
            config_path=self.config_path,
            db_path=self.db_path,
            outputs_root=self.outputs_root,
            task_output_root=self.task_root,
            mock_inference=True,
            max_workers=1,
        )

        self.client = TestClient(create_app(settings))

    def tearDown(self) -> None:
        self.client.close()
        self._tmp.cleanup()

    @staticmethod
    def _image_bytes(color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
        image = Image.new("RGB", (160, 120), color)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return buf.getvalue()

    def _create_single_task(self) -> int:
        files = {"images": ("demo.jpg", self._image_bytes(), "image/jpeg")}
        payload = {"type": "single", "mode": "single", "model_key": "drenet", "score_thr": "0.25"}
        resp = self.client.post("/api/v1/tasks/infer", files=files, data=payload)
        self.assertEqual(resp.status_code, 200)
        return resp.json()["task_id"]

    def _wait_until_done(self, task_id: int, timeout_sec: float = 5.0) -> dict:
        start = time.time()
        while time.time() - start <= timeout_sec:
            resp = self.client.get(f"/api/v1/tasks/{task_id}")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            if data["status"] in {"done", "failed"}:
                return data
            time.sleep(0.1)
        self.fail("task did not finish in time")

    def test_infer_param_validation(self) -> None:
        bad1 = self.client.post(
            "/api/v1/tasks/infer",
            data={"type": "single", "mode": "single", "score_thr": "0.25"},
            files={"images": ("a.jpg", self._image_bytes(), "image/jpeg")},
        )
        self.assertEqual(bad1.status_code, 400)

        bad2 = self.client.post(
            "/api/v1/tasks/infer",
            data={"type": "single", "mode": "ensemble", "score_thr": "0.25"},
            files=[
                ("images", ("a.jpg", self._image_bytes((255, 0, 0)), "image/jpeg")),
                ("images", ("b.jpg", self._image_bytes((0, 255, 0)), "image/jpeg")),
            ],
        )
        self.assertEqual(bad2.status_code, 400)

    def test_task_status_flow(self) -> None:
        task_id = self._create_single_task()
        data = self._wait_until_done(task_id)
        self.assertEqual(data["status"], "done")
        self.assertEqual(data["done_count"], 1)
        self.assertEqual(data["input_count"], 1)

    def test_result_persistence_consistency(self) -> None:
        task_id = self._create_single_task()
        data = self._wait_until_done(task_id)
        self.assertEqual(data["status"], "done")

        result_resp = self.client.get(f"/api/v1/tasks/{task_id}/results")
        self.assertEqual(result_resp.status_code, 200)
        result = result_resp.json()

        self.assertGreaterEqual(result["total_objects"], 1)
        self.assertEqual(result["task_id"], task_id)
        self.assertTrue(len(result["images"]) >= 1)
        self.assertEqual(result["images"][0]["image_name"], "demo.jpg")


if __name__ == "__main__":
    unittest.main()
