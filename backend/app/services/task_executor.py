from __future__ import annotations

import re
import json
import shutil
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.core.settings import Settings
from backend.app.db.models import ResultEntity, TaskEntity, TaskFileEntity
from backend.app.db.session import SessionFactory
from backend.app.services.inference_runtime import InferenceRuntime
from src.infrastructure.visualization import render_detections


class TaskExecutor:
    def __init__(self, settings: Settings, session_factory: SessionFactory, runtime: InferenceRuntime) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.runtime = runtime
        self._pool = ThreadPoolExecutor(max_workers=settings.max_workers, thread_name_prefix="task-executor")

    def submit(self, task_id: int) -> Future:
        return self._pool.submit(self._run_task, task_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run_task(self, task_id: int) -> None:
        session: Session = self.session_factory()
        try:
            task = session.get(TaskEntity, task_id)
            if task is None:
                return

            task.status = "running"
            task.started_at = datetime.utcnow()
            task.error_code = None
            task.error_message = None
            session.commit()

            task_dir = self.settings.task_output_root / str(task_id)
            input_dir = task_dir / "input"
            output_dir = task_dir / "output"
            vis_dir = task_dir / "vis"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            vis_dir.mkdir(parents=True, exist_ok=True)

            input_files = (
                session.query(TaskFileEntity)
                .filter(TaskFileEntity.task_id == task_id, TaskFileEntity.kind == "input")
                .all()
            )

            selected_model_keys = self._resolve_model_keys(task)
            if task.mode == "single" and not selected_model_keys:
                raise ValueError("single mode requires model_key")

            image_errors: list[str] = []

            for idx, input_row in enumerate(input_files, start=1):
                src_path = Path(input_row.path)
                if not src_path.exists():
                    image_errors.append(f"missing input file: {src_path}")
                    task.done_count = idx
                    session.commit()
                    continue

                image_name = src_path.name
                local_input = input_dir / image_name
                if src_path.resolve() != local_input.resolve():
                    shutil.copy2(src_path, local_input)

                if task.mode == "ensemble":
                    per_model, fused = self.runtime.predict_ensemble(
                        image_path=str(local_input),
                        model_keys=selected_model_keys,
                        score_thr=task.score_thr,
                    )
                    self._insert_rows(session, task_id, image_name, per_model, is_fused=False)
                    self._insert_rows(session, task_id, image_name, fused, is_fused=True)
                    payload = {"per_model": per_model, "fused": fused}

                    for model_name, rows in self._group_predictions_by_model(per_model).items():
                        vis_path = vis_dir / f"{local_input.stem}_vis_{self._safe_file_token(model_name)}.png"
                        render_detections(str(local_input), rows, str(vis_path))
                        session.add(TaskFileEntity(task_id=task_id, kind="vis", path=str(vis_path.resolve())))

                    fused_vis_path = vis_dir / f"{local_input.stem}_vis_fused.png"
                    render_detections(str(local_input), fused, str(fused_vis_path))
                    session.add(TaskFileEntity(task_id=task_id, kind="vis", path=str(fused_vis_path.resolve())))
                else:
                    model_key = selected_model_keys[0]
                    per_model = self.runtime.predict_single(
                        image_path=str(local_input),
                        model_key=model_key,
                        score_thr=task.score_thr,
                    )
                    self._insert_rows(session, task_id, image_name, per_model, is_fused=False)
                    payload = {"per_model": per_model, "fused": []}

                    vis_path = vis_dir / f"{local_input.stem}_vis_{self._safe_file_token(model_key)}.png"
                    render_detections(str(local_input), per_model, str(vis_path))
                    session.add(TaskFileEntity(task_id=task_id, kind="vis", path=str(vis_path.resolve())))

                out_json = output_dir / f"{local_input.stem}.json"
                out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                session.add(TaskFileEntity(task_id=task_id, kind="output", path=str(out_json.resolve())))

                task.done_count = idx
                session.commit()

            task.finished_at = datetime.utcnow()
            if image_errors:
                task.status = "failed"
                task.error_code = "IMAGE_PROCESS_FAILED"
                task.error_message = "; ".join(image_errors)[:4000]
            else:
                task.status = "done"
            session.commit()

        except Exception as exc:  # noqa: BLE001
            rollback_task = session.get(TaskEntity, task_id)
            if rollback_task is not None:
                rollback_task.status = "failed"
                rollback_task.error_code = "TASK_RUNTIME_ERROR"
                rollback_task.error_message = str(exc)[:4000]
                rollback_task.finished_at = datetime.utcnow()
                session.commit()
        finally:
            session.close()

    @staticmethod
    def _insert_rows(session: Session, task_id: int, image_name: str, rows: list[dict], is_fused: bool) -> None:
        for row in rows:
            bbox = row.get("bbox", [0, 0, 0, 0])
            x = float(bbox[0]) if len(bbox) > 0 else 0.0
            y = float(bbox[1]) if len(bbox) > 1 else 0.0
            w = float(bbox[2]) if len(bbox) > 2 else 0.0
            h = float(bbox[3]) if len(bbox) > 3 else 0.0
            session.add(
                ResultEntity(
                    task_id=task_id,
                    image_name=image_name,
                    source_model=str(row.get("model_name", "ensemble" if is_fused else "unknown")),
                    is_fused=is_fused,
                    bbox_x1=x,
                    bbox_y1=y,
                    bbox_x2=x + w,
                    bbox_y2=y + h,
                    score=float(row.get("score", 0.0)),
                    category_id=int(row.get("category_id", 0)),
                )
            )

    @staticmethod
    def _resolve_model_keys(task: TaskEntity) -> list[str]:
        if not task.model_key:
            return []
        return [x.strip() for x in task.model_key.split(",") if x.strip()]

    @staticmethod
    def _group_predictions_by_model(rows: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            model_name = str(row.get("model_name") or "unknown")
            grouped[model_name].append(row)
        return dict(sorted(grouped.items(), key=lambda item: item[0]))

    @staticmethod
    def _safe_file_token(value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
        cleaned = cleaned.strip("._-")
        return cleaned or "unknown"
