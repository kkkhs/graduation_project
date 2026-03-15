from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.db.models import ModelEntity, ResultEntity, TaskEntity, TaskFileEntity
from backend.app.schemas import (
    HealthResponse,
    ModelItem,
    ModelListResponse,
    ModelStats,
    ModelToggleRequest,
    ResultRecord,
    TaskCreateResponse,
    TaskListResponse,
    TaskResultImage,
    TaskResultsResponse,
    TaskSummary,
)

router = APIRouter(prefix="/api/v1", tags=["api"])



def get_session(request: Request) -> Session:
    session_factory = request.app.state.session_factory
    session = session_factory()
    try:
        yield session
    finally:
        session.close()



def _validate_task_params(task_type: str, mode: str, model_key: Optional[str], image_count: int, score_thr: float) -> None:
    if task_type not in {"single", "batch"}:
        raise HTTPException(status_code=400, detail="type must be single or batch")
    if mode not in {"single", "ensemble"}:
        raise HTTPException(status_code=400, detail="mode must be single or ensemble")
    if task_type == "single" and image_count != 1:
        raise HTTPException(status_code=400, detail="single type requires exactly one image")
    if task_type == "batch" and image_count < 1:
        raise HTTPException(status_code=400, detail="batch type requires at least one image")
    if mode == "single" and not model_key:
        raise HTTPException(status_code=400, detail="model_key is required when mode=single")
    if score_thr < 0 or score_thr > 1:
        raise HTTPException(status_code=400, detail="score_thr must be in [0,1]")



def _to_static_url(request: Request, file_path: str) -> str:
    outputs_root: Path = request.app.state.settings.outputs_root
    path = Path(file_path).resolve()
    try:
        relative = path.relative_to(outputs_root)
        return f"/static/{relative.as_posix()}"
    except ValueError:
        return str(path)


def _resolve_image_name_by_stem(candidates: dict[str, TaskResultImage], stem: str) -> str:
    for image_name in candidates.keys():
        if Path(image_name).stem == stem:
            return image_name
    return stem


@router.post("/tasks/infer", response_model=TaskCreateResponse)
def create_infer_task(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    task_type: Annotated[str, Form(alias="type")],
    mode: Annotated[str, Form()],
    model_key: Annotated[Optional[str], Form()] = None,
    score_thr: Annotated[float, Form()] = 0.25,
    images: Annotated[Optional[list[UploadFile]], File()] = None,
):
    upload_files = images or []
    _validate_task_params(task_type, mode, model_key, len(upload_files), score_thr)

    if mode == "single":
        row = session.query(ModelEntity).filter(ModelEntity.key == model_key).first()
        if row is None:
            raise HTTPException(status_code=404, detail=f"model not found: {model_key}")
        if not row.is_enabled:
            raise HTTPException(status_code=400, detail=f"model disabled: {model_key}")
        selected_models = [model_key]
    else:
        enabled = session.query(ModelEntity).filter(ModelEntity.is_enabled.is_(True)).all()
        selected_models = [item.key for item in enabled]
        if not selected_models:
            raise HTTPException(status_code=400, detail="no enabled models for ensemble mode")

    task = TaskEntity(
        type=task_type,
        status="queued",
        mode=mode,
        model_key=",".join(selected_models),
        score_thr=score_thr,
        input_count=len(upload_files),
        done_count=0,
        created_at=datetime.utcnow(),
    )
    session.add(task)
    session.commit()
    session.refresh(task)

    task_dir = request.app.state.settings.task_output_root / str(task.id)
    raw_input_dir = task_dir / "raw"
    raw_input_dir.mkdir(parents=True, exist_ok=True)

    for file in upload_files:
        if not file.filename:
            continue
        target = raw_input_dir / Path(file.filename).name
        with target.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        session.add(TaskFileEntity(task_id=task.id, kind="input", path=str(target.resolve())))
    session.commit()

    request.app.state.task_executor.submit(task.id)
    return TaskCreateResponse(task_id=task.id, status=task.status)


@router.get("/tasks", response_model=TaskListResponse)
def list_tasks(
    session: Annotated[Session, Depends(get_session)],
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    total = session.query(func.count(TaskEntity.id)).scalar() or 0
    rows = (
        session.query(TaskEntity)
        .order_by(TaskEntity.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return TaskListResponse(
        items=[TaskSummary.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/tasks/{task_id}", response_model=TaskSummary)
def get_task(task_id: int, session: Annotated[Session, Depends(get_session)]):
    row = session.get(TaskEntity, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskSummary.model_validate(row)


@router.get("/tasks/{task_id}/results", response_model=TaskResultsResponse)
def get_task_results(task_id: int, request: Request, session: Annotated[Session, Depends(get_session)]):
    task = session.get(TaskEntity, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")

    result_rows = session.query(ResultEntity).filter(ResultEntity.task_id == task_id).order_by(ResultEntity.id.asc()).all()
    file_rows = (
        session.query(TaskFileEntity)
        .filter(TaskFileEntity.task_id == task_id)
        .order_by(TaskFileEntity.id.asc())
        .all()
    )

    by_image: dict[str, TaskResultImage] = {}

    for row in result_rows:
        image_bucket = by_image.setdefault(row.image_name, TaskResultImage(image_name=row.image_name))
        image_bucket.records.append(
            ResultRecord(
                id=row.id,
                image_name=row.image_name,
                source_model=row.source_model,
                is_fused=row.is_fused,
                bbox=[row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2],
                score=row.score,
                category_id=row.category_id,
            )
        )

    for row in file_rows:
        if row.kind != "input":
            continue
        image_name = Path(row.path).name
        bucket = by_image.setdefault(image_name, TaskResultImage(image_name=image_name))
        bucket.input_url = _to_static_url(request, row.path)

    for row in file_rows:
        if row.kind == "input":
            continue

        name = Path(row.path).name
        stem = Path(name).stem
        if row.kind == "vis":
            vis_idx = stem.find("_vis")
            source_stem = stem[:vis_idx] if vis_idx >= 0 else stem
            image_name = _resolve_image_name_by_stem(by_image, source_stem)
        else:
            image_name = _resolve_image_name_by_stem(by_image, stem)

        bucket = by_image.setdefault(image_name, TaskResultImage(image_name=image_name))
        if row.kind == "vis":
            bucket.vis_urls.append(_to_static_url(request, row.path))
        elif row.kind == "output":
            bucket.output_urls.append(_to_static_url(request, row.path))

    model_stats: dict[str, list[float]] = {}
    for row in result_rows:
        model_stats.setdefault(row.source_model, []).append(row.score)

    by_model = [
        ModelStats(source_model=key, count=len(values), average_score=(sum(values) / len(values) if values else 0.0))
        for key, values in sorted(model_stats.items())
    ]

    scores = [row.score for row in result_rows]

    return TaskResultsResponse(
        task_id=task_id,
        images=[by_image[key] for key in sorted(by_image.keys())],
        total_objects=len(result_rows),
        average_score=(sum(scores) / len(scores) if scores else 0.0),
        by_model=by_model,
    )


@router.get("/models", response_model=ModelListResponse)
def list_models(session: Annotated[Session, Depends(get_session)]):
    rows = session.query(ModelEntity).order_by(ModelEntity.id.asc()).all()
    return ModelListResponse(items=[ModelItem.model_validate(row) for row in rows])


@router.patch("/models/{model_key}", response_model=ModelItem)
def toggle_model(model_key: str, payload: ModelToggleRequest, session: Annotated[Session, Depends(get_session)]):
    row = session.query(ModelEntity).filter(ModelEntity.key == model_key).first()
    if row is None:
        raise HTTPException(status_code=404, detail="model not found")
    row.is_enabled = payload.is_enabled
    session.commit()
    session.refresh(row)
    return ModelItem.model_validate(row)


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        config_path=str(settings.config_path),
        db_path=str(settings.db_path),
        outputs_root=str(settings.outputs_root),
        mock_inference=settings.mock_inference,
    )
