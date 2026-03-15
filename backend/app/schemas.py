from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


TaskType = Literal["single", "batch"]
TaskMode = Literal["single", "ensemble"]
TaskStatus = Literal["queued", "running", "done", "failed"]


class TaskCreateResponse(BaseModel):
    task_id: int
    status: str


class TaskSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    type: str
    status: str
    mode: str
    model_key: Optional[str]
    score_thr: float
    input_count: int
    done_count: int
    error_code: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class TaskListResponse(BaseModel):
    items: list[TaskSummary]
    total: int
    page: int
    page_size: int


class ResultRecord(BaseModel):
    id: int
    image_name: str
    source_model: str
    is_fused: bool
    bbox: list[float]
    score: float
    category_id: int


class TaskResultImage(BaseModel):
    image_name: str
    input_url: Optional[str] = None
    vis_urls: list[str] = Field(default_factory=list)
    output_urls: list[str] = Field(default_factory=list)
    records: list[ResultRecord] = Field(default_factory=list)


class ModelStats(BaseModel):
    source_model: str
    count: int
    average_score: float


class TaskResultsResponse(BaseModel):
    task_id: int
    images: list[TaskResultImage]
    total_objects: int
    average_score: float
    by_model: list[ModelStats]


class ModelItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    key: str
    weight_path: str
    is_enabled: bool
    created_at: datetime


class ModelListResponse(BaseModel):
    items: list[ModelItem]


class ModelToggleRequest(BaseModel):
    is_enabled: bool


class HealthResponse(BaseModel):
    status: str
    config_path: str
    db_path: str
    outputs_root: str
    mock_inference: bool
