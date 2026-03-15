from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.settings import Settings, load_settings
from backend.app.db.models import Base
from backend.app.db.session import build_engine, build_session_factory
from backend.app.services.inference_runtime import InferenceRuntime
from backend.app.services.model_registry import sync_models
from backend.app.services.task_executor import TaskExecutor



def create_app(settings: Optional[Settings] = None) -> FastAPI:
    cfg = settings or load_settings()
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.outputs_root.mkdir(parents=True, exist_ok=True)
    cfg.task_output_root.mkdir(parents=True, exist_ok=True)

    engine = build_engine(str(cfg.db_path))
    session_factory = build_session_factory(engine)

    # Keep app resilient when DB isn't initialized via alembic yet.
    Base.metadata.create_all(bind=engine)
    with session_factory() as session:
        sync_models(session, cfg.config_path)

    runtime = InferenceRuntime(cfg)
    task_executor = TaskExecutor(settings=cfg, session_factory=session_factory, runtime=runtime)

    app = FastAPI(title="Graduation Project Web API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = cfg
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.runtime = runtime
    app.state.task_executor = task_executor

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(cfg.outputs_root)), name="static")

    @app.on_event("shutdown")
    def _shutdown() -> None:
        app.state.task_executor.shutdown()

    return app


app = create_app()
