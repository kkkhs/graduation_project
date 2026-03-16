# 后端服务说明

后端基于 FastAPI + SQLAlchemy + SQLite，负责：
- 任务创建与调度
- 模型状态管理
- 结果查询与文件路由

## 关键入口
- 启动：`./scripts/start_backend.sh`
- 初始化数据库：`./scripts/init_db.sh`
- API 路由：`backend/app/api/routes.py`

## 主要模块
- `backend/app/services/task_executor.py`：任务执行与推理调度
- `backend/app/services/inference_runtime.py`：推理引擎入口
- `backend/app/db/`：数据库模型与会话
