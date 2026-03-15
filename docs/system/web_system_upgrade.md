# Web 系统升级说明（React + FastAPI + SQLite）

更新时间：2026-03-15

## 1. 目标与范围
- 目标：把原 Python 可视化入口升级为可演示、可回溯、可持久化的完整 Web 系统。
- 范围：本地单机运行，不引入鉴权、Redis/Celery、多机部署。

## 2. 目录结构与职责
- `frontend/`：React + Vite + TypeScript + AntD + ECharts，负责页面交互与可视化展示。
- `backend/`：FastAPI + SQLAlchemy + SQLite，负责 API、任务调度、状态管理与静态文件路由。
- `src/`：算法与推理内核（模型调度、融合逻辑、适配器、可视化渲染），由 `backend` 复用。
- `outputs/tasks/<task_id>/`：任务输入/输出/可视化产物落盘目录。

说明：`src/` 不是历史冗余目录，当前仍是系统核心能力层，`backend` 只是服务化封装层。

## 3. 后端接口（`/api/v1`）
- `POST /tasks/infer`
  - `multipart/form-data`
  - 字段：`type(single|batch)`、`mode(ensemble|single)`、`model_key(单模型必填)`、`score_thr`、`images[]`
  - 返回：`{task_id,status}`
- `GET /tasks`
- `GET /tasks/{task_id}`
- `GET /tasks/{task_id}/results`
- `GET /models`
- `PATCH /models/{model_key}`
- `GET /health`

## 4. 数据库（SQLite）
数据库：`app.db`

核心表：
- `models`
- `tasks`
- `results`
- `task_files`

ER 图见：`docs/system/database_er.md`

## 5. 当前前端交互规范（已落地）
- 关键词中文化：状态/模式/类型统一中文展示。
- 列表与详情统一进度样式：`进度条 + done/total`（数量比位于进度条右侧）。
- 类型与模式标签分色：
  - 类型：`单图(蓝)`、`批量(橙)`
  - 模式：`融合模式(紫)`、`单模型(青绿)`
- 状态语义色：`已完成(绿)`、`失败(红)`。
- 任务详情展示顺序：融合结果固定置后（图表、标签、结构化结果表）。
- 顶部导航：选中文字为蓝色，无额外选中背景。
- 右上角信息：移除时区块，仅保留执行器与状态，状态使用颜色强调。

## 6. 本地启动
1. 安装依赖
```bash
pip install -r backend/requirements.txt
```
2. 初始化数据库
```bash
./scripts/init_db.sh
```
3. 启动后端
```bash
./scripts/start_backend.sh
```
4. 启动前端
```bash
./scripts/start_frontend.sh
```

一键启动：
```bash
./scripts/dev_one_click.sh
```

## 7. 验证与证据
- 接口与联调：`docs/system/test_report_20260315.md`
- UI 回归截图：`outputs/ui-*.png`
