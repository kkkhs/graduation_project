# 系统详细流程说明（Web 端全链路）

本文件面向“从用户提交到结果展示”的完整流程，强调 **调用链、数据落盘、日志位置**，便于排障与论文写作。

---

## 1. 入口与启动顺序

### 1.1 启动后端
```bash
./scripts/start_backend.sh
```
- 端口：`http://localhost:8000`
- 依赖：`backend/requirements.txt`
- 说明：后端启动后会暴露 `/api/v1/*` 接口。

### 1.2 启动前端
```bash
./scripts/start_frontend.sh
```
- 端口：`http://localhost:5173`
- 说明：Vite dev server。

### 1.3 同步模型注册表
```bash
./scripts/init_db.sh
```
- 功能：将 `configs/models.yaml` 同步到 DB 的 `models` 表。
- 说明：如果模型列表为空，优先检查是否执行过该步骤。

---

## 2. 前端提交流程（用户视角）

1. 打开 `http://localhost:5173/`。
2. 选择：
   - 任务类型（单图/批量）
   - 推理模式（单模型/融合）
   - 置信度阈值 `score_thr`
   - 单模型模式下选择模型（`drenet / mmdet_fcos / yolo`）
3. 上传图片并点击「提交任务」。

前端会发起：
```
POST /api/v1/tasks/infer
```
表单字段：
```
type, mode, model_key(单模型), score_thr, images[]
```

---

## 3. 后端接收与任务创建

入口函数：`backend/app/api/routes.py:create_infer_task`

主要步骤：
1. 参数校验：模式/图片数量/阈值范围。
2. 选模型：
   - 单模型：必须存在且启用
   - 融合：取全部启用模型
3. 写入 DB：
   - `tasks` 表新增一条
   - `task_files` 记录上传图片路径
4. 原图落盘：
```
outputs/tasks/<task_id>/raw/<filename>
```
5. 将任务提交给 `TaskExecutor`。

---

## 4. 推理引擎调用链（后端执行）

执行入口：`backend/app/services/task_executor.py:_run_task`

**调用链（核心）**
```
TaskExecutor
  → InferenceRuntime.predict_single / predict_ensemble
    → AdapterFactory.create(model_config)
      → DRENetAdapter / MMDetAdapter / YOLOAdapter
        → 真实模型推理（PyTorch / MMDet / Ultralytics）
```

**推理输出统一结构**
```
[
  {
    "bbox": [x, y, w, h],
    "score": float,
    "category_id": int,
    "model_name": str
  },
  ...
]
```

---

## 5. 产物落盘与数据库写入

### 5.1 JSON 结果
```
outputs/tasks/<task_id>/output/<image>.json
```
内容：
```json
{
  "per_model": [...],
  "fused": [...]
}
```

### 5.2 可视化图
```
outputs/tasks/<task_id>/vis/<image>_vis_<model>.png
outputs/tasks/<task_id>/vis/<image>_vis_fused.png (融合模式)
```
可视化方法：`src/infrastructure/visualization.py:render_detections`

### 5.3 结构化结果写入 DB
写入表：`results`
字段：`bbox_x1, bbox_y1, bbox_x2, bbox_y2, score, source_model` 等

---

## 6. 任务日志（新增）

每个任务独立日志：
```
outputs/tasks/<task_id>/task.log
```
记录内容：
- 任务开始：模型 / 阈值 / 输入数量
- 每张图：预测数量、耗时、可视化路径
- 任务结束：状态与总耗时
- 异常堆栈（若失败）

示例：
```
[START] task_id=19 mode=single model_key=mmdet_fcos score_thr=0.25 inputs=1
[IMAGE] name=... per_model=6 fused=0 vis=... duration_s=5.781
[END] task_id=19 status=done errors=- duration_s=5.785
```

---

## 7. 前端结果展示流程

前端在任务详情页轮询：
```
GET /api/v1/tasks/{id}
GET /api/v1/tasks/{id}/results
```

展示内容：
- 状态、进度条
- 统计：检测总数 / 平均置信度
- 可视化图：输入图 + 各模型结果 + 融合结果
- 表格：每个框的结构化数据

---

## 8. 关键排障点

### 8.1 任务完成但结果为空
- 先看 `outputs/tasks/<id>/task.log` 的 `per_model` 数量
- 若为 0：
  - 可能阈值过高
  - 或模型在该图无置信输出

### 8.2 模型列表为空
- 执行 `./scripts/init_db.sh` 同步模型

### 8.3 MMDet/FCOS 无法加载
- 检查 `mmengine/mmcv/mmdet` 是否安装
- macOS 下 `mmcv` 可能编译慢或失败，建议使用 Linux/云端

---

## 9. 相关入口索引

- 架构说明：`docs/system/architecture.md`
- API 契约：`docs/system/predict_api_contract.md`
- Web 启动说明：`docs/system/web_system_upgrade.md`
- 本地推理入口：`docs/system/local_mvp_run.md`
