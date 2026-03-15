# Web 系统测试报告（2026-03-15）

## 1. 环境
- Python: 3.9 (`.venv`)
- Backend: FastAPI + SQLite
- Frontend: React + Vite + TypeScript + AntD
- 数据源: `experiment_assets/datasets/LEVIR-Ship/test/images`

## 2. 执行命令
```bash
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_*.py'
cd frontend && npm run build

./scripts/start_backend.sh
./scripts/start_frontend.sh
```

## 3. 自动化结果
- Python 测试：通过
- 前端构建：通过（`vite build success`）
- 接口健康检查：`/api/v1/health` 返回 `ok`

## 4. 实测任务（真实推理）
- task `15`：`single + yolo`，1 张图，状态 `done`
- task `16`：`single + ensemble`，1 张图，状态 `done`
- task `17`：`single + ensemble`，1 张图，状态 `done`

结果验证点：
- 任务状态机流转正常（queued -> running -> done）
- 可视化产物正常落盘并可回放
- `results/task_files` 入库与查询一致

## 5. 前端回归（本轮）
- 中文化关键词映射生效（类型/模式/状态）
- 任务列表与任务详情进度展示统一为“进度条 + 数量比”
- 类型与模式标签颜色分离，提升辨识度
- 任务详情中融合结果统一置后（图表、标签、结构化表格）
- 顶部导航选中文字为蓝色；右上角移除时区块并保留状态颜色

## 6. 证据截图
- 提交页：`outputs/ui-submit-no-chip-20260315.png`
- 列表页：`outputs/ui-tasks-progress-right-20260315.png`
- 详情页：`outputs/ui-detail-fused-last-full-20260315.png`
- 移动端：`outputs/ui-submit-mobile-20260315.png`

## 7. 结论
系统已满足毕设阶段“轻量但完整”交付要求：前端、后端、数据库、真实推理、任务追踪与可视化回看全部可用。
