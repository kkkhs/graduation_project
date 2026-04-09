# 系统最终交付检查清单

## 1. 启动入口

在仓库根目录执行：

```bash
source .venv/bin/activate
bash scripts/init_db.sh
bash scripts/start_backend.sh
bash scripts/start_frontend.sh
```

检查项：

- 前端页面：`http://127.0.0.1:5173`
- 健康接口：`http://127.0.0.1:8000/api/v1/health`

## 2. 最小闭环验收

使用固定样例池中的 `demo_easy_01`：

- 打开“任务提交”页
- 选择 `单图 + 单模型 + YOLO26`
- 上传 `assets/demo_cases/easy/...1536_16384.png`
- 提交任务并确认状态经历 `queued -> running -> done`
- 打开任务详情页，确认可视化结果与结构化结果表可见

## 3. 当前自动化验证

- Python 测试：`.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
- 当前结果基线：`15 tests OK`
- 已知非阻塞问题：FastAPI 生命周期仍存在 `on_event` 弃用告警

## 4. 前端回归与截图证据

- 提交页：`outputs/ui-submit-after-polish-20260315.png`
- 列表页：`outputs/ui-tasks-after-polish-20260315.png`
- 详情页：`outputs/ui-detail-fused-last-20260315.png`
- 完整详情页：`outputs/ui-detail-fused-last-full-20260315.png`

## 5. 失败兜底

若现场推理环境波动，按以下顺序切换静态截图：

1. `outputs/ui-submit-after-polish-20260315.png`
2. `outputs/ui-tasks-after-polish-20260315.png`
3. `outputs/ui-detail-fused-last-20260315.png`

口播保持为：

- 演示链路与系统在线流程一致
- 截图来自同一版本系统与同一任务链路
- 系统已具备真实模型接入、任务追踪与结果回看能力
