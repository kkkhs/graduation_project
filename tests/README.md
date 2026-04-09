# tests 目录说明

本目录存放 Python 侧的自动化测试，目标是保证“统一推理链路 + Web API 主流程”在改动后可快速回归。

## 测试文件与作用

- `test_predict_service.py`
  - 覆盖 `PredictService` 的核心行为：
  - 未知模型报错
  - 阈值参数边界校验
  - 多模型融合输出（`model_name=ensemble`）
- `test_web_api.py`
  - 覆盖 Web API 关键链路：
  - `POST /api/v1/tasks/infer` 参数校验
  - 任务状态流转（`queued/running/done|failed`）
  - 结果落库与查询一致性
  - 说明：该文件依赖 `fastapi.testclient` 与 `Pillow`，缺失时会自动 `skip`
- `test_fusion.py`
  - 覆盖融合逻辑的基础行为与稳定性
- `test_adapters_errors.py`
  - 覆盖适配器错误处理（异常与边界场景）
- `test_compat_imports.py`
  - 覆盖兼容导入路径，避免重构后旧入口失效

## 快速运行

在仓库根目录执行：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

如果只想跑某个文件：

```bash
.venv/bin/python -m unittest tests.test_web_api
```

## 当前结果基线（2026-04-09）

- 在本地 `.venv` 环境：`15 tests OK`
- 可能看到 FastAPI 的 `on_event` 弃用告警（不影响通过）

## 新增测试建议

新增测试时建议遵循：

1. 文件命名用 `test_*.py`
2. 每个测试只覆盖一个明确行为
3. 先补失败用例，再补实现
4. 对外部依赖（如模型、GPU）尽量用 mock，保证可在 CPU 环境跑通

