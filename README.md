# 毕设项目总览

题目：《遥感场景下的微小船舶检测系统设计与实现》

这个仓库是毕设工作台，目标是把论文、实验和系统实现放在同一套可追溯流程里推进。

## 1. 当前状态（2026-03-15）

- Web 系统主链路已打通：`React + FastAPI + SQLite + 真实模型推理`
- DRENet / YOLO 权重已接入 `models.yaml`，支持真实单图与批量推理
- 任务页支持中文语义、进度条+数量比、失败/完成颜色区分
- 任务详情已统一“融合结果在最后”（图表、标签、结构化表格）
- FCOS 训练实验仍待补全（不影响当前 Web 系统演示）

## 2. 近期重点（按顺序）

1. 完成 FCOS 冒烟、正式训练、评测与回填
2. 补齐三模型统一口径：`AP50 / AP50-95 / P / R / F1 / FPS / Params / FLOPs`
3. 完成 `docs/results/` 的定性与消融
4. 回填论文第四、第五章实验部分

## 3. 一眼看板

- 总看板： [spec_todo.md](/Users/khs/codes/graduation_project/docs/spec_todo.md)
- 阶段总结： [progress_summary_20260308.md](/Users/khs/codes/graduation_project/docs/experiments/progress_summary_20260308.md)

建议：日常只看 `spec_todo.md` 判断“下一步做什么”。

## 4. 核心文档入口

### 实验执行
- 实验总入口： [experiments/README.md](/Users/khs/codes/graduation_project/docs/experiments/README.md)
- 云端手册（AutoDL）： [cloud_execution_playbook.md](/Users/khs/codes/graduation_project/docs/experiments/cloud_execution_playbook.md)
- 本地 3060 手册： [3060_execution_playbook.md](/Users/khs/codes/graduation_project/docs/experiments/3060_execution_playbook.md)
- YOLO26 实跑日志： [run-20260308-yolo26-3060-3080ti.md](/Users/khs/codes/graduation_project/docs/experiments/logs/run-20260308-yolo26-3060-3080ti.md)
- DRENet 实跑日志： [exp-20260308-02-drenet-formal-resume-300.md](/Users/khs/codes/graduation_project/docs/experiments/logs/exp-20260308-02-drenet-formal-resume-300.md)

### 结果回填
- 主对比表： [baselines.md](/Users/khs/codes/graduation_project/docs/results/baselines.md)
- 消融表： [ablation.md](/Users/khs/codes/graduation_project/docs/results/ablation.md)
- 定性分析： [qualitative.md](/Users/khs/codes/graduation_project/docs/results/qualitative.md)

### 系统实现
- 架构： [architecture.md](/Users/khs/codes/graduation_project/docs/system/architecture.md)
- 推理接口契约： [predict_api_contract.md](/Users/khs/codes/graduation_project/docs/system/predict_api_contract.md)
- 产物接入规范： [artifact_integration_spec.md](/Users/khs/codes/graduation_project/docs/system/artifact_integration_spec.md)

## 5. 目录作用（高频）

- `docs/`：实验、论文、系统文档主目录
- `src/`：算法与推理内核（分层架构，给 Web 后端和脚本共用）
- `tools/`：可执行入口（推理、可视化、UI、转换脚本）
- `configs/`：模型配置
- `tests/`：测试
- `scripts/`：同步、快照、运维脚本
- `experiment_assets/`：训练产物与配置快照（本地归档）
- `assets/`：长期保留图件
- `outputs/`：临时运行输出（已忽略，不作正式交付）

关于“为什么还有 `src/`”：
- `backend/` 主要负责 Web API、任务队列、数据库与文件路由。
- `src/` 承载模型调度、融合逻辑、适配器、可视化渲染等核心能力。
- 目前 `backend` 是对 `src` 的服务化封装，不是替代关系；保留 `src` 可以避免算法代码在脚本端和 Web 端重复维护。

## 6. 训练产物约定（当前执行标准）

- run 目录：`experiment_assets/runs/<run_name>`
- trace 日志按功能分层：`experiment_assets/runs/trace/{drenet,yolo,sync,data,docker,infra,misc}`
- 数据 YAML 快照：`experiment_assets/configs/<run_name>/ship_autodl.yaml`
- 若历史路径为 `runs/detect/runs/<run_name>`，允许保留兼容软链接

本次 YOLO26 本地产物：
- run： [yolo26_main_512_formal012](/Users/khs/codes/graduation_project/experiment_assets/runs/yolo26_main_512_formal012)
- trace： [trace/yolo](/Users/khs/codes/graduation_project/experiment_assets/runs/trace/yolo)
- data yaml 快照： [ship_autodl.yaml](/Users/khs/codes/graduation_project/experiment_assets/configs/yolo26_main_512_formal012/ship_autodl.yaml)

## 7. 如何开始下一步（FCOS）

1. 按云端手册完成 FCOS 冒烟与正式训练
2. 将结果回填到 `docs/results/baselines.md`
3. 写一份对应 run log 到 `docs/experiments/logs/`
4. 更新 `spec_todo.md` 任务状态

## 8. 默认实验口径

- 数据集：LEVIR-Ship
- 三模型路线：DRENet / FCOS / YOLO26
- 主指标：`AP50`
- 辅指标：`AP50-95 / Precision / Recall / F1`
- 效率与复杂度：`FPS / Params / FLOPs`

## 9. Web 系统入口（React + FastAPI + SQLite）

系统升级为 Web 版本后，入口如下：

- 后端：`backend/`（FastAPI + SQLAlchemy + Alembic + SQLite）
- 前端：`frontend/`（React + Vite + TypeScript + AntD + ECharts）
- 任务输出：`outputs/tasks/<task_id>/`

快速启动：

1. `pip install -r backend/requirements.txt`
2. `./scripts/init_db.sh`
3. `./scripts/start_backend.sh`
4. `./scripts/start_frontend.sh`

或一键：

- `./scripts/dev_one_click.sh`

详细说明见： [web_system_upgrade.md](/Users/khs/codes/graduation_project/docs/system/web_system_upgrade.md)

联调与验证记录见： [test_report_20260315.md](/Users/khs/codes/graduation_project/docs/system/test_report_20260315.md)
