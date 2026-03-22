# 毕设项目总览

题目：《遥感场景下的微小船舶检测系统设计与实现》

这个仓库是毕设工作台，目标是把论文、实验和系统实现放在同一套可追溯流程里推进。

## 1. 当前状态（2026-03-19）

- 三模型主对比已完成：`DRENet / FCOS / YOLO26`
- 两组离线消融已完成：
  - 三模型统一阈值消融
  - `FCOS / YOLO26` 输入尺寸敏感性分析
- 定性分析、效率指标、关键 checkpoint 与实验日志已回填到仓库
- Web 系统主链路已打通：`React + FastAPI + SQLite + 真实模型推理`
- 三模型权重均已接入统一推理链路，支持真实单图与批量推理
- 论文第四、第五章已进入成稿收口阶段，正式图片与生成脚本已补齐
- 当前重点已从“补训练”切换到“论文收口 + 口径统一 + 系统默认权重策略 + 答辩演示材料”

## 2. 近期重点（按顺序）

1. 回填论文第四、第五章、摘要与总结章节
2. 固化口径说明与系统默认权重策略：区分 FCOS 论文主对比口径与系统/消融口径
3. 准备答辩与系统演示样例（易/中/难 + success/miss/false_positive）
4. 整理论文成稿资产说明，清理中间脚本与图片版本
5. 若时间允许，再补融合模式统一离线评测或系统默认参数说明

## 3. 一眼看板

- 总看板： [spec_todo.md](/Users/khs/codes/graduation_project/docs/spec_todo.md)
- 当前阶段总结： [progress_summary_20260318.md](/Users/khs/codes/graduation_project/docs/experiments/progress_summary_20260318.md)
- 历史阶段总结： [progress_summary_20260315.md](/Users/khs/codes/graduation_project/docs/experiments/progress_summary_20260315.md)

建议：日常只看 `spec_todo.md` 判断“下一步做什么”。

## 4. 核心文档入口

### 实验执行
- 实验总入口： [experiments/README.md](/Users/khs/codes/graduation_project/docs/experiments/README.md)
- 云端手册（AutoDL）： [cloud_execution_playbook.md](/Users/khs/codes/graduation_project/docs/experiments/cloud_execution_playbook.md)
- 本地 3060 手册： [3060_execution_playbook.md](/Users/khs/codes/graduation_project/docs/experiments/3060_execution_playbook.md)
- YOLO26 实跑日志： [run-20260308-yolo26-3060-3080ti.md](/Users/khs/codes/graduation_project/docs/experiments/logs/run-20260308-yolo26-3060-3080ti.md)
- DRENet 实跑日志： [exp-20260308-02-drenet-formal-resume-300.md](/Users/khs/codes/graduation_project/docs/experiments/logs/exp-20260308-02-drenet-formal-resume-300.md)
- FCOS 正式 run 日志： [run-20260315-fcos-wandb-single-run-epoch-log.md](/Users/khs/codes/graduation_project/docs/experiments/logs/run-20260315-fcos-wandb-single-run-epoch-log.md)
- 本地消融日志： [run-20260318-阈值消融正式执行.md](/Users/khs/codes/graduation_project/docs/experiments/logs/run-20260318-阈值消融正式执行.md)
- 本地尺寸敏感性日志： [run-20260318-尺寸敏感性正式执行.md](/Users/khs/codes/graduation_project/docs/experiments/logs/run-20260318-尺寸敏感性正式执行.md)

### 结果回填
- 主对比表： [baselines.md](/Users/khs/codes/graduation_project/docs/results/baselines.md)
- 消融表： [ablation.md](/Users/khs/codes/graduation_project/docs/results/ablation.md)
- 定性分析： [qualitative.md](/Users/khs/codes/graduation_project/docs/results/qualitative.md)
- 结果目录说明： [README.md](/Users/khs/codes/graduation_project/docs/results/README.md)

### 论文成稿资产
- 论文主入口： [main.tex](/Users/khs/codes/graduation_project/thesis_overleaf/main.tex)
- 第四章实验结果与分析： [citations.tex](/Users/khs/codes/graduation_project/thesis_overleaf/chapters/citations.tex)
- 第五章系统设计与实现： [publications.tex](/Users/khs/codes/graduation_project/thesis_overleaf/chapters/publications.tex)
- 成稿图片清单： [README.md](/Users/khs/codes/graduation_project/thesis_overleaf/figures/generated/README.md)

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

## 7. 如何开始下一步（当前推荐）

1. 先看 `docs/spec_todo.md`，确认还未收口的论文与系统任务。
2. 再看 `docs/experiments/progress_summary_20260318.md`，把当前阶段事实和边界条件过一遍。
3. 统一论文第四章、结果文档和系统文档中的 FCOS 口径说明，避免主表、系统演示和消融分析互相混写。
4. 回填论文第四章、第五章、摘要与总结，保证表格、文字和实验日志口径一致。
5. 固化系统默认权重策略，并整理答辩演示样例。

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
