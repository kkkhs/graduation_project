# 计划：论文收口与系统默认权重落地

状态: planned
执行: not started

## 1. 背景
- 已完成：
  - DRENet、FCOS、YOLO26 三模型正式结果
  - 三模型统一阈值消融
  - `FCOS / YOLO26` 输入尺寸敏感性分析
- 当前阶段：从“实验补齐”切换到“论文收口 + 系统默认权重落地 + 答辩演示整理”。

## 2. 本轮目标
1. 完成论文第四章、第五章、摘要与总结章节的统一回填。
2. 完成系统默认权重策略落地（`global-best` vs `stable-best`）。
3. 完成答辩演示样例与图表清单整理。
4. 将 FCOS 的“参考基线”定位与输入口径差异，在论文和系统文档中统一写清楚。

## 2.1 已有事实记录（上一阶段）
- FCOS 正式 run（已完成）：
  - `docs/experiments/logs/run-20260315-fcos-wandb-single-run-epoch-log.md`
- 本地消融日志（已完成）：
  - `docs/experiments/logs/run-20260318-阈值消融正式执行.md`
  - `docs/experiments/logs/run-20260318-尺寸敏感性正式执行.md`

## 3. 开跑前检查（必须通过）
- 进入论文与系统收口阶段前，再次确认：
  - `docs/results/` 三个结果文档已为最新
  - 论文中不存在“消融待补”的旧表述
  - `configs/models.yaml` 的默认权重策略与文档一致
  - 实验日志与关键 checkpoint 路径仍可追溯

## 4. 执行顺序
1. 回填论文第四章、第五章、摘要与结论文字。
2. 完成系统默认模型选择 A/B 记录并更新配置。
3. 整理答辩图表、定性样例与演示清单。
4. 如仍有余量，再补融合模式统一离线评测说明。

## 5. 留痕与产物要求
- 文本回填记录：
  - `docs/results/*.md`
  - `thesis_overleaf/chapters/*.tex`
- 系统默认权重记录：
  - `configs/models.yaml`
  - 相关系统文档与实验说明
- 图表与样例：
  - `assets/figures/`
  - `docs/results/qualitative.md`

## 6. 风险与应对
- 文本口径不一致：以 `docs/results/` 与 `docs/spec_todo.md` 为唯一事实源统一修正。
- 系统默认权重与论文口径混淆：明确区分 `global-best` 与 `stable-best`，并分别写入文档。
- 图表过多导致答辩重点分散：优先保留主对比表、两组消融表和三类定性样例。
