# 计划：下一轮实验执行（结果回填与论文收口优先）

状态: 计划中
执行: 下一阶段（待 FCOS 首轮正式 run 完成后）

## 1. 背景
- 已完成：DRENet、YOLO26、FCOS 首轮正式结果。
- 当前阶段：从“训练打通”切换到“论文回填 + 系统默认权重落地 + 二阶段决策”。

## 2. 本轮目标
1. 完成三模型主对比表（含效率项）回填与结论收敛。
2. 完成系统默认权重策略落地（`global-best` vs `stable-best`）。
3. 完成定性图与最小消融，形成论文可直接引用材料。
4. 将 FCOS 明确降级为“参考基线”，不再继续投入重训成本。

## 2.1 已有事实记录（上一阶段）
- FCOS 正式 run（已完成）：
  - `docs/experiments/logs/run-20260315-fcos-wandb-single-run-epoch-log.md`

## 3. 开跑前检查（必须通过）
- 进入二阶段训练前，再次确认：
  - GPU 可用与显存预算
  - 数据与标注版本一致
  - W&B 项目与 run 命名策略已固定
  - 回传策略仍使用 `--mmdet-thin`（避免大体积阻塞）

## 4. 执行顺序
1. 补齐 `docs/results/qualitative.md`（success/miss/false_positive）。（已完成）
2. 补齐 `docs/results/ablation.md`（至少一组）。
3. 完成系统默认模型选择 A/B 记录并更新配置。
4. 回填论文第四章、第五章文字，并加入 FCOS 口径限制说明。

## 5. 留痕与产物要求
- run 日志：`docs/experiments/logs/run-YYYYMMDD-fcos-*.md`
- 训练产物：
  - `experiment_assets/runs/<fcos_run_name>/`
  - `experiment_assets/runs/trace/fcos/`
- 配置快照：
  - `experiment_assets/configs/<fcos_run_name>/`（含 data yaml / 关键 cfg 片段）

## 6. 风险与应对
- 二阶段训练收益不确定：先以“当前结果+稳定性分析”支撑论文，再决定是否追加训练。
- 产物体积过大：持续使用 `--mmdet-thin` 与关键 checkpoint 归档策略。
