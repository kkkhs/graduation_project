# 结果回传与自动落盘流程

## 1. 流程目标
将 3060 笔记本训练产生的日志、可视化图、结果文件稳定同步到本机，并直接用于论文与答辩材料更新。

## 2. 回传脚本
使用：`scripts/sync_results_from_laptop.sh`

建议回传目录：
- 训练日志 -> `docs/experiments/logs/`
- 结果表格 -> `docs/results/`
- 可视化样例 -> `assets/figures/`

## 3. 推荐执行顺序
1. 训练机完成一轮实验并固化产物目录。
2. 本机执行 `sync_results_from_laptop.sh`。
3. 校验文件数量和时间戳，确认同步成功。
4. 更新：
   - `docs/results/baselines.md`
   - `docs/results/ablation.md`
   - `docs/results/qualitative.md`
5. 在实验日志中记录“回传完成时间”和“回填文档清单”。

## 4. 最小验收标准
- 至少 1 份实验日志已回传。
- 至少 1 组可视化图已回传。
- `baselines.md` 至少更新 1 个模型结果行。

## 5. 失败回滚
- 若同步中断，保留本机已有文件，不覆盖。
- 重新执行脚本并只同步缺失文件。
- 对冲突文件采用“新文件名+时间戳”保留双版本。
