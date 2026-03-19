# 毕设实验阶段进度总结（更新至 2026-03-18）

## 1. 当前结论
- 三模型正式结果已经齐全，论文主对比可直接使用：
  - `DRENet`：专项方法主线，召回优势明显
  - `FCOS`：anchor-free 参考基线，保留历史默认输入口径结果
  - `YOLO26`：轻量部署与系统默认候选
- 两组离线消融已经完成：
  - 三模型统一阈值消融
  - `FCOS / YOLO26` 输入尺寸敏感性分析
- `DRENet` 非 `512` 推理尺寸在当前本地插件路径下会触发 shape mismatch，已作为真实实验边界记录，不再强行纳入统一尺寸主表。
- 三模型定量、定性、效率和关键权重已经形成可追溯闭环，当前阶段重点转向论文收口与系统默认权重策略。

## 2. 已完成工作

### 2.1 三模型主对比
- DRENet 正式结果已固定并回填至：
  - `docs/results/baselines.md`
  - `docs/results/qualitative.md`
- FCOS 正式结果已固定为历史默认输入口径 run：
  - run id：`lvxs9xhk`
  - 论文中作为 anchor-free 参考基线
  - 系统侧额外保留 `stable-best` 候选
- YOLO26 正式结果已固定：
  - EarlyStopping 于 `263/300`
  - 当前作为系统默认模型的最强候选

### 2.2 离线消融
- 阈值消融已完成：
  - 模型：`DRENet / FCOS / YOLO26`
  - 维度：`conf=0.15 / 0.25 / 0.35`
  - 结果文件：`docs/results/ablation.md`
  - 日志：`docs/experiments/logs/run-20260318-阈值消融正式执行.md`
- 输入尺寸敏感性分析已完成：
  - 模型：`FCOS / YOLO26`
  - 维度：`imgsz=512 / 640 / 800`
  - DRENet 尺寸限制已留痕
  - 日志：`docs/experiments/logs/run-20260318-尺寸敏感性正式执行.md`

### 2.3 推理与脚本侧改造
- 统一推理链路已支持 `override_imgsz`，用于离线消融：
  - `YOLOAdapter`：支持覆写推理尺寸
  - `MMDetAdapter`：支持测试 resize 覆写
  - `DRENet`：接口已通，但非 `512` 在模型内部失败
- 已新增离线消融工具：
  - `tools/eval_ablation.py`
  - `scripts/run_ablation_matrix.sh`

### 2.4 论文与结果材料
- 结果主表：`docs/results/baselines.md`
- 消融表：`docs/results/ablation.md`
- 定性页：`docs/results/qualitative.md`
- FCOS 定性图已同时保留：
  - `global-best` 论文口径
  - `stable-best` 系统口径

## 3. 当前明确结论
1. DRENet 与 YOLO26 是论文主结论模型。
2. FCOS 保留为 anchor-free 参考基线，不承担“严格同输入公平对比”的主论断。
3. 阈值消融能够支撑“Precision / Recall 取舍差异”的论文分析。
4. 尺寸敏感性分析能够支撑“不同检测范式对输入尺度扰动的响应不同”的论文分析。
5. DRENet 的尺寸限制属于真实工程边界，反而有助于论文解释“系统集成与统一评测存在实现成本”。

## 4. 当前仍待完成
- 论文第四章、第五章的最终文字回填与统一润色
- 摘要、结论、补充说明中与“消融未完成”相关的旧表述清理
- 系统默认权重策略正式落地：
  - 论文：`global-best`
  - 系统：`stable-best`
- 演示样例整理：
  - 易/中/难样例
  - success / miss / false_positive 对照

## 5. 建议的下一步顺序
1. 先完成论文文本收口，尤其是第四章、第五章、摘要和结论。
2. 固化 `configs/models.yaml` 与相关文档中的默认权重策略。
3. 整理答辩材料所需的主表、消融表和定性图清单。
4. 若还有时间，再补融合模式统一离线评测或系统默认参数说明。

## 6. 关键入口
- 看板：`docs/spec_todo.md`
- 实验索引：`docs/experiments/README.md`
- 主对比：`docs/results/baselines.md`
- 消融：`docs/results/ablation.md`
- 定性：`docs/results/qualitative.md`
