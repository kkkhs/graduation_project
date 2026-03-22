## 结果文档说明

本目录用于论文第四章与答辩结果材料，统一分为三类：
- `baselines.md`：三模型主对比 + 融合结果（ensemble）。
- `ablation.md`：已完成的两组离线消融结果：
  - 三模型统一阈值消融
  - `FCOS / YOLO26` 输入尺寸敏感性分析（含 DRENet 尺寸限制说明）
- `qualitative.md`：四类场景定性分析（成功/误检/漏检/复杂背景）。

当前使用口径约定：
- `baselines.md` 中 FCOS 主表结果对应论文主对比口径，用于说明 anchor-free 参考基线的正式结果。
- `ablation.md` 中 FCOS 统一采用系统与消融口径对应的稳定候选 checkpoint，用于参数扰动分析。
- `qualitative.md` 同时保留两套 FCOS 原始样例来源，但论文第四章当前优先使用 `thesis_overleaf/figures/generated/` 下的成稿拼板图。

回填顺序固定：
1. `baselines.md`
2. `qualitative.md`
3. `ablation.md`
