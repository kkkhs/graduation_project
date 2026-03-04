# 定性结果模板（Qualitative）

> 目标：覆盖 4 类场景，支撑“为什么这样优化”的答辩解释。

## 1. 命令来源（可追溯）
- 检测可视化命令：`docs/experiments/3060_execution_playbook.md` 第7节
- 融合命令：`tools/visualize_predict.py --mode ensemble`

## 2. 场景分类（必须覆盖）
1. 成功检测（success）
2. 误检（false_positive）
3. 漏检（miss）
4. 复杂背景（hard_scene：碎云/强浪/近岸干扰）

## 3. 图像记录表
| 实验ID | 命令来源 | 模型 | 场景类型 | 图像路径 | 现象 | 原因分析 | 改进建议 |
|---|---|---|---|---|---|---|---|
|  | 7/system-ensemble | DRENet | success | assets/figures/... |  |  |  |
|  | 7/system-ensemble | MMDet | false_positive | assets/figures/... |  |  |  |
|  | 7/system-ensemble | YOLO | miss | assets/figures/... |  |  |  |
|  | 7/system-ensemble | Ensemble | hard_scene | assets/figures/... |  |  |  |

## 4. 论文插图建议
- 图4-x：同图三模型 + 融合结果对比
- 图4-y：典型误检案例（背景纹理干扰）
- 图4-z：典型漏检案例（微小目标/遮挡）
- 图4-w：复杂背景下融合结果改进示例

## 5. 最小交付要求
- 每类场景至少 2 张图
- 每张图必须有“现象 + 原因 + 建议”
- 图名与实验编号一致，便于回溯
