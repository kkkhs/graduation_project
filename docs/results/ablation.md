# 消融实验（最小 1-2 组）

## 1. 命令来源（可追溯）
- 基础训练命令来源：`docs/experiments/3060_execution_playbook.md` 第5节
- 评测命令来源：`docs/experiments/3060_execution_playbook.md` 第6节
- 融合推理命令来源：`tools/run_predict.py` / `tools/visualize_predict.py`

## 2. 阈值消融（必须）
| 实验ID | 命令来源 | 模型 | conf | iou | AP50 | Precision | Recall | F1 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
|  | system-ensemble | Ensemble | 0.25 | 0.50 |  |  |  |  |  |
|  | system-ensemble | Ensemble | 0.35 | 0.50 |  |  |  |  |  |
|  | system-ensemble | Ensemble | 0.25 | 0.60 |  |  |  |  |  |

## 3. 分辨率消融（可选）
| 实验ID | 命令来源 | 模型 | 输入尺寸 | AP50 | Precision | Recall | F1 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
|  | 5.3/6.3 | YOLO26n | 512 |  |  |  |  |  |
|  | 5.3/6.3 | YOLO26n | 640 |  |  |  |  |  |
|  | 5.3/6.3 | YOLO26n | 768 |  |  |  |  |  |

## 4. 总结
1. 最优阈值组合：`<填写>`
2. 分辨率与精度/速度权衡：`<填写>`
