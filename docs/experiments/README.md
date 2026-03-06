# 实验执行总规范

> 目标：三模型实验做到“命令可复制、版本可追踪、结果可回填”。

## 1. 前置条件（硬件 / 环境 / 目录）
- 支持两类训练环境：
  - Windows PowerShell（3060 6GB）
  - Linux Bash（云端 GPU）
- 默认路径：
  - Windows：项目 `E:/work/graduation_project`，数据 `E:/datasets/LEVIR-Ship`
  - Linux：项目 `/workspace/graduation_project`，数据 `/data/LEVIR-Ship`
- 本地机器：只做开发、结果汇总、论文回填。
- 训练机：只跑训练/评测/导出。

## 2. 数据准备（LEVIR-Ship + COCO）
- 固定数据版本与划分（train/val/test）。
- DRENet 训练前必须生成 degrade 图。
- MMDet/YOLO 使用统一数据口径（建议 COCO 标注统一）。

## 3. 三模型训练入口（固定 commit）
- DRENet: `a187dbe0f623b521a62c6176c7cafaa7322f5f66`
- MMDetection: `cfd5d3a985b0249de009b67d04f37263e11cdf3d`
- Ultralytics: `cac4699df04f2c06a4ef3872217eae561981349d`

## 4. 评测指标与统一口径
- 主指标：AP50
- 辅助指标：Precision / Recall / F1
- 统一定义：
  - `Precision = TP / (TP + FP)`
  - `Recall = TP / (TP + FN)`
  - `F1 = 2PR / (P + R)`

## 5. 产物目录规范
```text
artifacts/
  drenet/{weights,logs,metrics,figures}
  mmdet/{weights,logs,metrics,figures}
  yolo/{weights,logs,metrics,figures}
```

## 6. 回传流程（sync + 回填）
1. 回传：`scripts/sync_results_from_laptop.sh`
2. 回填顺序：
   1. `docs/results/baselines.md`
   2. `docs/results/qualitative.md`
   3. `docs/results/ablation.md`

## 7. 常见故障
- CUDA 不可用：检查驱动、Torch、CUDA 对应关系。
- OOM：`batch -> image_size -> amp`。
- 路径错误：优先查数据路径与权重路径。
- 中断恢复：必须从 checkpoint 续训。

## 8. 最小验收标准
- 三模型均完成冒烟训练 + 正式训练 + 评测。
- `ensemble` 推理能输出 JSON。
- 三份结果文档已可用于论文章节回填。

---

## 命令索引表（按模型）
| 模型 | 冒烟训练 | 正式训练 | 评测 | 详细命令文档 |
|---|---|---|---|---|
| DRENet | 第4.1节 | 第5.1节 | 第6.1节 | `docs/experiments/3060_execution_playbook.md` / `docs/experiments/cloud_execution_playbook.md` |
| MMDet(FCOS) | 第4.2节 | 第5.2节 | 第6.2节 | `docs/experiments/3060_execution_playbook.md` / `docs/experiments/cloud_execution_playbook.md` |
| YOLOv8n | 第4.3节 | 第5.3节 | 第6.3节 | `docs/experiments/3060_execution_playbook.md` / `docs/experiments/cloud_execution_playbook.md` |

> Windows 训练机以 `docs/experiments/3060_execution_playbook.md` 为准。  
> Linux 云 GPU 以 `docs/experiments/cloud_execution_playbook.md` 为准。
