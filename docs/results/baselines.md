# 主对比结果（Baselines + Ensemble）

## 1. 统一评测设置
- 数据集：LEVIR-Ship（固定划分）
- 指标：AP50（主）/ AP@0.5:0.95（辅）/ Precision / Recall / F1
- 测试脚本版本：`<填写>`
- 推理阈值：`conf=<填写>, iou=<填写>`
- 实验命名规范：`drenet_{dataset}_{imgsz}_{bs}_{seed}_{date}`
- 效率与复杂度测量协议：
  - FPS：`imgsz=512`、`batch=1`、同一 GPU、同一精度模式；warmup 50 次后计时 200 次，取平均
  - Params：统计总参数量，单位 `M`
  - FLOPs：输入 `512x512`，单位 `GFLOPs`（可写入实验日志备注）

## 2. 命令来源（可追溯）
- DRENet：`docs/experiments/3060_execution_playbook.md` 第4.1 / 5.1 / 6.1节
- MMDet(FCOS)：`docs/experiments/3060_execution_playbook.md` 第4.2 / 5.2 / 6.2节
- YOLO26n：`docs/experiments/3060_execution_playbook.md` 第4.3 / 5.3 / 6.3节
- 融合推理：`tools/run_predict.py --mode ensemble`

## 3. 三模型与融合结果
| 实验ID | W&B Run | 命令编号 | 模型 | 框架 | 输入尺寸 | AP50(主) | AP@0.5:0.95(辅) | Precision | Recall | F1 | FPS | Params(M) | 备注 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| exp-20260308-02 | 94d4wdmk | cloud formal resume | DRENet | DRENet | 512 | 0.7949 | 0.2919 | 0.4927 | 0.8511 | 0.6241 | 待测 | 待补 | run: `drenet_levirship_512_bs4_sna_20260307_formal01`，最终 `299/299`，best AP50=0.8017@276/299 |
|  |  | 4.2/5.2/6.2 | FCOS | MMDetection |  |  |  |  |  |  |  |  |  |
| exp-20260308-03 | yolo26_main_512_formal012_20260308_192322 | cloud formal + resume | YOLO26n | Ultralytics | 512 | 0.7950 | 0.3170 | 0.8430 | 0.7220 | 0.7778 | 待测 | 待补 | EarlyStopping at 263/300, best epoch=186；本地产物：`experiment_assets/runs/yolo26_main_512_formal012` |
|  |  | system-ensemble | Ensemble | Fusion(IoU+WBF) |  |  |  |  |  |  | - | - | 三模型融合输出 |

## 4. 可追溯性要求（回填时必须满足）
- 每行结果必须可回溯到：
  - `docs/experiments/logs/` 对应实验记录（含 exp_id）
  - 本地日志与权重路径
  - W&B run 链接
- 若发生重试，备注中注明“重试次数 + 最终生效 run”。

## 5. 论文可用结论
1. 精度结论：`<填写>`
2. 召回与误检平衡结论：`<填写>`
3. 系统部署建议（单模型/融合）：`<填写>`
