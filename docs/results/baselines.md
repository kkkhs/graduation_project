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
| exp-20260308-02 | 94d4wdmk | cloud formal resume | DRENet | DRENet | 512 | 0.7949 | 0.2919 | 0.4927 | 0.8511 | 0.6241 | 121.8993 | 4.7882 | run: `drenet_levirship_512_bs4_sna_20260307_formal01`，最终 `299/299`，best AP50=0.8017@276/299；FLOPs=4.2057G（单图口径） |
| exp-20260315-01 | lvxs9xhk | cloud formal resume | FCOS | MMDetection | 1333x800（历史） | 0.7700 | 0.2850 | - | 0.4050 | - | 45.9559* | 32.1664 | 历史 run：`fcos_main_fixedcfg_20260315_160824`，`120/120` 完成，但实际训练/评测口径为 `1333x800`，不再作为三模型统一 512 主对比结论；论文口径 `global-best`: AP50=0.7980@14；系统口径 `stable-ckpt(epoch_102)`: AP50=0.7750, AP50:95=0.2880, FPS≈45.6621（`time=0.0219`反推）；FLOPs=0.123T≈123G（MMDet `get_flops.py`，实际计算输入 `800x800`） |
| exp-20260308-03 | yolo26_main_512_formal012_20260308_192322 | cloud formal + resume | YOLO26n | Ultralytics | 512 | 0.7950 | 0.3170 | 0.8430 | 0.7220 | 0.7778 | 69.0625 | 2.5042 | EarlyStopping at 263/300, best epoch=186；本地产物：`experiment_assets/runs/yolo26_main_512_formal012`；FLOPs=3.6939G |
|  |  | system-ensemble | Ensemble | Fusion(IoU+WBF) |  |  |  |  |  |  | - | - | 三模型融合输出 |

## 3.1 FCOS 说明（论文写作口径）
- FCOS 保留为 `anchor-free` 参考基线。
- 当前论文采用的 FCOS 可用结果来自历史正式 run `lvxs9xhk`，其实际输入策略为 MMDetection 默认口径 `1333x800`。
- 2026-03-16 曾尝试两条 `512x512` 新 run（`xbqeun1r`、`pequnycv`）以统一输入尺寸，但均出现数值失稳：
  - `grad_norm/loss = nan`
  - `The testing results of the whole dataset is empty`
- 因此论文正文中不将 FCOS 解释为“严格同输入尺寸公平对比”，而是解释为“参考基线与工程补充对照”。
- 若答辩中被问到输入不一致，应直接说明：
  - DRENet 与 YOLO26 采用 `512` 主线；
  - FCOS 采用框架默认训练口径，作为 `anchor-free` 参考方法，不对其与 DRENet 的细粒度差异做强结论。

## 4. 可追溯性要求（回填时必须满足）
- 每行结果必须可回溯到：
  - `docs/experiments/logs/` 对应实验记录（含 exp_id）
  - 本地日志与权重路径
  - W&B run 链接
- 若发生重试，备注中注明“重试次数 + 最终生效 run”。

## 5. 论文可用结论
1. 精度结论：DRENet 与 YOLO26 在当前实验设置下表现接近，AP50 均约为 `0.795`；其中 DRENet 召回更高，YOLO26 精度更高。
2. 召回与误检平衡结论：DRENet 更适合作为强调漏检控制的主方法，YOLO26 更适合作为轻量化部署基线；FCOS 作为 anchor-free 参考基线纳入讨论，但不用于严格同口径结论强化。
3. 系统部署建议（单模型/融合）：单模型默认优先考虑 DRENet 或 YOLO26；FCOS 作为补充参考保留。若系统强调稳妥性与结果互补，可进一步采用三模型融合输出。
