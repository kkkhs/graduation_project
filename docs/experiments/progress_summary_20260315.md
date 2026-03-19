# 毕设实验阶段进度总结（历史快照，截至 2026-03-15 晚）

> 说明：本文档保留 2026-03-15 阶段状态，用于追溯当时的训练与决策背景，不代表当前最新进度。当前请优先查看 `docs/experiments/progress_summary_20260318.md`。

## 1. 当前结论（可直接对外汇报）
- DRENet 主链路已跑通并完成一轮正式结果沉淀（`299/299`）。
- YOLO26 首轮云端正式训练已完成（EarlyStopping 于 `263/300`，best epoch `186`）。
- FCOS 正式训练已完成（`120/120`），并统一了日志/看板口径：
  - W&B `coco/*` 按 `Step` 轴查看
  - 终端按 epoch 末打印
  - 固定单 run id 续写：`lvxs9xhk`（状态：`finished`）
- 训练产物已从云端回传到本地 `experiment_assets/`（采用 mmdet thin 策略），并完成一致性核验。
- 自动化链路已具备：
  - watch 同步回传
  - 训练结束 final sync
  - 远端关机
  - 本地 checkpoint 快照归档
- 主对比口径已统一：
  - 主指标：`AP50`
  - 辅指标：`AP@0.5:0.95`
  - 同时记录：`Precision/Recall/F1/FPS/Params/FLOPs`

## 1.1 2026-03-16 后续决策补充
- 已尝试两条 FCOS `512x512` 新 run，用于与 DRENet/YOLO 做更严格的同输入口径对齐。
- 两条新 run 均出现数值失稳：
  - `grad_norm/loss = nan`
  - `The testing results of the whole dataset is empty`
- 最终项目决策：
  - 不再继续投入 FCOS 重训成本
  - 论文中保留 FCOS 历史正式 run（默认输入策略）作为 `anchor-free` 参考基线
  - 主结论重点转回 DRENet 与 YOLO26
  - FCOS 定性样例已补齐（global-best 与 stable 两套口径）

## 2. 已完成工作明细

### 2.1 DRENet 实验与结果
- run 名：`drenet_levirship_512_bs4_sna_20260307_formal01`
- W&B run id：`94d4wdmk`
- 最终轮次：`299/299`
- 最终指标（last）：
  - `AP50=0.7949`
  - `AP@0.5:0.95=0.2919`
  - `Precision=0.4927`
  - `Recall=0.8511`
  - `F1=0.6241`
- 训练内 best（按 AP50）：
  - `0.8017 @ 276/299`

### 2.2 产物与回传
- 已确认回传：
  - `experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/`
  - `weights/{best.pt,last.pt}`
  - `results.txt`, `opt.yaml`, `hyp.yaml`, `runs/trace/*`
- 本地 checkpoint 已补齐阶段归档：
  - `..._last_ep299_..._manual_save.pt`
  - `..._best_ep299_..._manual_save.pt`
- 云端主机已在回传一致性核验后执行关机。

### 2.3 文档与脚本
- 已更新执行手册到当前策略：
  - MMDet：`FCOS`
  - Ultralytics：`YOLO26`
- 已统一指标说明，修正“论文 AP 口径”歧义：
  - DRENet 论文表格中的 `AP` 对应 `AP50(%)`
- 已新增/更新脚本：
  - `scripts/sync_autodl_experiment_assets.sh`
  - `scripts/watch_sync_then_shutdown_autodl.sh`
  - `scripts/snapshot_drenet_checkpoint.sh`
  - `sync_autodl_experiment_assets.sh --mmdet-thin`（仅回传 `best + last + logs/config/wandb`）

### 2.4 FCOS 正式结果（新增）
- run 名：`fcos_main_fixedcfg_20260315_160824`
- W&B run id：`lvxs9xhk`
- 最终轮次：`120/120`
- 最终指标（last）：
  - `AP50=0.770`
  - `AP@0.5:0.95=0.285`
  - `AR@100=0.405`
  - `AR@1000=0.405`
- 全局 best（按 AP50）：
  - `0.798 @ 14`
- 稳定窗口（101~120）：
  - `stable-best 指标点`：`epoch103`, `AP50=0.776`, `AP50:95=0.286`
  - 由于 `epoch103` 无保存权重，系统候选 checkpoint 选用 `epoch_102.pth`

## 3. 进行中与未完成
- D10：FCOS 正式训练与评测结果（已完成）
- D11：YOLO26 首轮已完成，若需继续需决策是否二阶段续训（如 `263 -> 400/600`）
- D12：主对比表三模型核心精度已补齐；效率项（FPS/Params/FLOPs）已补
- D13：消融实验待完成
- D14：定性图（success/miss/false_positive）已补齐（含 FCOS global-best 与 stable）
- F19：论文第4/5章待用完整三模型结果回填

## 4. 下一步执行顺序（建议严格按序）
1. 统一补齐效率与复杂度：
   - `FPS`（warmup 50 + timing 200）
   - `Params(M)`
   - `FLOPs(G)`
2. 更新结果文档：
   - `docs/results/baselines.md`
   - `docs/results/ablation.md`
   - `docs/results/qualitative.md`
3. 完成系统侧模型默认权重落地：
   - 论文：`global-best`
   - 系统：`stable-best`
   - 补固定样本集 A/B 校验记录
4. 生成下一轮计划：
   - 是否继续 YOLO26 二阶段续训（如 `263 -> 400/600`）
   - 是否继续 DRENet `300 -> 1000`
   - 是否追加 FCOS 二阶段训练（围绕 `AP50:95` 优化）

## 6. 文档归档/合并检查结果
- `docs/experiments/archive/` 下的旧执行计划与旧模型版本文档已处于归档区，暂不需要再移动。
- 当前有效入口建议只保留：
  - `docs/spec_todo.md`
  - `docs/experiments/3060_execution_playbook.md`
  - `docs/experiments/cloud_execution_playbook.md`
  - `docs/experiments/progress_summary_20260315.md`
- 本轮未做删除操作，避免影响历史追溯；后续在三模型结果补齐后可再做一次归档清理。

## 5. 关键入口文档
- 总看板：`docs/spec_todo.md`
- 云端执行手册：`docs/experiments/cloud_execution_playbook.md`
- 本地/3060 执行手册：`docs/experiments/3060_execution_playbook.md`
- 主对比表：`docs/results/baselines.md`
- 本轮 DRENet 正式日志：`docs/experiments/logs/exp-20260308-02-drenet-formal-resume-300.md`
- YOLO26 首轮实跑：`docs/experiments/logs/run-20260308-yolo26-3060-3080ti.md`
- FCOS 续训口径统一日志：`docs/experiments/logs/run-20260315-fcos-wandb-single-run-epoch-log.md`

## 7. YOLO26 当日执行补充（2026-03-08 晚）
- 3060（Windows）侧结论：
  - 已完成链路定位，但出现 `labels.cache` 锁冲突、`cv2` worker 导入失败、`WinError 1455`（页面文件），不适合本轮正式长训。
  - W&B 在 Windows 侧还出现路径名与 key 注入兼容问题。
- 3080Ti（云）侧执行：
  - 完成 1 epoch 速度对比：
    - `w0/noamp`: `81.1243s`
    - `w2/amp`: `61.0547s`
    - 提速：`1.3287x`
  - 正式训练完成（EarlyStopping）：
    - 最终停在 `263/300`，best epoch `186`
    - 末次验证：`P=0.843`，`R=0.722`，`mAP50=0.795`，`mAP50-95=0.317`
    - 远端运行目录：`/root/autodl-tmp/experiment_assets/runs/detect/runs/yolo26_main_512_formal012`
    - 本地归档目录：`/Users/khs/codes/graduation_project/experiment_assets/runs/yolo26_main_512_formal012`
    - trace 日志：`/Users/khs/codes/graduation_project/experiment_assets/runs/trace/yolo/`
- 完整留痕：
  - `docs/experiments/logs/run-20260308-yolo26-3060-3080ti.md`

## 8. FCOS 前置准备补充（2026-03-09，无卡实例）
- 已完成：
  - 在远端生成并校验 COCO 标注：
    - `annotations/train.json`（2321/2002）
    - `annotations/val.json`（788/665）
    - `annotations/test.json`（788/552）
  - 类别一致性校验通过：`ship`，`category_id=1`
- 当前阻塞：
  - 当前实例无 GPU（`nvidia-smi` 无设备，`torch.cuda.is_available()==False`）
  - 无卡实例无法进行 FCOS 冒烟/正式训练
- 已补齐（shipdet 环境）：
  - `mmengine=0.10.7`
  - `mmcv=2.1.0`（mmcv-lite）
  - `mmdet=3.3.0`
- 安装过程问题与处理：
  - `openmim` 链路首次导致 `setuptools` 降级并引发 `mmcv` 构建失败
  - 通过升级 `setuptools` + 固定 `mmcv-lite==2.1.0` 解决与 `mmdet 3.3.0` 的兼容问题
- 下一步：
  - 切回有卡实例后，直接跑 FCOS 1 epoch 冒烟
- 留痕：
  - `docs/experiments/archive/run-20260309-fcos-coco-prep-nogpu.md`
