状态: 计划中
执行: 可启动

# 计划：FCOS 512 统一口径干净 Run（2026-03-16）

## 1. 背景
- 旧 FCOS 正式 run `fcos_main_fixedcfg_20260315_160824` 已完成，但实际输入口径为 `1333x800`。
- 该结果不再作为三模型主对比的统一输入基线，只保留为历史记录。
- 现阶段需要一条新的干净 run：
  - 输入统一为 `512x512`
  - W&B 单独新开 run
  - 每个 epoch 都有验证点
  - 主图按 `progress/epoch` 展示

## 2. 本轮目标
1. 启动 `fcos_main_512_clean_20260316` 正式训练。
2. 保证 W&B 可直接观察每个 epoch 的指标变化。
3. 将本轮作为论文主对比候选 FCOS 基线。

## 3. 配置要点
- 模型：FCOS R50-FPN
- 框架：MMDetection
- 数据：LEVIR-Ship COCO 标注
- 输入：`512x512`
- `train/val/test` 全部 `Resize(scale=(512,512), keep_ratio=False)`
- `batch_size=16`
- `num_workers=10`
- `max_epochs=120`
- `val_interval=1`
- `save_best='coco/bbox_mAP_50'`
- `checkpoint.interval=5`
- `max_keep_ckpts=3`
- AMP：开启

## 4. W&B 口径要求
- project：`fcos`
- run name：`fcos_main_512_clean_20260316`
- 新图必须包含：
  - `metrics/mAP50`
  - `metrics/mAP50-95`
  - `metrics/recall`
  - `metrics/recall@1000`
- X 轴统一：
  - `progress/epoch`

## 5. 留痕要求
- 训练配置快照：
  - `experiment_assets/configs/fcos_main_512_clean_20260316/`
- 运行日志：
  - `docs/experiments/logs/run-20260316-fcos-512-clean-start.md`
- 远端产物目录：
  - `/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_512_clean_20260316`

## 6. 验收标准
1. 训练可正常启动，无路径错误、数据错误、CUDA 错误。
2. W&B 新建干净 run，不复用旧 `lvxs9xhk`。
3. 第 1 个 epoch 完成后，W&B 能看到：
   - `progress/epoch=1`
   - `metrics/mAP50`
   - `metrics/mAP50-95`
4. 终端日志按 epoch 末打印，不按 iter 刷屏。
