status: executed
evidence: commands + outputs + artifacts

# 运行日志：FCOS 512 统一口径干净 Run 启动（2026-03-16）

## 1. 目标
- 启动新的 FCOS 正式 run：`fcos_main_512_clean_20260316`
- 统一输入口径到 `512x512`
- W&B 使用新的干净 run
- 保证每个 epoch 都有验证点与主图指标

## 2. 启动前事实
- 远端主机：`root@connect.westb.seetacloud.com:29137`
- 容器：`autodl-container-a2fd40a1b8-db880c0e`
- GPU：`NVIDIA GeForce RTX 3080 Ti 12GB`
- 数据目录存在：
  - `/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/annotations`
- MMDetection 工作目录存在：
  - `/root/autodl-tmp/workspace/mmdetection`
- 旧 FCOS run 存在：
  - `/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824`

## 3. 本轮配置摘要
- 配置快照：
  - `experiment_assets/configs/fcos_main_512_clean_20260316/fcos_main_512_clean_20260316.py`
- 关键差异：
  - `Resize -> 512x512`
  - `val_interval -> 1`
  - `batch_size -> 16`
  - `checkpoint.interval -> 5`
  - `max_keep_ckpts -> 3`
  - 新增 `WandbEpochMirrorHook`

## 4. 启动命令
```bash
cd /root/autodl-tmp/workspace/mmdetection
export PYTHONPATH=/root/autodl-tmp/workspace/mmdetection/tools:${PYTHONPATH}
export WANDB_CONSOLE=wrap
conda run -n fcos310 python tools/train.py \
  /root/autodl-tmp/experiment_assets/configs/fcos_main_512_clean_20260316/fcos_main_512_clean_20260316.py \
  --work-dir /root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_512_clean_20260316
```

## 5. 待补充的运行事实
- 启动时间：`2026-03-16 11:27`（远端容器时间）
- 后台命令：
```bash
cd /root/autodl-tmp/workspace/mmdetection
export PYTHONPATH=/root/autodl-tmp/workspace/mmdetection/tools:${PYTHONPATH}
export WANDB_CONSOLE=wrap
nohup bash -lc 'cd /root/autodl-tmp/workspace/mmdetection && export PYTHONPATH=/root/autodl-tmp/workspace/mmdetection/tools:${PYTHONPATH} && export WANDB_CONSOLE=wrap && conda run -n fcos310 python tools/train.py /root/autodl-tmp/experiment_assets/configs/fcos_main_512_clean_20260316/fcos_main_512_clean_20260316.py --work-dir /root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_512_clean_20260316' > /root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_512_clean_20260316.log 2>&1 &
```
- 启动后主进程：
  - `PID 1430`：`python tools/train.py ...fcos_main_512_clean_20260316.py`
- W&B run id：`xbqeun1r`
- W&B run url：`https://wandb.ai/2964745405-hfut/fcos/runs/xbqeun1r`
- W&B 本地目录：
  - `/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_512_clean_20260316/20260316_112725/vis_data/wandb/run-20260316_112728-xbqeun1r`
- W&B console 文件：
  - `.../files/output.log`
- 首批验证点：
  - `Epoch(val) [1]`: `AP50=0.0170`, `AP50:95=0.0050`, `time=0.0225`
  - `Epoch(val) [2]`: `AP50=0.2910`, `AP50:95=0.0680`, `AR@100=0.211`, `time=0.0218`
  - `Epoch(val) [3]`: `AP50=0.4120`, `AP50:95=0.0920`, `AR@100=0.183`, `time=0.0211`
- 关键产物目录：
  - `/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_512_clean_20260316`
- 首个 best 权重更新：
  - `best_coco_bbox_mAP_50_epoch_1.pth`
  - 随后已更新到 `best_coco_bbox_mAP_50_epoch_3.pth`

## 6. 当前结论（启动阶段）
- 新 run 已成功与旧 `lvxs9xhk` 分离，W&B 使用独立新 run。
- 新配置实际生效：
  - `train Resize=(512,512)`
  - `val Resize=(512,512)`
  - `batch_size=16`
  - `val_interval=1`
- 终端日志按 epoch 末打印，未出现每 iter 刷屏。
- `WANDB_CONSOLE=wrap` 已生效，W&B 本地 `output.log` 正在持续写入。

## 7. 巡检快照（2026-03-16 11:32）
- 远端时间：
  - `2026-03-16 11:32:34`
- 主训练进度：
  - 最新 `Epoch(train) [8][75/75]`
  - 最新完整验证：`Epoch(val) [7][788/788]`
  - `Epoch(val) [8]` 已开始执行
- 最近验证指标：
  - `epoch 4`: `AP50=0.5250`, `AP50:95=0.1480`
  - `epoch 5`: `AP50=0.6190`, `AP50:95=0.1680`（当前最佳）
  - `epoch 6`: `AP50=0.0220`, `AP50:95=0.0030`
  - `epoch 7`: `AP50=0.2520`, `AP50:95=0.0460`
- 资源占用：
  - GPU 显存：约 `4741 MiB`
  - GPU 利用率：巡检瞬间 `0%`（当时处于评测阶段，不代表训练段利用率）
  - 功耗：约 `111.22 W`
  - 训练日志中的显存：约 `3949 MiB`
- 效率估计：
  - 当前每个 epoch 约 `75` 个 train iter
  - 最近 train iter 时间约 `0.066~0.081 s/iter`
  - 按当前节奏，完整 120 epoch 预计约 `70~80` 分钟量级
- 进程说明：
  - `PID 1430` 为主训练进程
  - 其余同命令 `python tools/train.py ...` 为 dataloader worker / 子进程，不是重复开了多份正式训练

## 8. 最终处置
- 后续继续训练中，run 在 `epoch 19` 起出现：
  - `grad_norm: nan`
  - `loss: nan`
  - `The testing results of the whole dataset is empty`
- 结论：
  - 该 `512x512` clean run 数值失稳，后段结果不可用。
  - 本 run 保留为失败留痕，不再作为论文主对比结论来源。
- 对应 W&B run：
  - `xbqeun1r`
  - note 已补充“数值失稳，仅保留前期留痕”
