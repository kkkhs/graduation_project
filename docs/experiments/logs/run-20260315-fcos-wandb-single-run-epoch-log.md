状态: 已执行
evidence: commands + outputs + artifacts

# 运行日志：FCOS 单 Run 续写与日志口径统一（2026-03-15）

## 1. 目标
- 固定 W&B 使用同一 run id：`lvxs9xhk`（不再产生同名碎片 run）。
- 终端日志改为每个 epoch 打印一次（不再每 step 刷屏）。
- 训练语义不变：`max_epochs=120`、`batch_size=10`、`num_workers=10`、`val_interval=2`、`save_best=coco/bbox_mAP_50`。
- 从当前 checkpoint 续训，不从头开跑。

## 2. 执行环境
- 远端：`ssh -p 29137 root@connect.westb.seetacloud.com`
- 工作目录：`/root/autodl-tmp/workspace/mmdetection`
- 训练目录：`/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824`
- W&B：`project=fcos`，`entity=2964745405-hfut`

## 3. 执行过程（命令与结果）

### 3.1 先确认无重复主训练进程
- 命令：
  - `pgrep -af "tools/train.py configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_levir_ship.py" || true`
- 结果：
  - 无输出（当时无该主进程在跑）。

### 3.2 确认续训锚点
- 命令：
  - `cat /root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/last_checkpoint`
- 输出：
  - `/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/epoch_8.pth`

### 3.3 生成“同 run 续写 + epoch 级终端打印”临时配置
- 生成文件：
  - `/tmp/fcos_main_resume_lvxs9xhk.py`
- 关键修改：
  - `default_hooks.logger.interval=1000`
  - `default_hooks.logger.ignore_last=False`
  - `resume=True`
  - `load_from=.../epoch_8.pth`
  - `WandbVisBackend.init_kwargs` 加入：
    - `id='lvxs9xhk'`
    - `resume='must'`
    - `allow_val_change=True`
    - `project='fcos'`
    - `entity='2964745405-hfut'`

### 3.4 重启续训（写回同一 work_dir）
- 命令：
  - `nohup python tools/train.py /tmp/fcos_main_resume_lvxs9xhk.py --work-dir /root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824 > /root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824.log 2>&1 &`
- 关键输出：
  - 日志中出现：`resumed epoch: 8, iter: 952`

### 3.5 W&B 单 run 连续性核验
- 命令：
  - `python - <<'PY' ... api.runs('2964745405-hfut/fcos') ... PY`
- 关键结果：
  - `lvxs9xhk ... running https://wandb.ai/2964745405-hfut/fcos/runs/lvxs9xhk`
  - 未出现新的 FCOS running run id。

### 3.6 终端打印频率核验（按 epoch）
- 命令：
  - `grep -n "Epoch(train)" ...fcos_main_fixedcfg_20260315_160824.log | tail -n 20`
- 关键结果：
  - `Epoch(train) [9][119/119] ...`
  - `Epoch(train) [10][119/119] ...`
  - `Epoch(train) [11][119/119] ...`
- 结论：
  - 终端已变为每个 epoch 末打印一次，不再每 iter 打印。

### 3.7 稳定性核验（无 NaN/空结果）
- 命令：
  - `grep -nE "loss: nan|grad_norm: nan|results of the whole dataset is empty" ...log || true`
  - `grep -n "coco/bbox_mAP" ...log | tail -n 20`
- 关键结果：
  - 未检出 `nan` 或 `results empty`。
  - 观测到验证点：
    - `Epoch(val) [10][788/788] coco/bbox_mAP_50: 0.7700`
    - `Epoch(val) [12][788/788] coco/bbox_mAP_50: 0.7790`
    - `Epoch(val) [14][788/788] coco/bbox_mAP_50: 0.7980`
  - 训练持续推进到：`Epoch(train) [15][119/119]`
  - 资源观察：`GPU 100%`，显存约 `9439/12288 MiB`

## 4. 遇到的问题与修复
- 问题：
  - 首次尝试 `resume='must'` 时，W&B 报“resumed run 不允许配置变更（default_hooks）”。
- 修复：
  - 在 `init_kwargs` 增加 `allow_val_change=True` 后恢复正常。

## 5. 过程口径声明（本轮起生效）
- W&B `coco/*` 图表默认按 `Step` 轴查看（不强制切 `epoch` 轴）。
- 终端日志默认按 `epoch` 末打印。
- FCOS 正式续训统一固定 run id：`lvxs9xhk`。

## 6. 当前状态（记录时）
- 训练仍在继续（`max_epochs=120` 目标不变）。
- 产物持续写入：
  - `/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/`

## 7. 巡检快照（2026-03-15 16:30~16:32）
- 进程结构：
  - 单主进程：`PID 6620`
  - 其余同命令 PID 均为该主进程子进程（dataloader workers）
- 训练推进：
  - `last_checkpoint -> epoch_16.pth`
  - 日志已推进到：`Epoch(train) [19][119/119]`
  - 最近验证：
    - `Epoch(val) [16] mAP50=0.604`
    - `Epoch(val) [18] mAP50=0.707`
  - 当前 best 仍为：`mAP50=0.798 @ epoch14`
- 资源：
  - GPU 利用率约 `85%~92%`
  - 显存约 `9439/12288 MiB`
  - 功耗约 `277W`
- 存储：
  - `/root/autodl-tmp` 可用约 `35GB`
- W&B：
  - run id: `lvxs9xhk`
  - URL: `https://wandb.ai/2964745405-hfut/fcos/runs/lvxs9xhk`
  - state: `running`

## 8. 口径升级执行（2026-03-15 晚）
- 执行目标：
  - 主图切换为按 `progress/epoch` 展示；
  - 每个 epoch 验证一次（`val_interval=1`）；
  - 保留 `coco/*` 同时新增 `metrics/*`；
  - 补齐 `metrics/*` 全历史真实点（不插值）。

- 关键动作：
  - 训练配置切换到：
    - `train_cfg = dict(max_epochs=120, ..., val_interval=1)`
    - `WANDB_CONSOLE=wrap`
    - 同 run id 续写：`lvxs9xhk`
  - 新增并启用 `WandbEpochMirrorHook`（`tools/wandb_epoch_mirror_hook.py`）：
    - `metrics/mAP50 <- coco/bbox_mAP_50`
    - `metrics/mAP50-95 <- coco/bbox_mAP`
    - `metrics/recall <- AR(all,maxDets=100)`
    - `metrics/recall@1000 <- AR(all,maxDets=1000)`
  - 用本地训练日志解析历史验证结果并回放到同一 run：
    - 回放区间：`epoch 10 -> 60`
    - 回放条数：`27`（真实验证点）

- 关键验收结果：
  - 续训进程：`fcos_main_resume_epoch_axis_val1.py` 运行中；
  - 已出现每个 epoch 的验证记录（如 `Epoch(val) [59]`）；
  - W&B 新键可见，最新示例：
    - `progress/epoch=59`
    - `metrics/mAP50=0.774`
    - `metrics/mAP50-95=0.293`
    - `metrics/recall=0.411`
    - `metrics/recall@1000=0.411`
  - `run_state=running`，`lastStep` 持续增长。

- 说明：
  - 由于中途修正后从 `epoch_58` 重新起训，`epoch=59` 在 `metrics/*` 中存在重复点（同 epoch 两次记录）；后续 epoch 按当前链路正常连续写入。

## 9. 正式训练收口结果（2026-03-15 18:29）
- 收口状态：
  - 已完成 `epoch 120/120`，并完成 `Epoch(val) [120]`。
  - W&B run：`lvxs9xhk` 状态为 `finished`，`summary_epoch=120`。
- 最终指标（同一 run，按日志与 W&B summary）：
  - `last@epoch120`: `AP50=0.770`，`AP50:95=0.285`，`AR@100=0.405`，`AR@1000=0.405`
  - `best(AP50)`: `0.798 @ epoch14`（`best_coco_bbox_mAP_50_epoch_14.pth`）
- 末段平台特征（近20个 epoch）：
  - `mAP50` 均值约 `0.774`，波动范围约 `0.018`
  - `mAP50-95` 均值约 `0.287`，波动范围约 `0.008`
  - `recall` 均值约 `0.405`，波动范围约 `0.005`

## 10. 本轮关键问题与处理
- 问题1：W&B 配置变更报错（`resume must` 下不允许默认配置差异）
  - 处理：`init_kwargs` 增加 `allow_val_change=True`，保持同 run 续写。
- 问题2：`metrics/recall@1000` 初次映射错误（误取到 `area=large`）
  - 处理：修正正则为 `area=all + maxDets=1000`，并重放历史真实点。
- 问题3：训练完成后进程未及时退出（GPU 显存被占用、利用率0）
  - 现象：`120/120` 与 W&B `finished` 后，`tools/train.py` 子进程仍驻留。
  - 处理：确认收尾上传完成后，执行 `pkill -f 'tools/train.py /tmp/fcos_main_resume_epoch_axis_val1.py'` 释放资源。

## 11. 产物与结论
- 远端主产物目录：
  - `/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824`
- 本轮可用于论文的核心结论：
  - FCOS 首轮正式训练完成（120 epoch），可回填主对比表（AP50/AP50:95/AR）。
  - `best` 出现在早期（epoch14），系统接入建议按“`global-best`（论文）+ `stable-best`（系统默认）”双口径执行。

## 12. 回传与关机收尾（2026-03-15 晚）
- 用户变更要求：
  - 不再立即全量拉取超大 checkpoint；
  - 先关机，后续以无卡模式再做 `runs/mmdet` 全量回传或加速拉取。
- 已执行动作：
  - 启动过的回传命令（后中断）：
    - `scp -P 29137 -r root@connect.westb.seetacloud.com:/root/autodl-tmp/experiment_assets/runs/mmdet /Users/khs/codes/graduation_project/experiment_assets/runs/`
  - 停止回传命令：
    - `kill -INT <scp_pid>`（本次为 `kill -INT 49961`）
  - 执行远端关机命令：
    - `shutdown -h now`
  - SSH 会话被远端主动关闭，表明关机命令已生效。
- 当前本机回传状态（部分）：
  - 已落地目录：`/Users/khs/codes/graduation_project/experiment_assets/runs/mmdet`
  - 当前体积约 `921M`（非完整全量）。
- 待续动作（下一次无卡模式）：
  - 仅继续拉取 `runs/mmdet`；
  - 采用断点续传或“先日志后权重”的策略，减少等待时间与费用。

## 13. Stable 参数计算与权重补拉（无卡模式）
- 目标：
  - 在不全量回传所有 `epoch_*.pth` 的前提下，补齐系统侧 `stable-best` 候选。
- 计算规则：
  - 稳定窗口：最后 20 个 epoch（`101~120`）
  - 监控指标：`coco/bbox_mAP_50`
  - 先选窗口内 `AP50` 最优，再用 `AP50:95` / 更晚 epoch 作为并列打破。
- 计算结果（基于完整主日志）：
  - `global-best`: `epoch14`, `AP50=0.798`, `AP50:95=0.289`
  - `stable-best(指标点)`: `epoch103`, `AP50=0.776`, `AP50:95=0.286`
  - 稳定窗口统计：
    - `AP50 mean/std = 0.77095 / 0.00183`
    - `AP50:95 mean/std = 0.28525 / 0.00114`
- 文件现实约束：
  - 该 run 非每轮保存 checkpoint，`epoch103` 无对应 `.pth` 文件；
  - 邻近可用 checkpoint：`epoch102.pth`、`epoch104.pth`。
- 最终选择与回传：
  - 选择 `epoch102.pth` 作为 `stable` 候选 checkpoint（窗口内可用权重里 AP50 更优）。
  - 已拉回本地：
    - `.../fcos_main_fixedcfg_20260315_160824/epoch_102.pth`
  - 计算产物已落盘：
    - `.../fcos_main_fixedcfg_20260315_160824/stable_selection.json`

## 14. MMDet 精简回传核验（无卡模式）
- 同步策略：
  - 仅拉 `runs/mmdet`，并启用 `--mmdet-thin`（只回传 `best + last + 日志/配置/wandb`）。
- 本次新增回传：
  - 主日志：`fcos_main_fixedcfg_20260315_160824.log`
  - 根目录配置：`fcos_main_fixedcfg_*.py`、`fcos_main_resume_*.py`
  - 关键权重：`best_coco_bbox_mAP_50_epoch_14.pth`、`epoch_120.pth`、`epoch_102.pth`
  - 口径文件：`stable_selection.json`
- 核验结论：
  - 本地 `runs/mmdet/fcos_main_fixedcfg_20260315_160824` 已具备论文与系统接入所需最小产物；
  - 其余中间 `epoch_*.pth` 保留在云端，不再回传。

## 15. 关键 checkpoint 归档（本机）
- 已在 `experiment_assets/checkpoints/` 建立 YOLO 与 MMDet 关键 checkpoint 归档：
  - `checkpoints/yolo/yolo26_main_512_formal012_{best,last}.pt`
  - `checkpoints/mmdet/fcos_main_fixedcfg_20260315_160824_{global_best_ep14,stable_ep102,last_ep120}.pth`
- 索引文件：
  - `experiment_assets/checkpoints/checkpoint_registry_20260315.md`
