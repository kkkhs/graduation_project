# 实验记录：DRENet 正式续训到 300 epoch（2026-03-08）

## 1. 目标
- 把正式实验 `drenet_levirship_512_bs4_sna_20260307_formal01` 从 `epoch=100` 继续跑到总轮次 `300`
- 同时打通“本机 watch 自动回传”链路

## 2. 关键前置事实
- 当前 `runs/.../weights/last.pt` 已被 strip 成纯权重：
  - `epoch / wandb_id / opt` 已不在文件内
- 真正可续训的完整 checkpoint 位于：
  - `/root/autodl-tmp/experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_last_ep100_20260307_165755.pt`
  - 其元数据为：
    - `epoch=100`
    - `wandb_id=94d4wdmk`

## 3. 提速探测
- 目的：
  - 在不污染主 run 的前提下验证 `workers=4`、`batch=6`
- 命令：
```bash
cd /root/autodl-tmp/workspace/experiments/drenet/DRENet
export WANDB_MODE=disabled WANDB_DISABLED=true
/root/autodl-tmp/envs/shipdet/bin/python train.py \
  --weights /root/autodl-tmp/experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_last_ep100_20260307_165755.pt \
  --data /root/autodl-tmp/workspace/experiments/drenet/DRENet/data/levir_ship_autodl.yaml \
  --epochs 102 \
  --batch-size 6 \
  --img-size 512 512 \
  --workers 4 \
  --device 0 \
  --project /root/autodl-tmp/experiment_assets/runs \
  --name drenet_probe_bs6_w4_20260308 \
  --exist-ok
```
- 结果：
  - 探测成功完成
  - 峰值显存约 `3.29G`
  - 指标：
    - `P=0.423`
    - `R=0.773`
    - `mAP@0.5=0.686`
    - `mAP@0.5:0.95=0.229`

## 4. 正式续训启动
- 训练策略：
  - `img=512`
  - `workers=4`
  - `batch=8`
- 续训语义：
  - 仍写回原 run 名
  - W&B 显式恢复原 run id `94d4wdmk`
- 后台启动命令：
```bash
cd /root/autodl-tmp/workspace/experiments/drenet/DRENet
export WANDB_RESUME=allow
export WANDB_RUN_ID=94d4wdmk
nohup /root/autodl-tmp/envs/shipdet/bin/python train.py \
  --weights /root/autodl-tmp/experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_last_ep100_20260307_165755.pt \
  --data /root/autodl-tmp/workspace/experiments/drenet/DRENet/data/levir_ship_autodl.yaml \
  --epochs 300 \
  --batch-size 8 \
  --img-size 512 512 \
  --workers 4 \
  --device 0 \
  --project /root/autodl-tmp/experiment_assets/runs \
  --name drenet_levirship_512_bs4_sna_20260307_formal01 \
  --exist-ok \
  > /root/autodl-tmp/experiment_assets/runs/trace/train_drenet_formal_resume_300_bs8_w4_20260308_101502.log 2>&1 < /dev/null &
```

## 5. 当前状态
- 远端进程：
  - Python PID `3046`
- 远端日志：
  - `/root/autodl-tmp/experiment_assets/runs/trace/train_drenet_formal_resume_300_bs8_w4_20260308_101502.log`
- W&B：
  - 已恢复到 run id `94d4wdmk`
  - 页面显示为：
    - `Resuming run drenet_levirship_512_bs4_sna_20260307_formal01`
- 当前训练进度已确认进入主循环：
  - `101/299`
- 当前显存占用已确认：
  - 约 `4.26G`

## 6. 本机自动回传
- 本机已启动 watch：
```bash
SYNC_SSH_PASSWORD='<redacted>' bash scripts/sync_autodl_experiment_assets.sh \
  root@connect.westb.seetacloud.com \
  --port 29137 \
  --run-name drenet_levirship_512_bs4_sna_20260307_formal01 \
  --no-sync-checkpoints \
  --no-sync-scripts \
  --watch-pattern 'train.py.*drenet_levirship_512_bs4_sna_20260307_formal01.*--epochs 300' \
  --interval 120 \
  --local-assets-root /Users/khs/codes/graduation_project/experiment_assets
```
- 当前状态：
  - watch 已识别“Remote training still active”
  - 同名 run 和 `runs/trace/` 会在训练期间持续拉回本机

## 7. 结论
- 正式续训到 `300 epoch` 已经开始执行
- 本机自动回传链路已打通
- 当前没有发现新的路径错误、W&B 错误或 CUDA/OOM 错误

## 8. 参数上调与自动关机收尾（补充）
- 用户后续要求：
  - `batch-size` 提到 `12`
  - `workers` 提到 `8`
  - 训练完成并回传核验后自动关机云端主机
- 执行动作：
  - 停止 `batch=8, workers=4` 的正式续训进程
  - 从当前完整 checkpoint（`epoch=105`）重启为：
    - `batch=12`
    - `workers=8`
  - 新日志：
    - `/root/autodl-tmp/experiment_assets/runs/trace/train_drenet_formal_resume_300_bs12_w8_20260308_101948.log`
- 运行状态（补充时刻）：
  - 远端已推进到 `111/299`
  - 显存占用约 `7.18G`
  - 本机同步已对齐到 `109/299`（随后继续追平）

## 9. 自动关机方案落地
- 新增脚本：
  - `/Users/khs/codes/graduation_project/scripts/watch_sync_then_shutdown_autodl.sh`
- 执行语义：
  1. 本机 watch 同步直到训练进程消失
  2. 再做一次最终同步
  3. 校验本地与远端 `results.txt` 最后一行一致
  4. 一致后发送远端关机命令（默认 `shutdown -h now`）
- 当前正在运行该脚本进行收尾托管。

## 10. 最终收尾结果（已完成）
- 最终训练轮次：
  - `299/299`（已完成）
- 本地/远端对账（同一 run）：
  - 远端 `results.txt` 最后一行：`299/299`
  - 本地 `results.txt` 最后一行：`299/299`
  - `weights/last.pt`、`weights/best.pt` 已回传并可读
- 最终指标（取 `299/299`）：
  - `Precision=0.4927`
  - `Recall=0.8511`
  - `mAP@0.5=0.7949`
  - `mAP@0.5:0.95=0.2919`
- 本轮 best（按 `mAP@0.5`）：
  - `276/299`
  - `mAP@0.5=0.8017`
- 回传补充说明：
  - `runs`、`checkpoints` 已回传
  - `datasets` 按用户要求未继续同步
  - 当前本地 `checkpoints` 下与该 run 匹配的仍是 `ep100` 归档快照（300 轮阶段未新增归档文件）
- 云主机关机：
  - 已执行 `shutdown -h now`
  - SSH 连接已断开，关机完成

## 11. 是否继续训练（300 是否足够）
- 量化依据（`results.txt`）：
  - 近 50 轮最优 `mAP@0.5` 相比前 50 轮仅提升约 `+0.0054`
  - 近 20 轮最优 `mAP@0.5` 相比前 20 轮为 `-0.0032`
- 结论：
  - 对“首轮论文可回填结果”目标，`299/299` 已足够，可先收口做评测/回填。
  - 仅当你要冲更高上限时，再开 `300->1000` 的长程续训（同 run、同 wandb id、从 `last.pt` 继续）。
