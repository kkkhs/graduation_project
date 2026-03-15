# 结果回传与自动落盘流程

## 1. 流程目标
把训练机或云主机产生的实验产物稳定回收到本机，并且让远端与本地目录一一对应，避免 run 名、checkpoint 名和结果文件脱节。

## 2. 当前推荐的两条同步链路

### 2.1 旧链路：项目文档/图表回传
使用：
- `scripts/sync_results_from_laptop.sh`

适用场景：
- 同步 `docs/experiments/logs/`
- 同步 `docs/results/`
- 同步 `assets/figures/`
- 同步老结构下的 `artifacts/`

### 2.2 新链路：`experiment_assets/` 精确对齐
使用：
- `scripts/sync_autodl_experiment_assets.sh`

适用场景：
- AutoDL / 云主机上的 DRENet 训练结果回传
- 要求本地与远端 `runs/checkpoints/scripts` 目录完全对应
- 需要按 `run_name` 精确回收并支持训练结束后的自动最终同步

兼容性说明：
- 若远端有 `rsync` 且本机已配置 SSH 免密，脚本优先走 `rsync`
- 若远端无 `rsync`，或当前是密码登录场景，脚本自动退回 `ssh + tar + scp`
- 密码登录时可通过环境变量 `SYNC_SSH_PASSWORD` 驱动 `ssh/scp`

## 3. 目录对应规范

### 3.1 runs 对应
- 远端：
  - `/root/autodl-tmp/experiment_assets/runs/<run_name>/`
- 本地：
  - `/Users/khs/codes/graduation_project/experiment_assets/runs/<run_name>/`

要求：
- `run_name` 必须保持完全一致
- `weights/last.pt`
- `weights/best.pt`
- `results.txt`
- `opt.yaml`
- `hyp.yaml`
都必须位于同名 run 目录内

### 3.2 checkpoints 对应
- 远端：
  - `/root/autodl-tmp/experiment_assets/checkpoints/drenet/<run_name>_*`
- 本地：
  - `/Users/khs/codes/graduation_project/experiment_assets/checkpoints/drenet/<run_name>_*`

### 3.3 scripts 对应
- 远端：
  - `/root/autodl-tmp/experiment_assets/scripts/`
- 本地：
  - `/Users/khs/codes/graduation_project/experiment_assets/scripts/`

### 3.4 datasets 对应
- 远端：
  - `/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/`
- 本地：
  - `/Users/khs/codes/graduation_project/experiment_assets/datasets/LEVIR-Ship/`

说明：
- 数据集体积大，默认不做每轮同步
- 除非远端数据版本发生变化，否则只在首次迁移时同步

## 4. 推荐执行方式

### 4.1 单次拉取
```bash
bash scripts/sync_autodl_experiment_assets.sh \
  root@connect.westc.gpuhub.com \
  --port 49353 \
  --run-name drenet_levirship_512_bs4_sna_20260307_formal01
```

效果：
- 只同步该 run
- 本地和远端 `runs/<run_name>` 保持同构
- 只同步匹配该 run 名的 checkpoint

### 4.2 训练期间轮询，训练结束后自动最终同步
```bash
SYNC_SSH_PASSWORD='<YOUR_PASSWORD>' bash scripts/sync_autodl_experiment_assets.sh \
  root@connect.westc.gpuhub.com \
  --port 49353 \
  --run-name drenet_levirship_512_bs4_sna_20260307_formal01 \
  --watch-pattern 'train.py|resume_formal01_autodl.sh' \
  --interval 120
```

效果：
- 训练进行中，每隔 `120s` 拉一次远端结果
- 当远端匹配进程消失后，再自动执行一轮最终同步
- 不要求云主机主动回连本机，因此适合 AutoDL 这类环境

### 4.3 当前 3080 Ti 正式续训示例
```bash
SYNC_SSH_PASSWORD='<YOUR_PASSWORD>' bash scripts/sync_autodl_experiment_assets.sh \
  root@connect.westb.seetacloud.com \
  --port 29137 \
  --run-name drenet_levirship_512_bs4_sna_20260307_formal01 \
  --watch-pattern 'train.py.*drenet_levirship_512_bs4_sna_20260307_formal01.*--epochs 300' \
  --interval 120 \
  --local-assets-root /Users/khs/codes/graduation_project/experiment_assets
```

### 4.4 MMDet 大 checkpoint 场景（推荐瘦身回传）
当 `runs/mmdet/<run_name>/` 下 `epoch_*.pth` 过大时，建议使用 `--mmdet-thin`：

```bash
SYNC_SSH_PASSWORD='<YOUR_PASSWORD>' bash scripts/sync_autodl_experiment_assets.sh \
  root@connect.westb.seetacloud.com \
  --port 29137 \
  --run-name mmdet/fcos_main_fixedcfg_20260315_160824 \
  --local-assets-root /Users/khs/codes/graduation_project/experiment_assets \
  --mmdet-thin \
  --no-sync-checkpoints \
  --no-sync-scripts \
  --no-sync-trace
```

该模式会：
- 回传：`best_*.pth`、`last_checkpoint` 指向的最终 `epoch_*.pth`、日志/配置/json/yaml/txt、`wandb/` 文件；
- 不回传：其余中间 `epoch_*.pth`（继续保留在云端）。

### 4.4 训练结束自动归档 checkpoint（推荐）
当使用 `watch_sync_then_shutdown_autodl.sh` 收尾时，脚本会在最终同步一致后，自动执行本地归档：

- 输入：
  - `experiment_assets/runs/<run_name>/weights/last.pt`
  - `experiment_assets/runs/<run_name>/weights/best.pt`
  - `experiment_assets/runs/<run_name>/results.txt`
- 输出：
  - `experiment_assets/checkpoints/drenet/<run_name>_last_ep<epoch>_<timestamp>.pt`
  - `experiment_assets/checkpoints/drenet/<run_name>_best_ep<epoch>_<timestamp>.pt`

可单独手动执行：
```bash
bash scripts/snapshot_drenet_checkpoint.sh \
  --run-name drenet_levirship_512_bs4_sna_20260307_formal01 \
  --assets-root /Users/khs/codes/graduation_project/experiment_assets
```

## 5. 推荐执行顺序
1. 训练前先固定 `run_name`。
2. 远端训练输出必须写入 `experiment_assets/runs/<run_name>/`。
3. 本机启动 `sync_autodl_experiment_assets.sh` 的 watch 模式。
4. 训练结束后检查：
   - 本地 `runs/<run_name>/weights/last.pt`
   - 本地 `runs/<run_name>/weights/best.pt`
   - 本地 `runs/<run_name>/results.txt`
   - 本地 `checkpoints/drenet/<run_name>_*`
5. 再回填：
   - `docs/results/baselines.md`
   - `docs/results/ablation.md`
   - `docs/results/qualitative.md`
6. 在实验日志里补：
   - 同步时间
   - 同步命令
   - 同步到的本地路径

## 6. 最小验收标准
- 本地 `experiment_assets/runs/<run_name>/` 与远端同名目录已建立映射
- 本地 `weights/last.pt` 与远端最新时间戳一致
- 本地 `results.txt` 能看到远端最新 epoch
- 至少 1 个 `checkpoint` 成功回收
- 本地 `runs/trace/` 已收到本轮正式训练日志

## 7. 失败处理
- 若 `rsync` 中断，重新执行脚本即可继续增量拉取
- 若远端无 `rsync`，脚本会自动改走 `ssh + tar + scp`，不需要人工切回 `scp -r`
- 若是密码登录且未设置 `SYNC_SSH_PASSWORD`，脚本会退回交互式 SSH/`scp`
- 若 `watch-pattern` 与真实训练进程不匹配，脚本会提前结束；此时改正 pattern 后重新执行
- 若训练已结束但本地结果不完整，先执行一次不带 `--watch-pattern` 的单次同步
