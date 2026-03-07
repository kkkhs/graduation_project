# DRENet 跨主机/云服务器迁移与续跑手册（Docker）

## 1. 目标与原则
- 目标：在 4090 或云服务器上直接拉起 DRENet 训练，并支持 checkpoint 续跑。
- 核心原则：代码版本固定、镜像版本固定、数据外置挂载、全程留痕、wandb 可追溯。
- 镜像命名规范：`drenet-train:<git_sha>-cu124`

## 2. 前置条件
- Linux 主机已安装：
  - NVIDIA 驱动
  - Docker
  - `nvidia-container-toolkit`
- 已准备数据目录（主机侧）：
  - `<DATASET_ROOT>/train/images|labels|degrade`
  - `<DATASET_ROOT>/val/images|labels|degrade`
  - `<DATASET_ROOT>/test/images|labels|degrade`
- 已准备 wandb 凭证：
  - `export WANDB_API_KEY=<your_key>` 或已 `wandb login`

## 3. 目录与接口标准
- 宿主机目录：
  - 项目：`<PROJECT_ROOT>`
  - 数据：`<DATASET_ROOT>`
  - 输出：`<RUNS_ROOT>`
  - 归档：`<CHECKPOINT_ROOT>`
- 容器内固定映射：
  - 代码：`/workspace/project`
  - 数据：`/workspace/datasets/LEVIR-Ship`
  - 输出：`/workspace/runs`
  - 归档：`/workspace/checkpoints`
- 实验命名：`drenet_{dataset}_{imgsz}_{bs}_{seed}_{date}`

## 4. 快速开始（首次）
```bash
cd <PROJECT_ROOT>
export WANDB_API_KEY=<your_key>

bash deploy/docker/run_formal.sh \
  --mode fresh \
  --dataset-root <DATASET_ROOT> \
  --runs-root <RUNS_ROOT> \
  --checkpoints-root <CHECKPOINT_ROOT> \
  --epochs 300 \
  --batch 4 \
  --workers 4 \
  --imgsz 512 \
  --seed 42 \
  --exp-name drenet_levirship_512_bs4_s42_20260307
```

## 5. 续跑策略（3060 -> 4090）

### 5.1 严格可比续跑（默认推荐）
- 目标：同 commit、同数据、同超参，从 `last.pt` 延续 epoch。
- 命令：
```bash
bash deploy/docker/run_formal.sh \
  --mode strict-resume \
  --dataset-root <DATASET_ROOT> \
  --runs-root <RUNS_ROOT> \
  --checkpoints-root <CHECKPOINT_ROOT> \
  --resume-ckpt <PATH_TO_LAST_PT> \
  --exp-name <原exp_name或续跑exp_name>
```
- 说明：建议用于“Docker 流程产出的 checkpoint -> Docker 流程续跑”。

### 5.2 提速重开（可选）
- 目标：从同 checkpoint 启动新 run，可按 4090 资源上调 batch/workers。
- 命令：
```bash
bash deploy/docker/run_formal.sh \
  --mode speed-restart \
  --dataset-root <DATASET_ROOT> \
  --runs-root <RUNS_ROOT> \
  --checkpoints-root <CHECKPOINT_ROOT> \
  --resume-ckpt <PATH_TO_BEST_OR_LAST_PT> \
  --epochs 200 \
  --batch 8 \
  --workers 8 \
  --imgsz 512 \
  --seed 42 \
  --exp-name drenet_levirship_512_bs8_s42_20260308 \
  --extra-tags resumed_from:<old_run_id>
```

## 6. wandb 规范
- 固定 `project=graduation-drenet`
- `run.name=<exp_name>`
- tags 最少包含：`dataset,imgsz,bs,seed,stage=formal,host`
- 同一 run 强制续写（仅在你明确需要时）：
```bash
export WANDB_RUN_ID=<existing_run_id>
export WANDB_RESUME_MODE=must
```

## 7. 留痕与产物
- 命令日志：`<RUNS_ROOT>/trace/train_<exp_name>_<timestamp>.log`
- 训练输出：`<RUNS_ROOT>/<exp_name>/...`
- 归档权重：`<CHECKPOINT_ROOT>/drenet/<exp_name>_{best,last}.pt`
- 文档回填：
  - `docs/experiments/logs/exp-YYYYMMDD-xx-*.md`
  - `docs/results/baselines.md`（实验ID + W&B Run）

## 8. 验证清单（迁移后必须过）
- 镜像可用：容器内 `torch.cuda.is_available()==True`
- 数据一致：train/val/test 样本计数与原机一致
- 续跑可用：`--mode strict-resume` 或 `--mode speed-restart` 至少跑通 1-2 epoch
- 追溯完整：能从结果表回溯到日志、checkpoint、wandb run

## 9. 常见问题
- `CUDA not available`：优先检查 `nvidia-container-toolkit` 与 `docker run --gpus all`。
- `W&B credentials missing`：先导出 `WANDB_API_KEY` 或完成 `wandb login`。
- `resume` 路径错误：确认 `--resume-ckpt` 指向真实 `.pt` 文件。
