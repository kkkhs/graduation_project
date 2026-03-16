# exp-20260307-04-cloud-migration-manual-copy-and-next-steps

## 1. 阶段结论
- 本地训练机已完成 DRENet 正式训练到 `epoch=100` 的阶段性结果沉淀。
- 后续不再使用 Docker 流程，统一采用原生 Python/Conda 训练与续跑。
- 当前策略为：先把 `datasets/runs/checkpoints` 手动归档到主机（Mac），再在云主机继续训练。

## 2. 已有关键资产（训练机）
- 数据集：`E:\workspace\datasets\LEVIR-Ship`
- 运行输出：`E:\workspace\runs\drenet_levirship_512_bs4_sna_20260307_formal01`
- 归档权重：`E:\workspace\checkpoints\drenet\drenet_levirship_512_bs4_sna_20260307_formal01_*_ep100_*.pt`
- 命令留痕：`E:\workspace\runs\trace\`

## 3. 手动同步建议（Mac）
建议放入项目目录下的隐藏资产目录：
- `datasets -> ~/codes/graduation_project/.workspace_assets/datasets`
- `runs -> ~/codes/graduation_project/.workspace_assets/runs`
- `checkpoints -> ~/codes/graduation_project/.workspace_assets/checkpoints`

并确保 Mac 端项目 `.gitignore` 包含：
- `.workspace_assets/`

## 4. 上云前闸门（必须通过）
1. 代码一致性
- 云主机代码 commit 与当前实验记录一致。

2. 数据一致性
- `train/val/test` 计数与本地一致。

3. checkpoint 可用
- 云端存在可续跑的 `last.pt`。

4. 环境可用
- `torch.cuda.is_available()==True`
- 依赖安装完整（PyTorch、OpenCV、NumPy、wandb 等）。

## 5. 云主机续跑流程
1. 先跑 `1-2 epoch` resume 冒烟测试，只验证链路。
2. 冒烟通过后开启正式长跑（例如 300 -> 1000）。
3. wandb 规范：
- `project=graduation-drenet`
- `run.name=<exp_name>`
- tags 至少包含：`dataset,imgsz,bs,seed,stage=formal,host`

## 6. 验收标准
- 续跑 epoch 连续（不是从 0 重启）。
- 新的 `last.pt/best.pt` 正常更新。
- wandb 指标曲线持续记录，run 元数据完整。
- 能回溯到：代码 commit、数据路径、checkpoint、日志证据。

## 7. 下一步
- [ ] 完成 `datasets/runs/checkpoints` 到 Mac 主机归档
- [ ] 在 Mac 项目仓确认 `.workspace_assets/` 已忽略
- [ ] 云主机拉取同一 commit 代码
- [ ] 云主机执行 resume 1-2 epoch 冒烟测试
- [ ] 通过后启动正式续跑并记录 wandb 链接
- [ ] 回填 `docs/results/baselines.md` 与实验日志

## 8. 备注
- 本阶段重点是“迁移与续跑链路稳定”，不是立即追峰值。
