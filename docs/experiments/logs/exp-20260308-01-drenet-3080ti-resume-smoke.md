# 实验记录：DRENet 在 3080 Ti 上的续训冒烟（2026-03-08）

## 1. 目标
- 在有卡云主机上验证 DRENet 正式实验可从 `epoch=100` 继续续训。
- 验证项：
  - CUDA 可用
  - 数据路径可读
  - wandb 能恢复到原 run
  - 训练不是从 `0` 开始
  - `results.txt`、`last.pt`、`best.pt` 会更新

## 2. 机器与环境
- SSH：`ssh -p 29137 root@connect.westb.seetacloud.com`
- 主机：`autodl-container-a2fd40a1b8-db880c0e`
- GPU：`NVIDIA GeForce RTX 3080 Ti 12GB`
- 数据盘：`/root/autodl-tmp`
- Python 环境：`/root/autodl-tmp/envs/shipdet`
- DRENet commit：`a187dbe0f623b521a62c6176c7cafaa7322f5f66`

## 3. 预检查事实
- 数据集已存在于：
  - `/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship`
- 续训权重已存在于：
  - `/root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights/last.pt`
- checkpoint 元数据已确认：
  - `epoch=100`
  - `wandb_id=94d4wdmk`
- wandb 登录态已存在：
  - `/root/.netrc`

## 4. 本轮遇到的真实问题

### 问题 A：`pkg_resources` 缺失
- 现象：
  - `ModuleNotFoundError: No module named 'pkg_resources'`
- 根因：
  - 当前 venv 中 `setuptools==82.0.0` 不再提供 DRENet 旧代码依赖的 `pkg_resources`
- 处理：
  - 执行 `pip install 'setuptools<81'`
- 结果：
  - 修复成功

### 问题 B：requirements 检查不认 `opencv-python-headless`
- 现象：
  - `DistributionNotFound: The 'opencv-python>=4.1.2' distribution was not found`
- 根因：
  - DRENet 的 requirements 检查要求的是 `opencv-python`
- 处理：
  - 补装 `opencv-python`
- 结果：
  - 修复成功

### 问题 C：`train.py` 仍受 PyTorch 2.6+ `weights_only=True` 影响
- 现象：
  - `_pickle.UnpicklingError: Weights only load failed`
- 根因：
  - `train.py` 中 `torch.load(weights, map_location=device)` 没有显式传 `weights_only=False`
- 处理：
  - 在云端 `train.py` 中补成：
    - `torch.load(weights, map_location=device, weights_only=False)`
- 结果：
  - 修复成功

## 5. 冒烟命令
```bash
/root/autodl-tmp/envs/shipdet/bin/python train.py \
  --weights /root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights/last.pt \
  --data /root/autodl-tmp/workspace/experiments/drenet/DRENet/data/levir_ship_autodl.yaml \
  --epochs 102 \
  --batch-size 4 \
  --img-size 512 512 \
  --workers 2 \
  --device 0 \
  --project /root/autodl-tmp/experiment_assets/runs \
  --name drenet_levirship_512_bs4_sna_20260307_formal01 \
  --exist-ok
```

## 6. 关键结果
- GPU 已被训练进程实际使用：
  - `CUDA:0 (NVIDIA GeForce RTX 3080 Ti, 11910.625MB)`
- wandb 恢复到原 run：
  - run id：`94d4wdmk`
  - run 链接：<https://wandb.ai/2964745405-hfut/runs/runs/94d4wdmk>
- 训练没有从 `0` 重启，而是从：
  - `101/101`
  开始执行
- 训练完成了 `1` 个续训 epoch
- 验证阶段输出：
  - `P=0.379`
  - `R=0.776`
  - `mAP@0.5=0.681`
  - `mAP@0.5:0.95=0.242`

## 7. 产物路径
- 运行目录：
  - `/root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01`
- 训练日志：
  - `/root/autodl-tmp/experiment_assets/runs/trace/train_drenet_resume_smoke_102_20260308_095552.log`
- 更新后的权重：
  - `/root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights/last.pt`
  - `/root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights/best.pt`

## 8. 结论
- 3080 Ti 环境上的 DRENet 续训链路已经打通。
- 当前正式实验可以继续从 `100 epoch` 往后跑。
- 在不改配置的前提下，下一步可以直接把总轮次继续拉向更高目标轮次。

## 9. 后续补充事实
- 这次 smoke 完成后，`runs/.../weights/last.pt` 被 DRENet 例行 strip 成了纯权重文件。
- 因此后续正式续训不能直接依赖该 `last.pt` 保存的元数据，而应改用：
  - `/root/autodl-tmp/experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_last_ep100_20260307_165755.pt`
- 若仍需接回原 W&B run，还需要显式设置：
  - `WANDB_RUN_ID=94d4wdmk`
  - `WANDB_RESUME=allow`

## 10. 后续动作
- [ ] 把 `train.py` 的 `weights_only=False` 修复补进本地 patch 文件
- [x] 决定正式续训目标总轮次（`300`）
- [x] 启动正式长跑
- [x] 解决云端自动回传脚本对 `rsync` 的依赖问题
