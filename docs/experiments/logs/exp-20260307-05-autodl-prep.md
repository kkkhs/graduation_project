# 实验记录：AutoDL 无卡主机准备与 DRENet 续训迁移（2026-03-07）

## 1. 目标
- 在 AutoDL 无卡实例上完成 DRENet 云端续训前的环境准备。
- 保证后续切换到有卡实例时，可直接复用：
  - 数据集
  - 代码与补丁
  - `last.pt/best.pt`
  - 正式实验元数据
  - wandb 续训入口

## 2. 云端实例事实
- SSH：`ssh -p 49353 root@connect.westc.gpuhub.com`
- 主机：`autodl-container-b781468847-4a4cea59`
- 系统：Ubuntu 22.04
- GPU：当前实例无 GPU（`nvidia-smi` 无设备）
- 数据盘：`/root/autodl-tmp`，约 `50G`
- 系统盘：`/`，约 `30G`

## 3. 已完成动作

### 3.1 数据与资产上传
- 上传 `train.zip`、`val.zip`、`test.zip` 到：
  - `/root/autodl-tmp/transfer/`
- 上传续训资产包：
  - `/root/autodl-tmp/transfer/drenet_resume_bundle_20260307.tar.gz`

### 3.2 数据集展开
- 目标目录：
  - `/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship`
- 结果：
  - `train/images`
  - `train/labels`
  - `train/degrade`
  - `val/images`
  - `val/labels`
  - `val/degrade`
  - `test/images`
  - `test/labels`
  - `test/degrade`

### 3.3 代码固定
- 本仓库：
  - `/root/autodl-tmp/workspace/graduation_project`
  - commit：`f34758aef1814fbd8ff713606a4d955d2020b04b`
- DRENet：
  - `/root/autodl-tmp/workspace/experiments/drenet/DRENet`
  - commit：`a187dbe0f623b521a62c6176c7cafaa7322f5f66`

### 3.4 补丁应用
- 补丁：
  - `/root/autodl-tmp/transfer/drenet_local_compat_20260307.patch`
- 应用结果：
  - `utils/datasets.py`
  - `utils/general.py`
  - `utils/loss.py`
  已进入修改状态。

### 3.5 续训资产展开
- 目标 run：
  - `/root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01`
- 已确认存在：
  - `weights/last.pt`
  - `weights/best.pt`
  - `results.txt`
  - `opt.yaml`
  - `hyp.yaml`
- 已确认存在的归档 checkpoint：
  - `/root/autodl-tmp/experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_best_ep100_20260307_165716.pt`
  - `/root/autodl-tmp/experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_best_ep100_20260307_165755.pt`
  - `/root/autodl-tmp/experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_last_ep100_20260307_165716.pt`
  - `/root/autodl-tmp/experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_last_ep100_20260307_165755.pt`

### 3.6 续训入口固化
- 新增云端数据配置：
  - `/root/autodl-tmp/workspace/experiments/drenet/DRENet/data/levir_ship_autodl.yaml`
- 新增云端续训脚本：
  - `/root/autodl-tmp/workspace/experiments/drenet/DRENet/scripts/resume_formal01_autodl.sh`
- 当前默认入口：
  - `python train.py --resume "$LAST_PT"`
- 备用 fallback：
  - `python train.py --weights "$LAST_PT" --data "$ROOT/data/levir_ship_autodl.yaml" --epochs 1000 --batch-size 4 --img-size 512 512 --workers 2 --project /root/autodl-tmp/experiment_assets/runs --name "$RUN_NAME" --exist-ok`

### 3.7 Python 环境
- conda 环境：
  - `/root/autodl-tmp/envs/shipdet`
- 已完成：
  - Python 3.10 环境创建
  - `pip install Cython matplotlib numpy opencv-python-headless Pillow PyYAML scipy tensorboard tqdm wandb seaborn pandas thop pycocotools`
  - `pip install torchvision`
- 当前环境核查：
  - `torch==2.10.0+cu128`
  - `torch.cuda.is_available()==False`（因为当前实例无 GPU）
  - `wandb==0.25.0`

## 4. 与 wandb 相关的事实
- 正式训练已接入 wandb。
- 当前同步过来的 run 目录中未直接看到单独的本地 `wandb/` 目录，但这不等于 wandb 元数据丢失。
- 已从 `last.pt` 读取到：
  - `epoch=100`
  - `wandb_id=94d4wdmk`
- 当前云端登录态核查结果：
  - `/root/.netrc` 已创建
  - `/root/.config/wandb` / `/root/.wandb` 不存在
- 已尝试 `wandb login --relogin <provided_key>`：
  - 第一次：失败
  - CLI 报错：`API key must have 40+ characters`
  - 第二次：成功
  - CLI 输出：`W&B API key is configured`
- 结论：
  - 最新提供的 key 已成功写入 `/root/.netrc`
  - wandb key 当前**已配置**
  - 但 checkpoint 已携带 `wandb_id`，后续只要在有卡实例上先完成 `wandb login`，再执行 `--resume <last.pt>`，就有机会延续同一 run 语义

## 5. 当前结论
- AutoDL 无卡实例已经达到“可迁移到有卡实例继续跑”的准备状态。
- 当前还没有启动训练，原因是：
  - 该实例无 GPU
  - 当前实例仅用于准备环境，不承担训练

## 6. 下一步
- [ ] 在有卡实例上执行 `wandb login`
- [ ] 切换到有卡实例后执行 `resume 1-2 epoch` 冒烟
- [ ] 冒烟通过后继续正式续训到更高总轮次
