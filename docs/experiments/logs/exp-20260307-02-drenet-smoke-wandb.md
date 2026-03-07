# 实验记录：DRENet 冒烟与 wandb 复跑（2026-03-07）

## 1. 阶段目标
- 在训练机 `E:\workspace` 完成 DRENet 最小可用训练闭环（环境、数据、训练、产物）。
- 采用“命令日志 + 关键截图”留痕，满足可审计与可复现。
- 接入 wandb，验证云端实验追踪可用。

## 2. 环境与资产状态（结论）
- 结论：训练机“环境已就绪，实验资产已补齐，可正式开跑”。
- 已确认：
  - GPU：RTX 3060 6GB 可用
  - `torch.cuda.is_available()==True`
  - Conda：`dl`（Python 3.11 / torch 2.6.0+cu124）
  - 磁盘：`E:` 可用空间充足（约 372GB）
- 落地目录：
  - 项目：`E:\workspace\projects\graduation_project`
  - 代码：`E:\workspace\projects\graduation_project\experiments\drenet\DRENet`
  - 数据：`E:\workspace\datasets\LEVIR-Ship`

## 3. 关键执行与结果

### 3.1 非 wandb 冒烟（1 epoch）
- 输出目录：`E:\workspace\runs\drenet_smoke`
- 权重产物：
  - `E:\workspace\runs\drenet_smoke\weights\best.pt`
  - `E:\workspace\runs\drenet_smoke\weights\last.pt`
- 结论：训练主流程跑通，退出码 `0`。

### 3.2 wandb 冒烟复跑（1 epoch）
- 命令：
  - `powershell -ExecutionPolicy Bypass -File E:\workspace\scripts\run_drenet_train.ps1 -Epochs 1 -BatchSize 2 -Workers 2 -RunName drenet_smoke_wandb_20260307_135422`
- 输出目录：`E:\workspace\runs\drenet_smoke_wandb_20260307_135422`
- 权重产物：
  - `E:\workspace\runs\drenet_smoke_wandb_20260307_135422\weights\best.pt`
  - `E:\workspace\runs\drenet_smoke_wandb_20260307_135422\weights\last.pt`
- 日志：
  - `E:\workspace\runs\trace\train_drenet_smoke_wandb_20260307_135422_20260307_135423.log`
- wandb：
  - Project: <https://wandb.ai/2964745405-hfut/runs>
  - Run: <https://wandb.ai/2964745405-hfut/runs/runs/961t6yk2>
- 结论：wandb 同步成功，退出码 `0`。

## 4. 留痕资产
- 统一日志目录：`E:\workspace\runs\trace`
- 截图目录：`E:\workspace\runs\trace\shots`
- 关键截图（命令证据图）：
  - `E:\workspace\runs\trace\shots\logshot_smoke_train_final_20260307_133739.png`
  - `E:\workspace\runs\trace\shots\logshot_smoke_wandb_20260307_135422.png`

## 5. 过程中问题与修复（已落地）
- `setuptools` 兼容问题导致 `pkg_resources` 缺失：降级 `setuptools<81`。
- `numpy` 新版移除 `np.int`：代码改为 `int`。
- PyTorch 2.6 `torch.load` 默认 `weights_only=True`：相关位置显式设为 `weights_only=False`。
- `loss.py` 中 `clamp_` 参数类型不兼容：上界强制转 `int`。

## 6. 当前判定
- 本阶段判定：DRENet 冒烟与 wandb 接入均已完成，可进入正式实验阶段。

## 7. 待办（正式跑前）
1. 固化环境（导出 `requirements` 或 conda `environment.yml`）。
2. 确认正式实验命名规范（含 `seed` / `batch` / `augment` 标签）。
3. 执行正式训练并将指标回填至结果文档（`docs/results/*`）。
