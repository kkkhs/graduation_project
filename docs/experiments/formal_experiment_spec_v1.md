# DRENet 正式实验记录与配置基线（v1）

> 适用范围：DRENet + LEVIR-Ship，训练机 `E:\workspace`，wandb 强制开启。

## 1. 命名与目录（统一规范）
- `exp_name`：`drenet_{dataset}_{imgsz}_{bs}_{seed}_{date}`
- 示例：`drenet_levirship_640_bs4_s42_20260307`
- 本地输出目录：`E:\workspace\runs\<exp_name>\`
- 留痕目录：`E:\workspace\runs\trace\`
- 截图目录：`E:\workspace\runs\trace\shots\`
- 权重归档目录：`E:\workspace\checkpoints\drenet\`

## 2. 正式实验必记字段

### 2.1 元数据
- `exp_id`、`exp_name`、日期、模型名、任务阶段（`stage=formal`）
- 代码 commit（当前仓库 commit）
- 数据版本与路径（含 train/val/test）
- 随机种子、操作者、机器标识（主机名）

### 2.2 训练配置
- `imgsz`、`batch`、`epochs`、`workers`
- 优化器、初始学习率、学习率策略
- 数据增强配置
- `amp`、梯度累积（`grad_accum`）

### 2.3 环境配置
- conda 环境名
- Python / torch / CUDA / cuDNN 版本
- GPU 型号与显存
- 关键依赖版本（如 numpy、opencv）

### 2.4 结果与异常
- `AP50`、`Precision`、`Recall`、`F1`
- best epoch、总训练耗时、推理耗时（如有）
- 失败重试记录（报错时间、命令、修复动作）

### 2.5 产物索引
- 命令日志路径（`runs/trace/*.log`）
- 权重路径（`best.pt` / `last.pt`）
- 关键可视化路径（曲线、混淆矩阵、预测图）
- wandb 项目链接、run 链接
- 关键截图路径（至少环境检查、训练启动、结果确认、异常）

## 3. wandb 规范（正式实验必须）
- `project` 固定：`graduation-drenet`
- `run.name`：使用 `<exp_name>`
- tags 至少包含：`dataset`、`imgsz`、`bs`、`seed`、`stage=formal`
- wandb 必须记录：
  - config（完整超参数）
  - 训练/验证关键指标曲线
  - 最终 best 指标
  - artifacts（`best.pt`、`last.pt`）

## 4. 执行闸门（每次正式跑前）
- 环境闸门：
  - `torch.cuda.is_available()==True`
  - GPU 可见，显存信息可读取
- 数据闸门：
  - `train/val/test` 目录完整
  - 样本计数符合预期
- 运行闸门：
  - 先做 dry-run（`1 epoch` + 小 batch）
  - dry-run 无路径/依赖错误后再启动正式训练

## 5. 执行入口（默认）
- 使用统一脚本：
  - `powershell -ExecutionPolicy Bypass -File E:\workspace\scripts\run_drenet_train.ps1 ...`
- 要求：通过同一入口同时保证命令留痕与 wandb 上报。
