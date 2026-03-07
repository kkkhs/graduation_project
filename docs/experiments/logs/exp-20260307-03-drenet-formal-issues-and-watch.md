# 实验记录：DRENet 正式训练问题总结与监控（2026-03-07）

## 1. 当前正式训练上下文
- 运行名：`drenet_levirship_512_bs4_sna_20260307_formal01`
- 运行目录：`E:\workspace\runs\drenet_levirship_512_bs4_sna_20260307_formal01`
- 训练状态：进行中（单实例）
- 当前进度（记录时）：约 `36/299`

## 2. 本阶段问题总结

### 问题 A：同名 run 被重复启动（双跑）
- 现象：
  - 同名 `wandb run` 出现 2 条 `Running`。
  - 本机存在两套 `train.py` 进程同时运行。
- 影响：
  - GPU/CPU 资源竞争，训练速度显著下降。
  - 结果可追溯性变差（同名 run 混淆）。
- 根因：
  - 首次启动后又二次启动，且首条链路存在孤儿进程。
- 处理：
  - 清理 14:19:35 这组孤儿训练链路，仅保留 14:20:10 启动的正式链路。
- 当前状态：已修复。

### 问题 B：wandb 页面“旧 run 显示 Running 但日志不更新”
- 现象：
  - 旧 run 日志文件停止更新，但页面仍显示 Running。
- 根因：
  - 旧 run 残留 `wandb-core` / `gpu_stats` 心跳进程导致状态滞后。
- 处理：
  - 已清理旧 run 对应残留进程。
  - API 核验结果：
    - `3d6erzre`：`crashed`（旧 run）
    - `94d4wdmk`：`running`（当前 run）
- 当前状态：已修复（页面可能存在短时缓存延迟）。

### 问题 C：Docker 已安装但 daemon 未起
- 现象：
  - Docker CLI 可用，但 `docker info` 报 Docker Desktop 无法启动。
- 根因：
  - 系统存在 `RebootPending`，WSL 相关组件启用后待重启生效。
- 处理：
  - 已完成 Docker Desktop 安装与前置组件检查。
  - 待训练到目标 checkpoint 后手动停训并重启系统。
- 当前状态：待执行重启步骤。

## 3. 训练速度与参数体检（当前）
- 参数：`epochs=300`, `batch=4`, `img=512`, `workers=2`
- 单 run 实测速度：约 `90 sec/epoch`（近期窗口测得）
- 预计剩余时长（记录时）：约 6~7 小时
- 参数结论：当前配置稳定可跑；若迁移 4090，可在下一次 run 提升 `batch/workers`。

## 4. checkpoint 停机计划
- 目标停机点：`epoch=100`（用于安全中断与 Docker 续跑）
- 达到目标后动作：
  1. 停止当前训练进程；
  2. 备份 `last.pt` / `best.pt`；
  3. 重启系统；
  4. 启动 Docker 与 GPU 容器验证；
  5. 用 Docker 续跑。

## 5. 证据与留痕路径
- 训练日志：`E:\workspace\runs\trace\train_drenet_levirship_512_bs4_sna_20260307_formal01_*.log`
- 速度/状态核验：`E:\workspace\runs\drenet_levirship_512_bs4_sna_20260307_formal01\results.txt`
- Docker 安装状态：`E:\workspace\runs\trace\docker_install_status_20260307_150129.log`
