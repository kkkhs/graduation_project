# 计划：YOLO26 首轮执行准备（2026-03-08）

status: archived
execution: completed_and_superseded
note: 首轮已执行完成，结果与问题请看 `docs/experiments/logs/run-20260308-yolo26-3060-3080ti.md`。

## 1. 目标
- 在不改变当前 DRENet 已完成结果的前提下，启动 YOLO26 首轮实验链路：
  - `1 epoch` 冒烟
  - 正式训练
  - 评测与指标回填
- 统一口径输出：
  - `AP50`（主）
  - `AP@0.5:0.95`（辅）
  - `Precision/Recall/F1`
  - `FPS/Params/FLOPs`

## 2. 已确认前置条件
- DRENet 正式 run 已完成并回传，本轮可独立推进 YOLO26。
- 统一结果表已预留 YOLO26 行：
  - `docs/results/baselines.md`
- 消融表已预留 YOLO26 尺寸行：
  - `docs/results/ablation.md`

## 3. 执行环境建议
- 优先环境：云 GPU（Linux）或 3060 笔记本（二选一即可）。
- 统一数据：LEVIR-Ship，`ship` 单类。
- 统一输入建议：`imgsz=640`（后续可做 512/640/768 消融）。

## 4. 命令清单（云端 Linux 版）

### 4.1 环境与版本核验
```bash
nvidia-smi
python3 --version
python3 -m pip install -U ultralytics
yolo version
```

### 4.2 冒烟（1 epoch）
```bash
cd /workspace/ultralytics
yolo detect train \
  model=yolo26n.pt \
  data=/data/LEVIR-Ship/ship.yaml \
  imgsz=640 \
  epochs=1 \
  batch=2 \
  device=0 \
  project=/workspace/runs \
  name=yolo26_smoke
```

### 4.3 正式训练（首轮）
```bash
cd /workspace/ultralytics
yolo detect train \
  model=yolo26n.pt \
  data=/data/LEVIR-Ship/ship.yaml \
  imgsz=640 \
  epochs=100 \
  batch=12 \
  device=0 \
  project=/workspace/runs \
  name=yolo26_main
```

### 4.4 评测
```bash
cd /workspace/ultralytics
yolo detect val \
  model=/workspace/runs/yolo26_main/weights/best.pt \
  data=/data/LEVIR-Ship/ship.yaml \
  imgsz=640 \
  batch=4 \
  device=0
```

### 4.5 效率与复杂度（统一协议）
```bash
# FPS（建议 warmup 50 + timing 200，batch=1，imgsz=512）
# 可用 ultralytics benchmark/profile 或自定义计时脚本，记录测试条件。

# Params/FLOPs（建议与 baseline 表一致口径）
# 记录 Params(M)、FLOPs(G) 到实验日志和 baselines 备注。
```

## 5. 回填目标
- `docs/results/baselines.md`
  - 补 YOLO26 行：AP50/AP50:95/P/R/F1/FPS/Params
- `docs/results/ablation.md`
  - 补 YOLO26 分辨率消融（至少 1 组）
- `docs/results/qualitative.md`
  - 补 success/miss/false_positive 图例

## 6. 风险与降级策略
- 若 `yolo26n.pt` 在当前版本不可用：
  - 先升级 Ultralytics；
  - 仍不可用时，临时用 `yolo11n.pt` 完成冒烟，正式训练再切回 YOLO26。
- 若 OOM：
  - 按顺序调：`batch -> imgsz -> workers/cache`。
- 若路径错误：
  - 先核查 `ship.yaml`，再核查数据目录权限和存在性。

## 7. 本轮验收标准
- 冒烟命令退出码为 0。
- 正式训练可稳定启动并生成 `weights/best.pt`、`weights/last.pt`。
- 评测可读出 AP50/AP50:95。
- 至少 1 条 YOLO26 结果写入 `docs/results/baselines.md`。
