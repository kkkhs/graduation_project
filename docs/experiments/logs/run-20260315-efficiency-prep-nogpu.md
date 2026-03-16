# run-20260315-efficiency-prep-nogpu

状态: 已执行  
evidence: commands + outputs + blockers

## 1) 目标
- 在无卡实例先完成“三模型效率指标（FPS/Params/FLOPs）”的前置准备：
  - 校验远端路径与关键权重
  - 生成统一测量清单与结果模板
  - 提前暴露环境阻塞，避免有卡时浪费算力

## 2) 远端连接与基础事实
- SSH: `ssh -p 29137 root@connect.westb.seetacloud.com`
- 主机: `autodl-container-a2fd40a1b8-db880c0e`
- 当前无卡: `nvidia-smi -> No devices were found`
- 代码仓库: `/root/autodl-tmp/workspace/graduation_project`

## 3) 已执行命令与结果
1. 拉取最新代码（含实验文档更新）
- 命令: `git pull --rebase`
- 结果: `Fast-forward` 成功，分支更新到最新 `main`。

2. 校验关键权重与路径
- DRENet best:  
  `/root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights/best.pt`
- FCOS best:  
  `/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/best_coco_bbox_mAP_50_epoch_14.pth`
- YOLO best:  
  `/root/autodl-tmp/experiment_assets/runs/detect/runs/yolo26_main_512_formal012/weights/best.pt`

3. 生成效率测量准备包（远端）
- 输出目录:
  - `/root/autodl-tmp/experiment_assets/benchmarks/efficiency_prep_20260315_195647/`
  - 软链: `/root/autodl-tmp/experiment_assets/benchmarks/latest_efficiency_prep`
- 生成文件:
  - `benchmark_manifest.json`
  - `efficiency_results_template.json`
  - `run_gpu_efficiency_commands.md`

4. 无卡预检（能做的都做了）
- YOLO 权重加载测试:
  - 命令: `from ultralytics import YOLO; YOLO(best.pt)`
  - 结果: `YOLO_LOAD_OK DetectionModel`
- MMDet FLOPs命令预跑:
  - 首次报错: `ModuleNotFoundError: No module named 'mmcv'`
  - 处理: 安装 `mmcv-lite==2.1.0`
  - 二次报错: `ModuleNotFoundError: No module named 'mmcv._ext'`

## 4) 当前阻塞与结论
- 阻塞点：FCOS 的 `get_flops.py` 依赖 `mmcv.ops`（`mmcv._ext`），仅 `mmcv-lite` 不够。
- 结论：无卡阶段已完成全部可做准备；FCOS FLOPs 需要在“匹配 torch/cuda 的可用 mmcv-full 环境”执行。

## 5) GPU 到位后的直接执行顺序
1. 进入 `shipdet` 环境并确认 GPU 可见。  
2. 先跑 YOLO 与 DRENet 的 Params/FLOPs/FPS（无 mmcv 阻塞）。  
3. 为 FCOS 补齐可用 `mmcv-full`（或切换到已验证的 mmdet 环境）后测 FLOPs/FPS。  
4. 将结果填入 `docs/results/baselines.md` 的 `FPS` / `Params(M)`（FLOPs 可同步记录在实验日志备注）。
