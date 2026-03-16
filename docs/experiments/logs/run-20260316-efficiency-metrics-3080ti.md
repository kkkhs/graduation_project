# run-20260316-efficiency-metrics-3080ti

status: executed  
evidence: commands + outputs + artifacts

## 1) 执行目标
- 在 3080Ti 云主机完成三模型效率项补齐：`FPS / Params(M)`，并尽可能补 `FLOPs(G)`。
- 统一口径：`imgsz=512, batch=1, warmup=50, timing=200`。

## 2) 机器与环境
- SSH: `ssh -p 29137 root@connect.westb.seetacloud.com`
- GPU: `NVIDIA GeForce RTX 3080 Ti (12GB)`，空闲可用。
- 环境: `/root/autodl-tmp/envs/shipdet`。
- 数据路径: `/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/test/images`

## 3) 关键命令与结果

### 3.1 YOLO26n
- 权重:  
  `/root/autodl-tmp/experiment_assets/runs/detect/runs/yolo26_main_512_formal012/weights/best.pt`
- 结果:
  - `FPS=69.0625`
  - `Params=2.5042M`
  - `FLOPs=3.6939G`
- 产物:
  - `/root/autodl-tmp/experiment_assets/benchmarks/latest_efficiency_prep/yolo26_efficiency.json`

### 3.2 DRENet
- 权重:  
  `/root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights/best.pt`
- 备注:
  - 由于 torch 新版本默认 `weights_only=True`，需设置：
    - `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
- 结果:
  - `FPS=121.8993`
  - `Params=4.7882M`
  - `FLOPs=4.2057G`
- 产物:
  - `/root/autodl-tmp/experiment_assets/benchmarks/latest_efficiency_prep/drenet_efficiency.json`

### 3.3 FCOS（MMDetection）
- 权重:  
  `/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/best_coco_bbox_mAP_50_epoch_14.pth`
- 环境处理：
  - `shipdet` 环境缺少 `mmcv._ext`，无法直接跑 `get_flops.py`。
  - 切换到现有可用环境：`/root/miniconda3/envs/fcos310`（`mmcv._ext` 可用）。
- 执行命令：
  - `conda activate /root/miniconda3/envs/fcos310`
  - `python tools/analysis_tools/get_flops.py <fcos_config.py>`
- 结果:
  - `FPS=45.9559`（日志反推）
  - `Params=32.1664M`
  - `FLOPs=0.123T`（约 `123G`，MMDet输出）
  - `get_flops.py` 输出的实际计算输入：`800x800`
  - `stable-ckpt(epoch_102)`（系统口径）补充：
    - `AP50=0.7750`
    - `AP50:95=0.2880`
    - `AP75=0.1220`
    - `val time=0.0219s/img`，`FPS≈45.6621`
- 产物:
  - `/root/autodl-tmp/experiment_assets/benchmarks/latest_efficiency_prep/fcos_efficiency.json`

## 4) 回传与本地落盘
- 已回传到本地：
  - `experiment_assets/benchmarks/efficiency_prep_20260315_195647/drenet_efficiency.json`
  - `experiment_assets/benchmarks/efficiency_prep_20260315_195647/fcos_efficiency.json`
  - `experiment_assets/benchmarks/efficiency_prep_20260315_195647/yolo26_efficiency.json`

## 5) 文档回填
- 已更新主对比表：
  - `docs/results/baselines.md`
- FCOS 行已明确标注：
  - `FPS` 为 `val log` 反推口径（带 `*`）
  - `FLOPs` 已补（并注明计算输入为 `800x800`）

## 6) stable 指标证据位置
- 日志文件：  
  `/root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/20260315_171908/20260315_171908.log`
- 对应行（示例）：
  - `Epoch(val) [102][788/788] coco/bbox_mAP: 0.2880  coco/bbox_mAP_50: 0.7750  coco/bbox_mAP_75: 0.1220 ... time: 0.0219`
