# 下一轮实验计划（2026-03-06）

## 目标
- 在 3060 训练机完成三模型冒烟训练（1 epoch）并确认流程可跑通。
- 至少完成 DRENet 正式训练启动与一次评测，产出首轮 AP50/P/R/F1。
- 回传首轮产物后，回填 `docs/results/baselines.md` 的 DRENet 行。

## 执行顺序（严格按序）
1. 环境检查：`Python/torch/CUDA`、数据目录、磁盘空间。
2. 数据准备：确认 `LEVIR-Ship` 划分；生成 COCO 标注（供 MMDet）。
3. 冒烟训练：DRENet -> MMDet(FCOS) -> YOLOv8n（都跑 1 epoch）。
4. 正式训练：先跑 DRENet 主实验（300 epoch，可中断续训）。
5. 评测导出：DRENet `test.py` + `detect.py`，收集指标与图像。
6. 结果回传：执行 `scripts/sync_results_from_laptop.sh`。
7. 文档回填：`docs/results/baselines.md`、`docs/results/qualitative.md`。

## 今日必跑命令（3060 PowerShell）

### 1) DRENet 冒烟（必须）
```powershell
cd E:\work\DRENet
python train.py --cfg .\models\DRENet.yaml --epochs 1 --workers 2 --batch-size 2 --device 0 --project .\runs\drenet_smoke --data .\data\ship.yaml
```

### 2) DRENet 正式训练（必须）
```powershell
cd E:\work\DRENet
python train.py --cfg .\models\DRENet.yaml --epochs 300 --workers 4 --batch-size 4 --device 0 --project .\runs\drenet_main --data .\data\ship.yaml
```

### 3) DRENet 评测（必须）
```powershell
cd E:\work\DRENet
python test.py --weights .\runs\drenet_main\exp\weights\best.pt --project .\runs\drenet_eval --device 0 --batch-size 4 --data .\data\ship.yaml
python detect.py --weights .\runs\drenet_main\exp\weights\best.pt --source E:\datasets\LEVIR-Ship\test\images --device 0
```

### 4) MMDet/YOLO 冒烟（建议当天完成）
```powershell
# MMDet(FCOS) 1 epoch
cd E:\work\mmdetection
python tools/train.py configs\fcos\fcos_r50-caffe_fpn_gn-head_1x_coco.py `
  --work-dir .\work_dirs\fcos_smoke `
  --cfg-options train_cfg.max_epochs=1 model.bbox_head.num_classes=1 `
  train_dataloader.dataset.data_root=E:/datasets/LEVIR-Ship/ `
  train_dataloader.dataset.ann_file=annotations/train.json `
  train_dataloader.dataset.data_prefix.img=train/images/ `
  val_dataloader.dataset.data_root=E:/datasets/LEVIR-Ship/ `
  val_dataloader.dataset.ann_file=annotations/val.json `
  val_dataloader.dataset.data_prefix.img=val/images/ `
  test_dataloader.dataset.data_root=E:/datasets/LEVIR-Ship/ `
  test_dataloader.dataset.ann_file=annotations/test.json `
  test_dataloader.dataset.data_prefix.img=test/images/

# YOLOv8n 1 epoch
cd E:\work\ultralytics
yolo detect train model=yolov8n.pt data=E:/datasets/LEVIR-Ship/ship.yaml imgsz=640 epochs=1 batch=2 device=0 project=E:/work/runs name=yolo_smoke
```

## 验收标准（今日）
- DRENet 冒烟日志与权重存在：`runs/drenet_smoke/.../weights`.
- DRENet 正式训练已开始，并至少完成 1 个 epoch。
- DRENet 评测输出可读，拿到 AP50/P/R（至少一组数字）。
- 至少导出 3 张测试集可视化图（成功/漏检/误检各 1 张）。

## 风险与兜底
- OOM：先降 `batch-size` 到 `2`，再降输入尺寸。
- 中断：使用 `--resume` 从 `last.pt` 续训。
- 路径错：优先检查 `ship.yaml` 与 `annotations/*.json` 路径。

