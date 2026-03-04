# 3060 训练机可复制执行手册（PowerShell + 固定 commit）

> 适用系统：Windows 原生 PowerShell
> 默认路径前缀：
> - 项目：`E:/work/graduation_project`
> - 数据：`E:/datasets/LEVIR-Ship`

## 0. 从零准备（工作目录 + 数据集下载）
```powershell
# 创建工作目录
New-Item -ItemType Directory -Force E:\work | Out-Null
New-Item -ItemType Directory -Force E:\datasets | Out-Null

# 拉项目
cd E:\work
git clone <repo_url> graduation_project

# 下载 LEVIR-Ship 数据集
cd E:\datasets
git clone https://github.com/WindVChen/LEVIR-Ship.git

# 结构检查（应看到 train/val/test）
Get-ChildItem E:\datasets\LEVIR-Ship
```

## 1. SSH 连接与项目同步
```powershell
ssh <user>@<host>
cd E:\work\graduation_project
git pull
```

## 2. Conda 环境创建
```powershell
conda create -n shipdet python=3.10 -y
conda activate shipdet
```

## 3. 依赖安装（按框架拆分）
```powershell
pip install -U pip
pip install pyyaml pillow matplotlib numpy scipy pandas tqdm opencv-python
# 按训练机 CUDA 版本安装 torch（示例 cu121）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3.1 YOLO 依赖
```powershell
pip install ultralytics
```

### 3.2 MMDet 依赖
```powershell
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

## 4. 数据配置（DRENet 与 MMDet）

### 4.1 DRENet 的 `ship.yaml`
编辑：`E:\work\DRENet\data\ship.yaml`，将 `train/val/test` 路径指向你的 `E:/datasets/LEVIR-Ship`。

### 4.2 MMDet 需要 COCO 标注（train/val/test.json）
在项目仓库执行：
```powershell
cd E:\work\graduation_project

python tools/convert_yolo_to_coco.py --images E:/datasets/LEVIR-Ship/train/images --labels E:/datasets/LEVIR-Ship/train/labels --output E:/datasets/LEVIR-Ship/annotations/train.json --category-name ship
python tools/convert_yolo_to_coco.py --images E:/datasets/LEVIR-Ship/val/images   --labels E:/datasets/LEVIR-Ship/val/labels   --output E:/datasets/LEVIR-Ship/annotations/val.json   --category-name ship
python tools/convert_yolo_to_coco.py --images E:/datasets/LEVIR-Ship/test/images  --labels E:/datasets/LEVIR-Ship/test/labels  --output E:/datasets/LEVIR-Ship/annotations/test.json  --category-name ship
```

## 5. 冒烟运行（1 epoch）

### 5.1 DRENet 冒烟
```powershell
cd E:\work
git clone https://github.com/WindVChen/DRENet.git
cd DRENet
git checkout a187dbe0f623b521a62c6176c7cafaa7322f5f66
pip install -r requirements.txt

# 生成退化图（DRENet 训练必需）
cd E:\datasets\LEVIR-Ship
python E:\work\DRENet\DegradeGenerate.py

# 冒烟训练
cd E:\work\DRENet
python train.py --cfg .\models\DRENet.yaml --epochs 1 --workers 2 --batch-size 2 --device 0 --project .\runs\drenet_smoke --data .\data\ship.yaml
```

### 5.2 MMDet(FCOS) 冒烟
```powershell
cd E:\work
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout cfd5d3a985b0249de009b67d04f37263e11cdf3d

python tools/train.py configs\fcos\fcos_r50-caffe_fpn_gn-head_1x_coco.py `
  --work-dir .\work_dirs\fcos_smoke `
  --cfg-options `
  train_cfg.max_epochs=1 `
  model.bbox_head.num_classes=1 `
  train_dataloader.dataset.data_root=E:/datasets/LEVIR-Ship/ `
  train_dataloader.dataset.ann_file=annotations/train.json `
  train_dataloader.dataset.data_prefix.img=train/images/ `
  val_dataloader.dataset.data_root=E:/datasets/LEVIR-Ship/ `
  val_dataloader.dataset.ann_file=annotations/val.json `
  val_dataloader.dataset.data_prefix.img=val/images/ `
  test_dataloader.dataset.data_root=E:/datasets/LEVIR-Ship/ `
  test_dataloader.dataset.ann_file=annotations/test.json `
  test_dataloader.dataset.data_prefix.img=test/images/
```

### 5.3 YOLOv8 冒烟
```powershell
cd E:\work
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
git checkout cac4699df04f2c06a4ef3872217eae561981349d
pip install -e .

yolo detect train model=yolov8n.pt data=E:/datasets/LEVIR-Ship/ship.yaml imgsz=640 epochs=1 batch=2 device=0 project=E:/work/runs name=yolo_smoke
```

## 6. 正式训练命令（3 模型）

### 6.1 DRENet 正式训练
```powershell
cd E:\work\DRENet
python train.py --cfg .\models\DRENet.yaml --epochs 300 --workers 4 --batch-size 4 --device 0 --project .\runs\drenet_main --data .\data\ship.yaml
```

### 6.2 MMDet(FCOS) 正式训练
```powershell
cd E:\work\mmdetection
python tools/train.py configs\fcos\fcos_r50-caffe_fpn_gn-head_1x_coco.py `
  --work-dir .\work_dirs\fcos_main `
  --cfg-options `
  train_cfg.max_epochs=24 `
  model.bbox_head.num_classes=1 `
  train_dataloader.dataset.data_root=E:/datasets/LEVIR-Ship/ `
  train_dataloader.dataset.ann_file=annotations/train.json `
  train_dataloader.dataset.data_prefix.img=train/images/ `
  val_dataloader.dataset.data_root=E:/datasets/LEVIR-Ship/ `
  val_dataloader.dataset.ann_file=annotations/val.json `
  val_dataloader.dataset.data_prefix.img=val/images/ `
  test_dataloader.dataset.data_root=E:/datasets/LEVIR-Ship/ `
  test_dataloader.dataset.ann_file=annotations/test.json `
  test_dataloader.dataset.data_prefix.img=test/images/
```

### 6.3 YOLOv8 正式训练
```powershell
cd E:\work\ultralytics
yolo detect train model=yolov8n.pt data=E:/datasets/LEVIR-Ship/ship.yaml imgsz=640 epochs=100 batch=4 device=0 project=E:/work/runs name=yolo_main
```

## 7. 评测命令

### 7.1 DRENet 评测与推理
```powershell
cd E:\work\DRENet
python test.py --weights .\runs\drenet_main\exp\weights\best.pt --project .\runs\drenet_eval --device 0 --batch-size 4 --data .\data\ship.yaml
python detect.py --weights .\runs\drenet_main\exp\weights\best.pt --source E:\datasets\LEVIR-Ship\test\images --device 0
```

### 7.2 MMDet 评测
```powershell
cd E:\work\mmdetection
python tools/test.py configs\fcos\fcos_r50-caffe_fpn_gn-head_1x_coco.py .\work_dirs\fcos_main\epoch_24.pth --work-dir .\work_dirs\fcos_eval
```

### 7.3 YOLO 评测
```powershell
cd E:\work\ultralytics
yolo detect val model=E:/work/runs/yolo_main/weights/best.pt data=E:/datasets/LEVIR-Ship/ship.yaml imgsz=640 batch=4 device=0
```

## 8. 导出融合可视化（项目统一入口）
```powershell
cd E:\work\graduation_project
$env:PYTHONPATH='.'
python tools/visualize_predict.py --config configs/models.yaml --image E:/datasets/LEVIR-Ship/test/images/xxx.png --mode ensemble --models drenet,mmdet_fcos,yolo --vis-out outputs/visualizations/vis_result.jpg --json-out outputs/predictions/pred_result.json
```

## 9. checkpoint 续训命令
- DRENet（示例）：
```powershell
python train.py --resume .\runs\drenet_main\exp\weights\last.pt
```
- MMDet（示例）：
```powershell
python tools/train.py <config.py> --resume .\work_dirs\fcos_main\epoch_12.pth
```
- YOLO（示例）：
```powershell
yolo detect train resume model=E:/work/runs/yolo_main/weights/last.pt
```

## 10. 回传命令与回填
在本机项目目录执行：
```bash
bash scripts/sync_results_from_laptop.sh
```
回填顺序固定：
1. `docs/results/baselines.md`
2. `docs/results/qualitative.md`
3. `docs/results/ablation.md`

## 11. 关机前检查清单
- [ ] 本轮权重已落盘（best/last）
- [ ] 指标文件已导出（AP50/P/R/F1）
- [ ] 可视化图已导出
- [ ] 实验日志已更新
- [ ] 回传已完成或已打包待回传
- [ ] `configs/models.yaml` 已更新最新权重路径
