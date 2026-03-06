# 3060 训练机可复制执行手册（PowerShell + 固定 commit）

> 适用系统：Windows 原生 PowerShell  
> 默认路径前缀：
> - 项目：`E:/work/graduation_project`
> - 数据：`E:/datasets/LEVIR-Ship`
> - 训练环境：`conda activate shipdet`

> 使用原则：
> 1. 先做环境预检，再拉代码。
> 2. 先做数据预检，再训练。
> 3. 先跑 DRENet 首轮，再扩展 MMDet 和 YOLO。
> 4. 每通过一个 checkpoint 再进入下一阶段。

## 0. 从零准备

### 0.1 环境预检
先确认训练机具备最小训练条件：

```powershell
nvidia-smi
python --version
git --version
conda --version
```

若以上任一命令失败，先修环境，不要继续执行后续步骤。

创建并验证训练环境：

```powershell
conda create -n shipdet python=3.10 -y
conda activate shipdet

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

通过标准：
- `torch` 可导入
- `torch.cuda.is_available()` 为 `True`
- 能识别 RTX 3060

### 0.2 工作目录与磁盘预检
```powershell
New-Item -ItemType Directory -Force E:\work | Out-Null
New-Item -ItemType Directory -Force E:\datasets | Out-Null
Get-PSDrive -Name E
```

建议保留至少：
- 数据盘剩余空间 > 20GB
- 项目盘剩余空间 > 20GB

### 0.3 拉项目与数据
```powershell
cd E:\work
git clone <repo_url> graduation_project

cd E:\datasets
git clone https://github.com/WindVChen/LEVIR-Ship.git
```

### 0.4 数据目录预检
```powershell
Get-ChildItem E:\datasets\LEVIR-Ship
Get-ChildItem E:\datasets\LEVIR-Ship\train
Get-ChildItem E:\datasets\LEVIR-Ship\val
Get-ChildItem E:\datasets\LEVIR-Ship\test
```

必须能看到这些目录：
- `train/images`
- `train/labels`
- `val/images`
- `val/labels`
- `test/images`
- `test/labels`

Checkpoint 1：
- 环境通过
- 数据根目录通过

## 1. 项目同步
```powershell
cd E:\work\graduation_project
git pull
```

## 2. 依赖安装（按框架拆分）
```powershell
conda activate shipdet
pip install -U pip
pip install pyyaml pillow matplotlib numpy scipy pandas tqdm opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.1 YOLO 依赖
```powershell
pip install ultralytics
```

### 2.2 MMDet 依赖
```powershell
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

## 3. 数据配置（DRENet / MMDet / YOLO 共用）

### 3.1 DRENet 代码与 `ship.yaml`
```powershell
cd E:\work
git clone https://github.com/WindVChen/DRENet.git
cd DRENet
git checkout a187dbe0f623b521a62c6176c7cafaa7322f5f66
pip install -r requirements.txt
```

编辑 `E:\work\DRENet\data\ship.yaml`，至少确认以下字段：

```yaml
train: E:/datasets/LEVIR-Ship/train/images
val: E:/datasets/LEVIR-Ship/val/images
test: E:/datasets/LEVIR-Ship/test/images
nc: 1
names: ['ship']
```

要求：
- `train/val/test` 都指向 `images` 目录
- 类别固定单类：`ship`
- YOLO 也应共用同一份数据语义，即单类检测

### 3.2 生成 DRENet 所需退化图
```powershell
cd E:\datasets\LEVIR-Ship
python E:\work\DRENet\DegradeGenerate.py
```

生成后检查：

```powershell
Get-ChildItem E:\datasets\LEVIR-Ship\train\degrade | Select-Object -First 5
(Get-ChildItem E:\datasets\LEVIR-Ship\train\images).Count
(Get-ChildItem E:\datasets\LEVIR-Ship\train\degrade).Count
```

通过标准：
- `train/degrade` 目录存在
- `degrade` 数量接近 `train/images`
- 文件名能与原图对应

### 3.3 生成 MMDet 所需 COCO 标注
```powershell
cd E:\work\graduation_project

python tools/convert_yolo_to_coco.py --images E:/datasets/LEVIR-Ship/train/images --labels E:/datasets/LEVIR-Ship/train/labels --output E:/datasets/LEVIR-Ship/annotations/train.json --category-name ship
python tools/convert_yolo_to_coco.py --images E:/datasets/LEVIR-Ship/val/images   --labels E:/datasets/LEVIR-Ship/val/labels   --output E:/datasets/LEVIR-Ship/annotations/val.json   --category-name ship
python tools/convert_yolo_to_coco.py --images E:/datasets/LEVIR-Ship/test/images  --labels E:/datasets/LEVIR-Ship/test/labels  --output E:/datasets/LEVIR-Ship/annotations/test.json  --category-name ship
```

### 3.4 数据转换验收
```powershell
Get-Item E:\datasets\LEVIR-Ship\annotations\train.json
Get-Item E:\datasets\LEVIR-Ship\annotations\val.json
Get-Item E:\datasets\LEVIR-Ship\annotations\test.json

python - <<'PY'
import json
for split in ["train", "val", "test"]:
    path = fr"E:\datasets\LEVIR-Ship\annotations\{split}.json"
    data = json.load(open(path, "r", encoding="utf-8"))
    print(split, "images=", len(data["images"]), "annotations=", len(data["annotations"]), "categories=", data["categories"])
PY
```

通过标准：
- `train/val/test.json` 全部存在且可读取
- `images` 与 `annotations` 非空
- `categories` 中类别名为 `ship`

Checkpoint 2：
- DRENet 的 `degrade` 已生成
- MMDet 的 3 份 COCO 标注已生成

## 4. 冒烟运行（1 epoch）

### 4.1 DRENet 冒烟
```powershell
cd E:\work\DRENet
python train.py --cfg .\models\DRENet.yaml --epochs 1 --workers 2 --batch-size 2 --device 0 --project .\runs\drenet_smoke --data .\data\ship.yaml
```

### 4.2 MMDet(FCOS) 冒烟
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

### 4.3 YOLOv8 冒烟
```powershell
cd E:\work
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
git checkout cac4699df04f2c06a4ef3872217eae561981349d
pip install -e .

yolo detect train model=yolov8n.pt data=E:/datasets/LEVIR-Ship/ship.yaml imgsz=640 epochs=1 batch=2 device=0 project=E:/work/runs name=yolo_smoke
```

### 4.4 冒烟通过判定
分别检查：
- DRENet：`E:\work\DRENet\runs\drenet_smoke\`
- MMDet：`E:\work\mmdetection\work_dirs\fcos_smoke\`
- YOLO：`E:\work\runs\yolo_smoke\`

通过标准：
- 训练能完整跑完 1 epoch
- 至少生成日志与权重文件
- 没有路径错误、数据错误、显存错误

Checkpoint 3：
- 三模型至少一个已冒烟成功
- 建议先确认 DRENet 成功，再进入正式训练

## 5. 正式训练命令（先 DRENet，后 MMDet / YOLO）

### 5.1 DRENet 正式训练
```powershell
cd E:\work\DRENet
python train.py --cfg .\models\DRENet.yaml --epochs 300 --workers 4 --batch-size 4 --device 0 --project .\runs\drenet_main --data .\data\ship.yaml
```

### 5.2 MMDet(FCOS) 正式训练
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

### 5.3 YOLOv8 正式训练
```powershell
cd E:\work\ultralytics
yolo detect train model=yolov8n.pt data=E:/datasets/LEVIR-Ship/ship.yaml imgsz=640 epochs=100 batch=4 device=0 project=E:/work/runs name=yolo_main
```

### 5.4 正式训练中的失败分支
- OOM：按固定顺序调整
  - `batch-size 4 -> 2`
  - `workers 4 -> 2`
  - 必要时再降输入尺寸
- 路径错误：
  - 先查 `ship.yaml`
  - 再查 `annotations/*.json`
  - 最后查权重或配置路径
- `torch.cuda.is_available() == False`：
  - 立即停止训练，回到环境预检阶段

Checkpoint 4：
- DRENet 正式训练已启动并能持续产出 checkpoint

## 6. 评测命令

### 6.1 DRENet 评测与推理
```powershell
cd E:\work\DRENet
python test.py --weights .\runs\drenet_main\exp\weights\best.pt --project .\runs\drenet_eval --device 0 --batch-size 4 --data .\data\ship.yaml
python detect.py --weights .\runs\drenet_main\exp\weights\best.pt --source E:\datasets\LEVIR-Ship\test\images --device 0
```

### 6.2 MMDet 评测
```powershell
cd E:\work\mmdetection
python tools/test.py configs\fcos\fcos_r50-caffe_fpn_gn-head_1x_coco.py .\work_dirs\fcos_main\epoch_24.pth --work-dir .\work_dirs\fcos_eval
```

### 6.3 YOLO 评测
```powershell
cd E:\work\ultralytics
yolo detect val model=E:/work/runs/yolo_main/weights/best.pt data=E:/datasets/LEVIR-Ship/ship.yaml imgsz=640 batch=4 device=0
```

### 6.4 指标提取与保存
每个模型至少保存以下字段：
- `experiment_id`
- `model_name`
- `dataset`
- `input_size`
- `ap50`
- `precision`
- `recall`
- `f1`
- `fps`
- `weight_path`

建议目录：
```text
artifacts/
  drenet/{weights,logs,metrics,figures}
  mmdet/{weights,logs,metrics,figures}
  yolo/{weights,logs,metrics,figures}
```

建议命名：
- 权重：`{model}_{dataset}_{input}_{metric}-{value}_{date}.{ext}`
- 指标：`metrics_{model}_{exp_id}.json`
- 图像：`{model}_{exp_id}_{case_type}_{image_id}.jpg`

对 DRENet 首轮，最小要求：
- 至少拿到 `AP50 / Precision / Recall / F1`
- 至少整理 `success / miss / false_positive` 三类图各 1 张

Checkpoint 5：
- 评测成功
- 指标可回填
- 图像可用于定性分析

## 7. 导出融合可视化（项目统一入口）
```powershell
cd E:\work\graduation_project
$env:PYTHONPATH='.'
python tools/visualize_predict.py --config configs/models.yaml --image E:/datasets/LEVIR-Ship/test/images/xxx.png --mode ensemble --models drenet,mmdet_fcos,yolo --vis-out outputs/visualizations/vis_result.jpg --json-out outputs/predictions/pred_result.json
```

## 8. checkpoint 续训命令
- DRENet：
```powershell
python train.py --resume .\runs\drenet_main\exp\weights\last.pt
```
- MMDet：
```powershell
python tools/train.py <config.py> --resume .\work_dirs\fcos_main\epoch_12.pth
```
- YOLO：
```powershell
yolo detect train resume model=E:/work/runs/yolo_main/weights/last.pt
```

## 9. 回传命令与回填
在本机项目目录执行：
```bash
bash scripts/sync_results_from_laptop.sh <user@host> <remote_project_path>
```

回填顺序固定：
1. `docs/results/baselines.md`
2. `docs/results/qualitative.md`
3. `docs/results/ablation.md`

### 9.1 回传前文件清单
回传前至少确认这些内容存在：
- `docs/experiments/logs/`
- `docs/results/`
- `assets/figures/`
- `artifacts/drenet/weights/`
- `artifacts/drenet/metrics/`
- `artifacts/drenet/figures/`
- 其他模型的 `artifacts/*/`

Checkpoint 6：
- 回传成功
- 本机已收到日志、图、指标和结果文档

## 10. 关机前检查清单
- [ ] 本轮权重已落盘（best/last）
- [ ] 指标文件已导出（AP50/P/R/F1）
- [ ] 可视化图已导出
- [ ] 实验日志已更新
- [ ] 回传已完成或已打包待回传
- [ ] `configs/models.yaml` 已更新最新权重路径

## 11. 故障分流表

### 11.1 环境失败
- `nvidia-smi` 不可用：先修驱动
- `torch.cuda.is_available() == False`：先修 Torch/CUDA 版本匹配

### 11.2 数据失败
- `train/degrade` 缺失：重新执行 `DegradeGenerate.py`
- `annotations/*.json` 缺失或为空：重新执行 `convert_yolo_to_coco.py`

### 11.3 训练失败
- OOM：先减 `batch-size`，再减 `workers`，最后再减输入尺寸
- 路径错误：先查 `ship.yaml`，再查 COCO 标注路径

### 11.4 评测失败
- 没有 `best.pt` / `best.pth`：先确认训练是否完整结束
- 指标未输出：先保留终端原始输出，再整理成 `metrics_{model}_{exp_id}.json`

Checkpoint 7：
- 文档回填成功
- DRENet 首轮结果已进入仓库
- 后续可继续 MMDet / YOLO 正式实验
