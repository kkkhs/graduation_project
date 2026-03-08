# 云端 GPU 实验执行手册（Linux + Bash + 固定 commit）

> 适用系统：Linux 云主机 / 云 GPU 容器  
> 默认路径前缀：
> - 项目：`/workspace/graduation_project`
> - 数据：`/data/LEVIR-Ship`
> - 训练环境：`conda activate shipdet`

> 使用原则：
> 1. 实验阶段与 3060 手册完全一致：环境预检 -> 数据预检 -> 冒烟训练 -> 正式训练 -> 评测 -> 回传。
> 2. 主要差异只有 Shell、路径、依赖安装方式和数据挂载方式。
> 3. 优先先跑 DRENet 首轮，再扩展 MMDet 和 YOLO。

## 0. 从零准备

### 0.1 环境预检
先确认云端机器可用：

```bash
nvidia-smi
python3 --version
git --version
conda --version
df -h
```

若以上任一命令失败，先修环境，不要继续执行。

创建并验证训练环境：

```bash
conda create -n shipdet python=3.10 -y
conda activate shipdet

python3 - <<'PY'
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
- 能识别当前云端 GPU

### 0.2 工作目录与数据目录
```bash
mkdir -p /workspace
mkdir -p /data
```

建议保留至少：
- 数据盘剩余空间 > 20GB
- 项目盘剩余空间 > 20GB

### 0.3 拉项目与数据
如果云端可以直接联网：

```bash
cd /workspace
git clone <repo_url> graduation_project

cd /data
git clone https://github.com/WindVChen/LEVIR-Ship.git
mv LEVIR-Ship /data/LEVIR-Ship
```

如果数据已通过对象存储或平台挂载，跳过下载，只保留目录核查。

### 0.4 数据目录预检
```bash
ls -la /data/LEVIR-Ship
ls -la /data/LEVIR-Ship/train
ls -la /data/LEVIR-Ship/val
ls -la /data/LEVIR-Ship/test
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
```bash
cd /workspace/graduation_project
git pull
```

## 2. 依赖安装（按框架拆分）
```bash
conda activate shipdet
python3 -m pip install -U pip
python3 -m pip install pyyaml pillow matplotlib numpy scipy pandas tqdm opencv-python
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.1 YOLO 依赖
```bash
python3 -m pip install ultralytics
```

### 2.2 MMDet 依赖
```bash
python3 -m pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

## 3. 数据配置（DRENet / MMDet / YOLO 共用）

### 3.1 DRENet 代码与 `ship.yaml`
```bash
cd /workspace
git clone https://github.com/WindVChen/DRENet.git
cd DRENet
git checkout a187dbe0f623b521a62c6176c7cafaa7322f5f66
python3 -m pip install -r requirements.txt
```

编辑 `/workspace/DRENet/data/ship.yaml`，至少确认以下字段：

```yaml
train: /data/LEVIR-Ship/train/images
val: /data/LEVIR-Ship/val/images
test: /data/LEVIR-Ship/test/images
nc: 1
names: ['ship']
```

要求：
- `train/val/test` 都指向 `images` 目录
- 类别固定单类：`ship`
- YOLO 与 DRENet 保持同一数据语义

### 3.2 生成 DRENet 所需退化图
```bash
cd /data/LEVIR-Ship
python3 /workspace/DRENet/DegradeGenerate.py
```

生成后检查：

```bash
ls -la /data/LEVIR-Ship/train/degrade | head
find /data/LEVIR-Ship/train/images -type f | wc -l
find /data/LEVIR-Ship/train/degrade -type f | wc -l
```

通过标准：
- `train/degrade` 目录存在
- `degrade` 数量接近 `train/images`
- 文件名能与原图对应

### 3.3 生成 MMDet 所需 COCO 标注
```bash
cd /workspace/graduation_project

python3 tools/convert_yolo_to_coco.py --images /data/LEVIR-Ship/train/images --labels /data/LEVIR-Ship/train/labels --output /data/LEVIR-Ship/annotations/train.json --category-name ship
python3 tools/convert_yolo_to_coco.py --images /data/LEVIR-Ship/val/images   --labels /data/LEVIR-Ship/val/labels   --output /data/LEVIR-Ship/annotations/val.json   --category-name ship
python3 tools/convert_yolo_to_coco.py --images /data/LEVIR-Ship/test/images  --labels /data/LEVIR-Ship/test/labels  --output /data/LEVIR-Ship/annotations/test.json  --category-name ship
```

### 3.4 数据转换验收
```bash
ls -lh /data/LEVIR-Ship/annotations/train.json
ls -lh /data/LEVIR-Ship/annotations/val.json
ls -lh /data/LEVIR-Ship/annotations/test.json

python3 - <<'PY'
import json
for split in ["train", "val", "test"]:
    path = f"/data/LEVIR-Ship/annotations/{split}.json"
    data = json.load(open(path, "r", encoding="utf-8"))
    print(split, "images=", len(data["images"]), "annotations=", len(data["annotations"]), "categories=", data["categories"])
PY
```

Checkpoint 2：
- DRENet 的 `degrade` 已生成
- MMDet 的 3 份 COCO 标注已生成

## 4. 冒烟运行（1 epoch）

### 4.1 DRENet 冒烟
```bash
cd /workspace/DRENet
python3 train.py --cfg ./models/DRENet.yaml --epochs 1 --workers 2 --batch-size 2 --device 0 --project ./runs/drenet_smoke --data ./data/ship.yaml
```

### 4.2 MMDet(FCOS) 冒烟
```bash
cd /workspace
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout cfd5d3a985b0249de009b67d04f37263e11cdf3d

python3 tools/train.py configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py \
  --work-dir ./work_dirs/fcos_smoke \
  --cfg-options \
  train_cfg.max_epochs=1 \
  model.bbox_head.num_classes=1 \
  train_dataloader.dataset.data_root=/data/LEVIR-Ship/ \
  train_dataloader.dataset.ann_file=annotations/train.json \
  train_dataloader.dataset.data_prefix.img=train/images/ \
  val_dataloader.dataset.data_root=/data/LEVIR-Ship/ \
  val_dataloader.dataset.ann_file=annotations/val.json \
  val_dataloader.dataset.data_prefix.img=val/images/ \
  test_dataloader.dataset.data_root=/data/LEVIR-Ship/ \
  test_dataloader.dataset.ann_file=annotations/test.json \
  test_dataloader.dataset.data_prefix.img=test/images/
```

### 4.3 YOLO26 冒烟
```bash
cd /workspace
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
git checkout cac4699df04f2c06a4ef3872217eae561981349d
python3 -m pip install -e .

yolo detect train model=yolo26n.pt data=/data/LEVIR-Ship/ship.yaml imgsz=640 epochs=1 batch=2 device=0 project=/workspace/runs name=yolo_smoke
```
说明：若当前 Ultralytics 版本暂不支持 `yolo26n.pt`，先升级 Ultralytics；必要时可临时改用 `yolo11n.pt` 做冒烟。

### 4.4 冒烟通过判定
分别检查：
- DRENet：`/workspace/DRENet/runs/drenet_smoke/`
- MMDet：`/workspace/mmdetection/work_dirs/fcos_smoke/`
- YOLO：`/workspace/runs/yolo_smoke/`

Checkpoint 3：
- 至少一个模型冒烟成功
- 建议先确认 DRENet 成功，再进入正式训练

## 5. 正式训练命令（先 DRENet，后 MMDet / YOLO）

### 5.1 DRENet 正式训练
```bash
cd /workspace/DRENet
python3 train.py --cfg ./models/DRENet.yaml --epochs 300 --workers 4 --batch-size 4 --device 0 --project ./runs/drenet_main --data ./data/ship.yaml
```

### 5.2 MMDet(FCOS) 正式训练
```bash
cd /workspace/mmdetection
python3 tools/train.py configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py \
  --work-dir ./work_dirs/fcos_main \
  --cfg-options \
  train_cfg.max_epochs=24 \
  model.bbox_head.num_classes=1 \
  train_dataloader.dataset.data_root=/data/LEVIR-Ship/ \
  train_dataloader.dataset.ann_file=annotations/train.json \
  train_dataloader.dataset.data_prefix.img=train/images/ \
  val_dataloader.dataset.data_root=/data/LEVIR-Ship/ \
  val_dataloader.dataset.ann_file=annotations/val.json \
  val_dataloader.dataset.data_prefix.img=val/images/ \
  test_dataloader.dataset.data_root=/data/LEVIR-Ship/ \
  test_dataloader.dataset.ann_file=annotations/test.json \
  test_dataloader.dataset.data_prefix.img=test/images/
```

### 5.3 YOLO26 正式训练
```bash
cd /workspace/ultralytics
yolo detect train model=yolo26n.pt data=/data/LEVIR-Ship/ship.yaml imgsz=640 epochs=100 batch=12 device=0 project=/workspace/runs name=yolo_main
```

### 5.4 正式训练中的失败分支
- OOM：按 `batch-size -> workers -> input size` 顺序调整
- 路径错误：先查 `ship.yaml`，再查 `annotations/*.json`
- `torch.cuda.is_available() == False`：立即停止训练，回到环境预检阶段

Checkpoint 4：
- DRENet 正式训练已启动并能持续产出 checkpoint

## 6. 评测命令

### 6.1 DRENet 评测与推理
```bash
cd /workspace/DRENet
python3 test.py --weights ./runs/drenet_main/exp/weights/best.pt --project ./runs/drenet_eval --device 0 --batch-size 4 --data ./data/ship.yaml
python3 detect.py --weights ./runs/drenet_main/exp/weights/best.pt --source /data/LEVIR-Ship/test/images --device 0
```

### 6.2 MMDet 评测
```bash
cd /workspace/mmdetection
python3 tools/test.py configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py ./work_dirs/fcos_main/epoch_24.pth --work-dir ./work_dirs/fcos_eval
```

### 6.3 YOLO 评测
```bash
cd /workspace/ultralytics
yolo detect val model=/workspace/runs/yolo_main/weights/best.pt data=/data/LEVIR-Ship/ship.yaml imgsz=640 batch=4 device=0
```

### 6.4 指标提取与保存
保存规则与 3060 手册一致：
- `artifacts/drenet/{weights,logs,metrics,figures}`
- `artifacts/mmdet/{weights,logs,metrics,figures}`
- `artifacts/yolo/{weights,logs,metrics,figures}`

指标文件最少包含：
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

Checkpoint 5：
- 评测成功
- 指标可回填
- 图像可用于定性分析

### 6.5 FPS / FLOPs / Params 测量规范（统一口径）
- 执行时机：
  - 建议在训练结束后，使用最终权重（通常 `best`）测量并回填。
- 固定条件（必须一致）：
  - `imgsz=512`
  - `batch=1`
  - 同一 GPU
  - 同一精度模式（FP32 或 FP16，需在日志中注明）
  - 同一后处理阈值（conf/iou）
- FPS 测量：
  - 先 warmup（建议 50 次），再计时（建议 200 次），取平均单张耗时；
  - `FPS = 1 / 平均单张耗时(秒)`。
- Params 测量：
  - 统计模型参数总量，单位 `M`。
- FLOPs 测量：
  - 统一在输入尺寸 `512x512` 下测一次前向的 `GFLOPs`。
- 结果回填：
  - 写入 `docs/results/baselines.md` 的 `FPS`、`Params(M)`；
  - 若某模型 FLOPs 不便直接写入主表，可在实验日志备注中补充。

## 7. 续训命令
- DRENet：
```bash
python3 train.py --resume ./runs/drenet_main/exp/weights/last.pt
```
- MMDet：
```bash
python3 tools/train.py <config.py> --resume ./work_dirs/fcos_main/epoch_12.pth
```
- YOLO：
```bash
yolo detect train resume model=/workspace/runs/yolo_main/weights/last.pt
```

### 7.1 W&B 续训防重复（强烈建议）
若同一实验会多次重启，为避免出现“同名多个 run”，续训时固定：

```bash
export WANDB_RUN_ID=<existing_run_id>
export WANDB_RESUME=must
```

然后再执行 `resume` 命令。  
说明：W&B 唯一键是 `run_id`，不是 run `name`。只改名字不能防重复。

## 8. 回传与回填
如果云端可以直接访问本机：

```bash
cd /workspace/graduation_project
bash scripts/sync_results_from_laptop.sh <local_user@local_host> /workspace/graduation_project
```

如果不能直接回传，至少手动导出这些目录：
- `docs/experiments/logs/`
- `docs/results/`
- `assets/figures/`
- `artifacts/`
- `runs/`
- `work_dirs/`

Checkpoint 6：
- 本机已收到日志、图、指标和结果文档

### 8.1 本地目录建议（实跑经验）
- 建议本地统一落到：
  - `experiment_assets/runs/<run_name>`
  - `experiment_assets/runs/trace/{drenet,yolo,sync,...}`
- 输入配置（data yaml）建议单独快照：
  - `experiment_assets/configs/<run_name>/ship_autodl.yaml`
- 要求：
  - 正式训练启动前先复制一份原始 `data yaml`
  - 若训练后才补录，必须在日志中注明“根据运行日志恢复”
- 若历史目录为 `runs/detect/runs/<run_name>`，可迁移到 `runs/` 根下，并保留兼容软链接。

## 9. 关机前检查清单
- [ ] 本轮权重已落盘（best/last）
- [ ] 指标文件已导出（AP50/P/R/F1）
- [ ] 可视化图已导出
- [ ] 实验日志已更新
- [ ] 回传已完成或已打包待回传
- [ ] `configs/models.yaml` 已更新最新权重路径

## 10. 结论
云端 GPU 与 3060 本地训练机的实验步骤完全一致，差别只在：
- Shell：`bash` 替代 `PowerShell`
- 路径：Linux 路径替代 `E:/...`
- 数据来源：挂载盘 / 对象存储替代本地磁盘
- 回传方式：`rsync` / `scp` / 平台下载替代局域网同步

实验顺序、停点验收、指标口径和产物规范不变。
