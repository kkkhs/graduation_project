# DRENet 复现指南

> 本文档提供 DRENet 论文的详细复现步骤，确保实验可复现。

---

## 一、复现目标

### 1.1 主要目标

- 在 LEVIR-Ship 数据集上复现 DRENet 的训练、评测和推理流程
- 达到论文报告的性能指标：AP 82.4，FPS 85，Params 4.79M
- 输出完整的实验记录和可视化结果

### 1.2 预期成果

- 训练好的模型权重
- 测试集评测结果
- 可视化样例（成功/失败/难例）
- 完整的实验记录（使用 `exp_log_template.md`）

---

## 二、准备工作

### 2.1 硬件要求

**推荐配置**：
- GPU：NVIDIA GPU（显存 ≥ 8GB）
- CPU：多核 CPU（≥ 8 核）
- 内存：≥ 16GB
- 硬盘：≥ 50GB 可用空间

**最低配置**：
- GPU：NVIDIA GPU（显存 ≥ 4GB）
- CPU：≥ 4 核
- 内存：≥ 8GB
- 硬盘：≥ 20GB 可用空间

### 2.2 软件环境

**操作系统**：
- Linux（推荐 Ubuntu 18.04+）
- Windows 10/11（需要 WSL2 或适配）
- macOS（需要适配）

**软件依赖**：
- Python 3.8+
- CUDA 11.0+（对应 PyTorch 版本）
- Git

### 2.3 目录结构

```
graduation_project/
├── data/
│   └── LEVIR-Ship/
│       ├── train/
│       ├── val/
│       └── test/
├── experiments/
│   └── drenet/
│       ├── configs/
│       ├── checkpoints/
│       ├── logs/
│       └── results/
├── scripts/
│   ├── prepare_data.py
│   ├── train_drenet.py
│   ├── eval_drenet.py
│   └── infer_drenet.py
└── docs/
    └── experiments/
        └── drenet_exp_log.md
```

---

## 三、详细步骤

### 步骤 1：环境配置（预计 30 分钟）

#### 1.1 创建虚拟环境

```bash
# 使用 conda
conda create -n drenet python=3.8 -y
conda activate drenet

# 或使用 venv
python -m venv drenet_env
source drenet_env/bin/activate  # Linux/mac
# 或
drenet_env\Scripts\activate  # Windows
```

#### 1.2 安装 PyTorch

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### 1.3 安装其他依赖

```bash
# 基础依赖
pip install numpy opencv-python pillow matplotlib tqdm

# YOLOv5 依赖
pip install pyyaml requests scipy pandas seaborn

# 可选：tensorboard 用于可视化训练过程
pip install tensorboard
```

#### 1.4 验证环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### 步骤 2：获取代码（预计 10 分钟）

#### 2.1 克隆 DRENet 仓库

```bash
cd experiments
git clone https://github.com/WindVChen/DRENet.git
cd DRENet

# 或下载 ZIP
# 访问 https://github.com/WindVChen/DRENet
# 下载并解压到 experiments/drenet/
```

#### 2.2 查看仓库结构

```bash
ls -la
# 应该看到：
# models/  - 网络定义
# data/    - 数据加载
# utils/   - 工具函数
# train.py - 训练脚本
# detect.py - 推理脚本
# requirements.txt - 依赖列表
```

#### 2.3 安装仓库依赖

```bash
# 如果有 requirements.txt
pip install -r requirements.txt

# 否则手动安装
pip install -r requirements.txt
```

---

### 步骤 3：准备数据集（预计 1-2 小时）

#### 3.1 下载 LEVIR-Ship 数据集

**方法 1：GitHub 下载**

```bash
cd data
git clone https://github.com/WindVChen/LEVIR-Ship.git
```

**方法 2：手动下载**

1. 访问 https://github.com/WindVChen/LEVIR-Ship
2. 下载 ZIP 文件
3. 解压到 `data/LEVIR-Ship/`

#### 3.2 验证数据集结构

```bash
cd data/LEVIR-Ship
ls -la

# 应该看到：
# train/  - 训练集
# val/    - 验证集
# test/   - 测试集
# 或其他官方划分方式
```

#### 3.3 检查标注格式

```bash
# 查看标注文件格式
ls train/*.txt  # 或其他标注格式

# LEVIR-Ship 可能使用：
# - YOLO 格式：每个图像对应一个 .txt 文件
# - COCO 格式：一个 JSON 文件包含所有标注
# - VOC 格式：每个图像对应一个 XML 文件
```

#### 3.4 统计数据集信息

```bash
# 统计图像数量
find train -name "*.jpg" | wc -l
find val -name "*.jpg" | wc -l
find test -name "*.jpg" | wc -l

# 统计目标数量
grep -r " " train/*.txt | wc -l
```

**预期结果**：
- 训练集：2320 张图像，2002 个目标
- 验证集：788 张图像，665 个目标
- 测试集：788 张图像，552 个目标

---

### 步骤 4：配置数据路径（预计 10 分钟）

#### 4.1 修改配置文件

找到 DRENet 的配置文件（通常是 `data/xxx.yaml` 或 `configs/xxx.yaml`）：

```yaml
# 示例：data/levir_ship.yaml
path: ../../data/LEVIR-Ship  # 数据集根目录
train: train  # 训练集相对路径
val: val      # 验证集相对路径
test: test    # 测试集相对路径

nc: 1  # 类别数量
names: ['ship']  # 类别名称
```

#### 4.2 创建数据集软链接（可选）

```bash
# 在 DRENet 目录下创建软链接
cd experiments/drenet/DRENet
ln -s ../../data/LEVIR-Ship data/LEVIR-Ship
```

---

### 步骤 5：准备训练配置（预计 30 分钟）

#### 5.1 查看默认训练配置

```bash
# 查看训练脚本
cat train.py

# 或查看配置文件
cat models/yolov5s.yaml
```

#### 5.2 关键训练参数

根据论文，DRENet 的训练参数：

```yaml
# 模型配置
model: yolov5s  # 基于 YOLOv5s

# 训练配置
epochs: 500
batch_size: 16
img_size: 512  # LEVIR-Ship 图像尺寸

# 优化器
optimizer: SGD
lr: 0.01
momentum: 0.937
weight_decay: 0.0005

# 学习率调度
lr_scheduler: cosine
warmup_epochs: 3

# 数据增强
mosaic: 1.0  # Mosaic 增强
mixup: 0.0   # Mixup 增强
copy_paste: 0.0  # Copy-Paste 增强
flip: 0.5    # 水平翻转
scale: 0.5    # 缩放

# DRENet 特有配置
use_dre: True  # 启用退化重建增强器
use_crma: True  # 启用 CRMA 模块
```

#### 5.3 创建自定义配置文件

```bash
# 复制默认配置
cp data/coco.yaml data/levir_ship.yaml
cp models/yolov5s.yaml models/drenet.yaml

# 编辑配置文件
vim data/levir_ship.yaml
vim models/drenet.yaml
```

---

### 步骤 6：训练模型（预计 6-12 小时）

#### 6.1 开始训练

```bash
cd experiments/drenet/DRENet

# 基础训练命令
python train.py --data data/levir_ship.yaml \
              --cfg models/drenet.yaml \
              --weights yolov5s.pt \
              --epochs 500 \
              --batch-size 16 \
              --img 512 \
              --device 0 \
              --workers 8 \
              --project runs/train \
              --name drenet_levir \
              --exist-ok

# 使用多 GPU（如果有）
python train.py --data data/levir_ship.yaml \
              --cfg models/drenet.yaml \
              --weights yolov5s.pt \
              --epochs 500 \
              --batch-size 32 \
              --img 512 \
              --device 0,1 \
              --workers 8 \
              --project runs/train \
              --name drenet_levir \
              --exist-ok
```

#### 6.2 监控训练过程

**方法 1：TensorBoard**

```bash
# 启动 TensorBoard
tensorboard --logdir runs/train/drenet_levir

# 在浏览器中访问
# http://localhost:6006
```

**方法 2：查看日志**

```bash
# 实时查看训练日志
tail -f runs/train/drenet_levir/train.log

# 或查看结果 CSV
cat runs/train/drenet_levir/results.csv
```

#### 6.3 训练检查点

训练过程中会自动保存检查点到：
```
runs/train/drenet_levir/weights/
├── best.pt      # 最佳模型（根据验证集 mAP）
├── last.pt      # 最后一个 epoch 的模型
└── epoch_xxx.pt # 每 10 个 epoch 保存一次
```

#### 6.4 固定随机种子

为了确保可复现，在训练脚本中添加：

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

### 步骤 7：评测模型（预计 30 分钟）

#### 7.1 在验证集上评测

```bash
python val.py --data data/levir_ship.yaml \
             --weights runs/train/drenet_levir/weights/best.pt \
             --img 512 \
             --batch-size 16 \
             --device 0 \
             --task val \
             --project runs/val \
             --name drenet_levir \
             --exist-ok
```

#### 7.2 在测试集上评测

```bash
python val.py --data data/levir_ship.yaml \
             --weights runs/train/drenet_levir/weights/best.pt \
             --img 512 \
             --batch-size 16 \
             --device 0 \
             --task test \
             --project runs/test \
             --name drenet_levir \
             --exist-ok
```

#### 7.3 查看评测结果

```bash
# 查看评测结果
cat runs/val/drenet_levir/results.txt

# 或查看 TensorBoard
tensorboard --logdir runs/val/drenet_levir
```

**预期结果**：
- AP50：约 82.4（论文报告值）
- FPS：约 85（在 512×512 输入下）

---

### 步骤 8：推理可视化（预计 30 分钟）

#### 8.1 单张图像推理

```bash
python detect.py --weights runs/train/drenet_levir/weights/best.pt \
               --source data/LEVIR-Ship/test/xxx.jpg \
               --img 512 \
               --conf-thres 0.25 \
               --iou-thres 0.45 \
               --device 0 \
               --project runs/detect \
               --name single_test \
               --exist-ok
```

#### 8.2 批量推理

```bash
python detect.py --weights runs/train/drenet_levir/weights/best.pt \
               --source data/LEVIR-Ship/test/ \
               --img 512 \
               --conf-thres 0.25 \
               --iou-thres 0.45 \
               --device 0 \
               --project runs/detect \
               --name batch_test \
               --exist-ok
```

#### 8.3 查看可视化结果

```bash
# 结果保存在
ls runs/detect/batch_test/

# 查看可视化图像
open runs/detect/batch_test/xxx.jpg  # macOS
# 或
xdg-open runs/detect/batch_test/xxx.jpg  # Linux
```

#### 8.4 导出推理结果

```bash
# 导出为 JSON 格式
python detect.py --weights runs/train/drenet_levir/weights/best.pt \
               --source data/LEVIR-Ship/test/ \
               --img 512 \
               --conf-thres 0.25 \
               --iou-thres 0.45 \
               --device 0 \
               --project runs/detect \
               --name batch_test \
               --save-txt \
               --save-conf \
               --exist-ok
```

---

### 步骤 9：整理实验记录（预计 1 小时）

#### 9.1 创建实验记录

复制实验记录模板：

```bash
cp docs/experiments/exp_log_template.md docs/experiments/drenet_exp_log.md
```

#### 9.2 填写实验信息

```markdown
## 实验记录：DRENet 复现

### 1. 实验信息
- **实验编号**：exp-2026-02-07-01
- **日期**：2026-02-07
- **模型/框架**：DRENet / YOLOv5s
- **代码版本**：https://github.com/WindVChen/DRENet (commit: xxx)
- **数据版本**：LEVIR-Ship (官方 GitHub)
- **划分方式**：官方划分
- **随机种子**：42

### 2. 训练设置
- **输入尺寸**：512×512
- **batch size**：16
- **epoch/iters**：500
- **优化器**：SGD (lr=0.01, momentum=0.937, weight_decay=0.0005)
- **学习率策略**：cosine (warmup=3 epochs)
- **数据增强**：mosaic=1.0, flip=0.5, scale=0.5
- **损失/后处理关键设置**：NMS iou=0.45, conf=0.25

### 3. 资源与耗时
- **硬件**：NVIDIA RTX 3090 (24GB), Intel i9-10900K, 64GB RAM
- **训练耗时**：约 8 小时（500 epochs）
- **显存峰值**：约 6GB

### 4. 结果（定量）
- **AP50**：82.4
- **FPS**：85
- **Params**：4.79M
- **FLOPs**：8.3G
- **备注**：与论文报告值一致

### 5. 结果（定性）
- **可视化路径**：runs/detect/batch_test/
- **成功案例**：
  - 平静海面下的船舶检测准确
  - 薄云场景下的船舶检测良好
- **失败案例/难例**：
  - 厚云场景下的部分漏检
  - 碎云场景下的少量误检
  - 极小目标（<10像素）的漏检

### 6. 复现命令

```bash
# train
python train.py --data data/levir_ship.yaml \
              --cfg models/drenet.yaml \
              --weights yolov5s.pt \
              --epochs 500 \
              --batch-size 16 \
              --img 512 \
              --device 0 \
              --workers 8 \
              --project runs/train \
              --name drenet_levir \
              --exist-ok

# eval
python val.py --data data/levir_ship.yaml \
             --weights runs/train/drenet_levir/weights/best.pt \
             --img 512 \
             --batch-size 16 \
             --device 0 \
             --task test \
             --project runs/test \
             --name drenet_levir \
             --exist-ok

# infer/visualize
python detect.py --weights runs/train/drenet_levir/weights/best.pt \
               --source data/LEVIR-Ship/test/ \
               --img 512 \
               --conf-thres 0.25 \
               --iou-thres 0.45 \
               --device 0 \
               --project runs/detect \
               --name batch_test \
               --save-txt \
               --save-conf \
               --exist-ok
```
```

---

## 四、常见问题与解决方案

### 4.1 CUDA 内存不足

**问题**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
```bash
# 减小 batch size
python train.py --batch-size 8  # 从 16 改为 8

# 或减小输入尺寸
python train.py --img 416  # 从 512 改为 416

# 或使用梯度累积
python train.py --batch-size 4 --accumulate 4
```

### 4.2 数据集路径错误

**问题**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/LEVIR-Ship/train'
```

**解决方案**：
```bash
# 检查数据集路径
ls -la data/LEVIR-Ship/

# 修改配置文件中的路径
vim data/levir_ship.yaml

# 或创建软链接
ln -s ../../data/LEVIR-Ship data/LEVIR-Ship
```

### 4.3 训练不收敛

**问题**：
```
训练过程中 loss 不下降或震荡
```

**解决方案**：
```bash
# 检查学习率
python train.py --lr 0.001  # 降低学习率

# 检查数据增强
python train.py --mosaic 0.5  # 减少数据增强强度

# 检查数据标注
# 确保标注格式正确，没有错误标注
```

### 4.4 性能低于论文报告

**问题**：
```
AP50 只有 70-75，低于论文报告的 82.4
```

**解决方案**：
```bash
# 检查训练 epoch
# 论文训练了 500 epochs，确保训练充分

# 检查数据增强
# DRENet 使用了 Selective Degradation，确保启用

# 检查输入尺寸
# 论文使用 512×512，确保一致

# 检查评测设置
# 确保使用正确的 IoU 阈值（0.5）
```

---

## 五、下一步

完成 DRENet 复现后，可以：

1. **对比实验**：使用相同数据集训练其他模型（Faster R-CNN、YOLOv8）
2. **消融实验**：测试不同组件的影响（DRE、CRMA）
3. **可视化分析**：分析不同场景下的性能差异
4. **系统集成**：将 DRENet 集成到检测系统中

---

## 六、参考资源

- **DRENet 代码**：https://github.com/WindVChen/DRENet
- **LEVIR-Ship 数据集**：https://github.com/WindVChen/LEVIR-Ship
- **YOLOv5 官方文档**：https://github.com/ultralytics/yolov5
- **论文**：Chen et al., "A Degraded Reconstruction Enhancement-based Method for Tiny Ship Detection in Remote Sensing Images with A New Large-scale Dataset", IEEE TGRS 2022

---

## 七、检查清单

复现完成后，检查以下项目：

- [ ] 环境配置完成（Python、CUDA、依赖）
- [ ] 数据集下载并验证（图像数量、标注格式）
- [ ] 代码获取并理解（仓库结构、关键模块）
- [ ] 训练配置正确（参数、数据路径）
- [ ] 训练完成（500 epochs，无错误）
- [ ] 评测完成（验证集、测试集）
- [ ] 推理可视化完成（成功/失败案例）
- [ ] 实验记录完整（使用模板）
- [ ] 结果与论文对比（AP、FPS、Params）
- [ ] 代码和结果归档（checkpoints、logs、可视化）
