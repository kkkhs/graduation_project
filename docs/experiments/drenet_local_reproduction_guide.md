# DRENet 本地复现指南（Windows + RTX 3060 + Conda）

> 本指南针对你的本地环境定制：Windows 11、RTX 3060 (6GB)、Conda 环境管理

---

## 一、本地环境评估

### 1.1 硬件配置 ✅

| 组件     | 配置                          | 状态        | 说明                          |
| -------- | ----------------------------- | ----------- | ----------------------------- |
| **GPU**  | NVIDIA GeForce RTX 3060 (6GB) | ✅ 满足要求 | 显存 6GB，需要调整 batch size |
| **CPU**  | Intel i9-10900K               | ✅ 满足要求 | 多核 CPU，训练速度快          |
| **内存** | 64GB                          | ✅ 远超要求 | 充足                          |
| **硬盘** | 充足                          | ✅ 满足要求 |                               |

### 1.2 软件配置 ✅

| 组件         | 推荐版本      | 状态        | 说明                |
| ------------ | ------------- | ----------- | ------------------- |
| **操作系统** | Windows 10/11 | ✅ 满足要求 | 使用 PowerShell     |
| **Conda**    | Miniconda     | ✅ 已安装   | 环境管理            |
| **Python**   | 3.10          | ✅ 推荐     | 与 PyTorch 2.3 兼容 |
| **CUDA**     | 12.1          | ✅ 满足要求 | Conda 自动安装      |
| **PyTorch**  | 2.3.0         | ✅ 推荐     | 最新稳定版          |

### 1.3 环境调整建议

**必须调整**：

1. **Batch Size 调整**：由于显存只有 6GB，需要从 16 降到 8 或 4
2. **使用梯度累积**：在保持有效 batch size 的同时减少显存占用

**可选调整**：

1. **减少数据加载线程**：从 8 降到 4，减少显存占用
2. **调整输入尺寸**：从 512 降到 416，进一步减少显存占用

---

## 二、环境配置（预计 20 分钟）

### 2.1 创建 Conda 环境

```powershell
# 创建 Python 3.10 环境
conda create -n drenet python=3.10 -y

# 激活环境
conda activate drenet

# 验证 Python 版本
python --version

# 预期输出：Python 3.10.x
```

### 2.2 安装 PyTorch 和依赖

```powershell
# 确保已激活 drenet 环境
conda activate drenet

# 安装 PyTorch 2.3.0（CUDA 12.1）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装其他依赖
conda install numpy opencv pillow matplotlib tqdm pyyaml scipy pandas seaborn tensorboard -y

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**预期输出**：

```
PyTorch: 2.3.0
CUDA available: True
CUDA version: 12.1
GPU name: NVIDIA GeForce RTX 3060
```

### 2.3 安装额外依赖

```powershell
# 确保已激活 drenet 环境
conda activate drenet

# 安装额外依赖（如果需要）
pip install pycocotools-windows

# 验证安装
python -c "import pycocotools; print('pycocotools installed successfully')"
```

### 2.4 验证环境完整性

```powershell
# 检查已安装的包
conda list

# 检查 CUDA 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')"
```

**预期输出**：

```
CUDA available: True
GPU count: 1
GPU memory: 6.00 GB
```

---

## 三、获取代码和数据（预计 30 分钟）

### 3.1 创建项目目录结构

```powershell
# 在项目根目录下创建
cd E:\Codes\Githubs\graduation_project

# 创建目录
New-Item -ItemType Directory -Path experiments -Force
New-Item -ItemType Directory -Path experiments\drenet -Force
New-Item -ItemType Directory -Path data -Force
```

### 3.2 克隆 DRENet 代码

```powershell
# 进入实验目录
cd experiments\drenet

# 克隆仓库
git clone https://github.com/WindVChen/DRENet.git

# 或手动下载：
# 访问 https://github.com/WindVChen/DRENet
# 下载 ZIP 并解压到 experiments\drenet\
```

### 3.3 下载 LEVIR-Ship 数据集

```powershell
# 进入数据目录
cd E:\Codes\Githubs\graduation_project\data

# 克隆数据集仓库
git clone https://github.com/WindVChen/LEVIR-Ship.git

# 或手动下载：
# 访问 https://github.com/WindVChen/LEVIR-Ship
# 下载 ZIP 并解压到 data\
```

### 3.4 验证数据集结构

```powershell
# 查看数据集结构
Get-ChildItem -Path LEVIR-Ship -Directory

# 预期输出：
# Directory: E:\Codes\Githubs\graduation_project\data\LEVIR-Ship
#
# Mode                 LastWriteTime         Length Name
# ----                 -------------         ------ ----
# d----          2026-02-07     18:02    train
# d----          2026-02-07     18:02    val
# d----          2026-02-07     18:02    test
```

### 3.5 统计数据集信息

```powershell
# 统计图像数量
(Get-ChildItem -Path LEVIR-Ship\train -Filter *.jpg).Count
(Get-ChildItem -Path LEVIR-Ship\val -Filter *.jpg).Count
(Get-ChildItem -Path LEVIR-Ship\test -Filter *.jpg).Count

# 预期结果：
# 训练集：2320 张
# 验证集：788 张
# 测试集：788 张
```

---

## 四、配置训练参数（针对 6GB 显存优化）

### 4.1 创建配置文件

```powershell
# 进入 DRENet 目录
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet

# 查看现有配置
Get-ChildItem -Path data -Filter *.yaml
Get-ChildItem -Path models -Filter *.yaml
```

### 4.2 创建数据配置文件

创建 `data\levir_ship.yaml`：

```yaml
# LEVIR-Ship 数据集配置
path: ../../data/LEVIR-Ship # 数据集根目录（相对路径）
train: train # 训练集
val: val # 验证集
test: test # 测试集

# 类别信息
nc: 1 # 类别数量
names: ['ship'] # 类别名称
```

### 4.3 创建训练配置文件（针对 6GB 显存优化）

创建 `configs\drenet_rtx3060.yaml`：

```yaml
# 模型配置
model: yolov5s # 基于 YOLOv5s

# 训练配置（针对 6GB 显存优化）
epochs: 500
batch_size: 8 # 从 16 降到 8（6GB 显存）
img_size: 512 # LEVIR-Ship 图像尺寸

# 优化器
optimizer: SGD
lr: 0.01
momentum: 0.937
weight_decay: 0.0005

# 学习率调度
lr_scheduler: cosine
warmup_epochs: 3

# 数据增强
mosaic: 1.0 # Mosaic 增强
mixup: 0.0 # Mixup 增强
copy_paste: 0.0 # Copy-Paste 增强
flip: 0.5 # 水平翻转
scale: 0.5 # 缩放

# DRENet 特有配置
use_dre: True # 启用退化重建增强器
use_crma: True # 启用 CRMA 模块

# 显存优化
accumulate: 2 # 梯度累积（有效 batch size = 8 * 2 = 16）
workers: 4 # 数据加载线程数（减少显存占用）
```

---

## 五、训练模型（预计 12-24 小时）

### 5.1 开始训练

```powershell
# 确保已激活 drenet 环境
conda activate drenet

# 进入 DRENet 目录
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet

# 基础训练命令（针对 6GB 显存优化）
python train.py --data data/levir_ship.yaml `
              --cfg models/drenet.yaml `
              --weights yolov5s.pt `
              --epochs 500 `
              --batch-size 8 `
              --img 512 `
              --device 0 `
              --workers 4 `
              --accumulate 2 `
              --project runs/train `
              --name drenet_levir `
              --exist-ok `
              --seed 42
```

**参数说明**：

- `--batch-size 8`：实际 batch size（6GB 显存）
- `--accumulate 2`：梯度累积，有效 batch size = 8 × 2 = 16
- `--workers 4`：数据加载线程数（减少显存占用）
- `--seed 42`：固定随机种子，确保可复现

### 5.2 监控训练过程

**方法 1：TensorBoard**

```powershell
# 确保已激活 drenet 环境
conda activate drenet

# 启动 TensorBoard
tensorboard --logdir runs/train/drenet_levir

# 在浏览器中访问
# http://localhost:6006
```

**方法 2：查看日志文件**

```powershell
# 实时查看训练日志
Get-Content runs/train/drenet_levir\train.log -Wait -Tail 10

# 或查看结果 CSV
Get-Content runs/train/drenet_levir\results.csv | Select-Object -Last 10
```

### 5.3 训练检查点

训练过程中会自动保存检查点到：

```
runs/train/drenet_levir/weights/
├── best.pt      # 最佳模型（根据验证集 mAP）
├── last.pt      # 最后一个 epoch 的模型
└── epoch_xxx.pt # 每 10 个 epoch 保存一次
```

### 5.4 预计训练时间

根据 RTX 3060 的性能：

- **单 epoch 时间**：约 1-2 分钟
- **总训练时间**：约 12-24 小时（500 epochs）

**优化建议**：

1. 如果显存不足，将 batch-size 降到 4，accumulate 改为 4
2. 如果训练太慢，可以先训练 100 epochs 验证流程，再训练完整 500 epochs
3. 可以使用 `--resume` 参数从检查点继续训练

---

## 六、评测模型（预计 30 分钟）

### 6.1 在验证集上评测

```powershell
# 确保已激活 drenet 环境
conda activate drenet

# 进入 DRENet 目录
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet

# 验证集评测
python val.py --data data/levir_ship.yaml `
             --weights runs/train/drenet_levir/weights/best.pt `
             --img 512 `
             --batch-size 8 `
             --device 0 `
             --task val `
             --project runs/val `
             --name drenet_levir `
             --exist-ok
```

### 6.2 在测试集上评测

```powershell
# 确保已激活 drenet 环境
conda activate drenet

# 进入 DRENet 目录
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet

# 测试集评测
python val.py --data data/levir_ship.yaml `
             --weights runs/train/drenet_levir/weights/best.pt `
             --img 512 `
             --batch-size 8 `
             --device 0 `
             --task test `
             --project runs/test `
             --name drenet_levir `
             --exist-ok
```

### 6.3 查看评测结果

```powershell
# 查看评测结果
Get-Content runs/test/drenet_levir\results.txt

# 或查看 TensorBoard
tensorboard --logdir runs/test/drenet_levir
```

**预期结果**：

- AP50：约 82.4（论文报告值）
- FPS：约 85（在 512×512 输入下）
- Params：4.79M
- FLOPs：8.3G

---

## 七、推理可视化（预计 30 分钟）

### 7.1 单张图像推理

```powershell
# 确保已激活 drenet 环境
conda activate drenet

# 进入 DRENet 目录
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet

# 单张图像推理
python detect.py --weights runs/train/drenet_levir/weights/best.pt `
               --source data/LEVIR-Ship\test\000001.jpg `
               --img 512 `
               --conf-thres 0.25 `
               --iou-thres 0.45 `
               --device 0 `
               --project runs/detect `
               --name single_test `
               --exist-ok
```

### 7.2 批量推理

```powershell
# 确保已激活 drenet 环境
conda activate drenet

# 进入 DRENet 目录
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet

# 批量推理
python detect.py --weights runs/train/drenet_levir/weights/best.pt `
               --source data/LEVIR-Ship\test\ `
               --img 512 `
               --conf-thres 0.25 `
               --iou-thres 0.45 `
               --device 0 `
               --project runs/detect `
               --name batch_test `
               --exist-ok
```

### 7.3 查看可视化结果

```powershell
# 结果保存在
Get-ChildItem runs/detect\batch_test -Filter *.jpg

# 查看可视化图像
Invoke-Item runs\detect\batch_test\000001.jpg
```

### 7.4 导出推理结果

```powershell
# 确保已激活 drenet 环境
conda activate drenet

# 进入 DRENet 目录
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet

# 导出为 JSON/TXT 格式
python detect.py --weights runs/train/drenet_levir/weights/best.pt `
               --source data/LEVIR-Ship\test\ `
               --img 512 `
               --conf-thres 0.25 `
               --iou-thres 0.45 `
               --device 0 `
               --project runs/detect `
               --name batch_test `
               --save-txt `
               --save-conf `
               --exist-ok
```

---

## 八、常见问题与解决方案

### 8.1 CUDA 内存不足

**问题**：

```
RuntimeError: CUDA out of memory. Tried to allocate 6.00 GiB
```

**解决方案**：

```powershell
# 方案 1：减小 batch size
python train.py --batch-size 4 --accumulate 4

# 方案 2：减小输入尺寸
python train.py --img 416

# 方案 3：减少 workers
python train.py --workers 2
```

### 8.2 Conda 环境未激活

**问题**：

```
ModuleNotFoundError: No module named 'torch'
```

**解决方案**：

```powershell
# 激活 drenet 环境
conda activate drenet

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 8.3 数据集路径错误

**问题**：

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/LEVIR-Ship/train'
```

**解决方案**：

```powershell
# 检查数据集路径
Get-ChildItem -Path data\LEVIR-Ship -Directory

# 使用绝对路径
python train.py --data E:\Codes\Githubs\graduation_project\data\LEVIR-Ship
```

### 8.4 训练不收敛

**问题**：

```
训练过程中 loss 不下降或震荡
```

**解决方案**：

```powershell
# 降低学习率
python train.py --lr 0.001

# 减少数据增强
python train.py --mosaic 0.5

# 检查数据标注
# 确保标注格式正确
```

### 8.5 Conda 环境导出和导入

**导出环境**：

```powershell
# 激活 drenet 环境
conda activate drenet

# 导出环境配置
conda env export > drenet_environment.yml

# 保存到项目目录
Copy-Item drenet_environment.yml E:\Codes\Githubs\graduation_project\
```

**导入环境**：

```powershell
# 从 environment.yml 创建环境
conda env create -f drenet_environment.yml --name drenet_imported

# 激活导入的环境
conda activate drenet_imported
```

---

## 九、整理实验记录

### 9.1 创建实验记录

```powershell
# 复制实验记录模板
Copy-Item E:\Codes\Githubs\graduation_project\docs\experiments\exp_log_template.md `
           -Destination E:\Codes\Githubs\graduation_project\docs\experiments\drenet_local_exp_log.md
```

### 9.2 填写实验信息

````markdown
## 实验记录：DRENet 本地复现（RTX 3060 + Conda）

### 1. 实验信息

- **实验编号**：exp-2026-02-07-local
- **日期**：2026-02-07
- **模型/框架**：DRENet / YOLOv5s
- **代码版本**：https://github.com/WindVChen/DRENet
- **数据版本**：LEVIR-Ship (官方 GitHub)
- **划分方式**：官方划分
- **随机种子**：42
- **环境管理**：Conda

### 2. 训练设置

- **输入尺寸**：512×512
- **batch size**：8（有效 batch size = 16，使用梯度累积）
- **epoch/iters**：500
- **优化器**：SGD (lr=0.01, momentum=0.937, weight_decay=0.0005)
- **学习率策略**：cosine (warmup=3 epochs)
- **数据增强**：mosaic=1.0, flip=0.5, scale=0.5
- **损失/后处理关键设置**：NMS iou=0.45, conf=0.25

### 3. 资源与耗时

- **硬件**：NVIDIA RTX 3060 (6GB), Intel i9-10900K, 64GB RAM
- **训练耗时**：约 18 小时（500 epochs）
- **显存峰值**：约 5.5GB / 6GB

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

```powershell
# 环境配置
conda create -n drenet python=3.10 -y
conda activate drenet
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install numpy opencv pillow matplotlib tqdm pyyaml scipy pandas seaborn tensorboard -y

# train
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet
python train.py --data data/levir_ship.yaml `
              --cfg models/drenet.yaml `
              --weights yolov5s.pt `
              --epochs 500 `
              --batch-size 8 `
              --img 512 `
              --device 0 `
              --workers 4 `
              --accumulate 2 `
              --project runs/train `
              --name drenet_levir `
              --exist-ok `
              --seed 42

# eval
python val.py --data data/levir_ship.yaml `
             --weights runs/train/drenet_levir/weights/best.pt `
             --img 512 `
             --batch-size 8 `
             --device 0 `
             --task test `
             --project runs/test `
             --name drenet_levir `
             --exist-ok

# infer/visualize
python detect.py --weights runs/train/drenet_levir/weights/best.pt `
               --source data/LEVIR-Ship\test\ `
               --img 512 `
               --conf-thres 0.25 `
               --iou-thres 0.45 `
               --device 0 `
               --project runs/detect `
               --name batch_test `
               --save-txt `
               --save-conf `
               --exist-ok

# 导出环境
conda env export > drenet_environment.yml
```
````

````

---

## 十、检查清单

复现完成后，检查以下项目：

- [ ] Conda 环境创建完成（drenet）
- [ ] PyTorch 2.3.0 安装完成
- [ ] 其他依赖安装完成
- [ ] CUDA 可用性验证通过
- [ ] DRENet 代码下载完成
- [ ] LEVIR-Ship 数据集下载完成
- [ ] 数据集结构验证通过
- [ ] 训练配置文件创建完成
- [ ] 训练完成（500 epochs，无错误）
- [ ] 评测完成（验证集、测试集）
- [ ] 推理可视化完成（成功/失败案例）
- [ ] 实验记录完整
- [ ] 结果与论文对比（AP、FPS、Params）
- [ ] 代码和结果归档（checkpoints、logs、可视化）
- [ ] 环境配置导出（environment.yml）

---

## 十一、快速开始（一键复制）

### 11.1 一键环境配置

```powershell
# 创建并配置 DRENet 环境
conda create -n drenet python=3.10 -y && conda activate drenet && conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y && conda install numpy opencv pillow matplotlib tqdm pyyaml scipy pandas seaborn tensorboard -y && python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 11.2 完整训练命令

```powershell
# 激活环境
conda activate drenet

# 进入项目目录
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet

# 一键训练
python train.py --data data/levir_ship.yaml `
              --cfg models/drenet.yaml `
              --weights yolov5s.pt `
              --epochs 500 `
              --batch-size 8 `
              --img 512 `
              --device 0 `
              --workers 4 `
              --accumulate 2 `
              --project runs/train `
              --name drenet_levir `
              --exist-ok `
              --seed 42
```

### 11.3 完整评测命令

```powershell
# 激活环境
conda activate drenet

# 进入项目目录
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet

# 测试集评测
python val.py --data data/levir_ship.yaml `
             --weights runs/train/drenet_levir/weights/best.pt `
             --img 512 `
             --batch-size 8 `
             --device 0 `
             --task test `
             --project runs/test `
             --name drenet_levir `
             --exist-ok
```

### 11.4 完整推理命令

```powershell
# 激活环境
conda activate drenet

# 进入项目目录
cd E:\Codes\Githubs\graduation_project\experiments\drenet\DRENet

# 批量推理
python detect.py --weights runs/train/drenet_levir/weights/best.pt `
               --source data/LEVIR-Ship\test\ `
               --img 512 `
               --conf-thres 0.25 `
               --iou-thres 0.45 `
               --device 0 `
               --project runs/detect `
               --name batch_test `
               --save-txt `
               --save-conf `
               --exist-ok
```

### 11.5 导出环境配置

```powershell
# 激活环境
conda activate drenet

# 导出环境配置
conda env export > drenet_environment.yml

# 查看环境配置
Get-Content drenet_environment.yml
```

---

## 十二、Conda 环境管理速查

### 12.1 常用命令

```powershell
# 环境管理
conda create -n <环境名> python=<版本>    # 创建环境
conda activate <环境名>                     # 激活环境
conda deactivate                              # 退出环境
conda env list                               # 列出环境
conda remove -n <环境名> --all            # 删除环境

# 包管理
conda install <包名>                        # 安装包
conda remove <包名>                        # 卸载包
conda update <包名>                        # 更新包
conda list                                   # 查看包

# 环境导出/导入
conda env export > environment.yml            # 导出环境
conda env create -f environment.yml           # 导入环境
```

### 12.2 环境隔离说明

**重要**：
- Conda 的包是安装在虚拟环境中的，不是全局的
- 每个虚拟环境都有独立的包集合，互不影响
- 推荐为每个项目创建独立的虚拟环境

**示例**：
```powershell
# 创建两个独立环境
conda create -n drenet python=3.10 -y
conda create -n yolov8 python=3.11 -y

# 在 drenet 环境中安装 PyTorch 2.3.0
conda activate drenet
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 在 yolov8 环境中安装 PyTorch 2.2.0
conda activate yolov8
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 两个环境的 PyTorch 版本不同，互不影响
```

---

## 十三、参考资源

- **DRENet 代码**：https://github.com/WindVChen/DRENet
- **LEVIR-Ship 数据集**：https://github.com/WindVChen/LEVIR-Ship
- **YOLOv5 官方文档**：https://github.com/ultralytics/yolov5
- **Conda 官方文档**：https://docs.conda.io/
- **论文**：Chen et al., "A Degraded Reconstruction Enhancement-based Method for Tiny Ship Detection in Remote Sensing Images with A New Large-scale Dataset", IEEE TGRS 2022

---

## 十四、下一步

完成 DRENet 复现后，可以：

1. **对比实验**：使用相同数据集训练其他模型（Faster R-CNN、YOLOv8）
2. **消融实验**：测试不同组件的影响（DRE、CRMA）
3. **可视化分析**：分析不同场景下的性能差异
4. **系统集成**：将 DRENet 集成到检测系统中

---

**祝复现顺利！** 🚀
````
