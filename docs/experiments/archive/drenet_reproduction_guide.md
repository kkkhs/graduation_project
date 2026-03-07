# DRENet 复现指南（固定 commit + PowerShell）

## 1. 复现目标
1. 跑通 DRENet 在 LEVIR-Ship 上的训练、评测、推理流程。
2. 产出可写入论文的 `AP50 / Precision / Recall / F1`。
3. 整理至少 1 组可视化样例，支持 `success / miss / false_positive` 三类分析。

## 2. 版本锁定
- 仓库：`https://github.com/WindVChen/DRENet.git`
- 固定 commit：`a187dbe0f623b521a62c6176c7cafaa7322f5f66`
- 训练系统：Windows PowerShell
- 建议 Python：`3.10`

## 3. 环境与代码准备
```powershell
cd E:\work
git clone https://github.com/WindVChen/DRENet.git
cd DRENet
git checkout a187dbe0f623b521a62c6176c7cafaa7322f5f66
pip install -r requirements.txt
```

最小环境检查：
```powershell
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## 4. 数据配置

### 4.1 `ship.yaml` 示例
编辑 `E:\work\DRENet\data\ship.yaml`，至少确认这些字段：

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
- 数据集目录中必须同时存在对应 `labels`

### 4.2 退化图准备
> DRENet 训练依赖 `train/degrade`，必须先生成。

```powershell
cd E:\datasets\LEVIR-Ship
python E:\work\DRENet\DegradeGenerate.py
```

生成后必须检查：
```powershell
Get-ChildItem E:\datasets\LEVIR-Ship\train\degrade | Select-Object -First 5
(Get-ChildItem E:\datasets\LEVIR-Ship\train\images).Count
(Get-ChildItem E:\datasets\LEVIR-Ship\train\degrade).Count
```

核对项：
- `E:/datasets/LEVIR-Ship/train/degrade` 已生成
- 文件数量与 `train/images` 接近
- 文件名能与原图对应

## 5. 冒烟训练（1 epoch）
```powershell
cd E:\work\DRENet
python train.py --cfg .\models\DRENet.yaml --epochs 1 --workers 2 --batch-size 2 --device 0 --project .\runs\drenet_smoke --data .\data\ship.yaml
```

冒烟通过标准：
- 命令能完整跑完
- 生成 `runs/drenet_smoke/.../weights`
- 没有路径错误、数据错误、显存错误

## 6. 正式训练（论文主结果）
```powershell
cd E:\work\DRENet
python train.py --cfg .\models\DRENet.yaml --epochs 300 --workers 4 --batch-size 4 --device 0 --project .\runs\drenet_main --data .\data\ship.yaml
```

若 OOM，固定按以下顺序处理：
1. `batch-size 4 -> 2`
2. `workers 4 -> 2`
3. 必要时再降输入尺寸

保留：
- `best.pt`
- `last.pt`

## 7. 评测与推理
```powershell
cd E:\work\DRENet
python test.py --weights .\runs\drenet_main\exp\weights\best.pt --project .\runs\drenet_eval --device 0 --batch-size 4 --data .\data\ship.yaml
python detect.py --weights .\runs\drenet_main\exp\weights\best.pt --source E:\datasets\LEVIR-Ship\test\images --device 0
```

评测后至少收集：
- `AP50`
- `Precision`
- `Recall`
- `F1`

建议保存到：
- `artifacts/drenet/metrics/metrics_drenet_<exp_id>.json`
- `artifacts/drenet/figures/`
- `artifacts/drenet/weights/`

## 8. 首轮最小成功定义
满足以下条件才算 DRENet 首轮完成：
1. 冒烟训练成功
2. 正式训练成功启动并生成 checkpoint
3. `test.py` 能输出 `AP50 / Precision / Recall / F1`
4. 至少 3 张有效图，覆盖 `success / miss / false_positive`
5. 至少 1 份实验日志可回填到 `docs/experiments/logs/`

## 9. 常见报错与处理顺序

### 9.1 CUDA 不可用
- 现象：`torch.cuda.is_available() == False`
- 处理：先修 Torch / CUDA / 驱动，不要继续训练

### 9.2 `train/degrade` 缺失
- 现象：训练阶段读取退化图失败
- 处理：重新运行 `DegradeGenerate.py`，再检查数量与文件名

### 9.3 `ship.yaml` 路径错误
- 现象：找不到训练、验证或测试集
- 处理：先核对 `train/val/test` 是否都指向 `images`

### 9.4 显存不足
- 现象：训练中 `CUDA out of memory`
- 处理：按 `batch-size -> workers -> input size` 顺序调整

### 9.5 没有 `best.pt`
- 现象：评测时找不到最优权重
- 处理：先确认正式训练是否完整跑过，并检查 `runs/drenet_main/.../weights`

## 10. 续训命令
```powershell
python train.py --resume .\runs\drenet_main\exp\weights\last.pt
```

## 11. 交付标准
- 至少 1 份完整实验日志（`docs/experiments/logs/`）
- 至少 1 份指标结果（`AP50/P/R/F1`）
- 至少 1 组可视化样例（成功/误检/漏检）
- 至少 1 份可回填到 `docs/results/baselines.md` 的结果摘要
