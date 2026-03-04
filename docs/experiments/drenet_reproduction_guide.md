# DRENet 复现指南（固定 commit + PowerShell）

## 1. 复现目标
1. 跑通 DRENet 在 LEVIR-Ship 的训练、评测、推理流程。
2. 产出可写入论文的 AP50 / Precision / Recall / F1 与可视化结果。

## 2. 版本锁定
- 仓库：`https://github.com/WindVChen/DRENet.git`
- 固定 commit：`a187dbe0f623b521a62c6176c7cafaa7322f5f66`

## 3. 环境与代码准备
```powershell
cd E:\work
git clone https://github.com/WindVChen/DRENet.git
cd DRENet
git checkout a187dbe0f623b521a62c6176c7cafaa7322f5f66
pip install -r requirements.txt
```

## 4. 数据与退化图准备
> DRENet 训练依赖 `train/degrade`，必须先生成。
```powershell
cd E:\datasets\LEVIR-Ship
python E:\work\DRENet\DegradeGenerate.py
```
完成后检查：
- `E:/datasets/LEVIR-Ship/train/images`
- `E:/datasets/LEVIR-Ship/train/degrade`
- `E:/datasets/LEVIR-Ship/train/labels`

## 5. 冒烟训练（1 epoch）
```powershell
cd E:\work\DRENet
python train.py --cfg .\models\DRENet.yaml --epochs 1 --workers 2 --batch-size 2 --device 0 --project .\runs\drenet_smoke --data .\data\ship.yaml
```

## 6. 正式训练（论文主结果）
```powershell
python train.py --cfg .\models\DRENet.yaml --epochs 300 --workers 4 --batch-size 4 --device 0 --project .\runs\drenet_main --data .\data\ship.yaml
```

## 7. 评测与推理
```powershell
python test.py --weights .\runs\drenet_main\exp\weights\best.pt --project .\runs\drenet_eval --device 0 --batch-size 4 --data .\data\ship.yaml
python detect.py --weights .\runs\drenet_main\exp\weights\best.pt --source E:\datasets\LEVIR-Ship\test\images --device 0
```

## 8. 续训命令
```powershell
python train.py --resume .\runs\drenet_main\exp\weights\last.pt
```

## 9. 交付标准
- 至少 1 份完整实验日志（`docs/experiments/logs/`）
- 至少 1 份指标结果（AP50/P/R/F1）
- 至少 1 组可视化样例（成功/误检/漏检）
