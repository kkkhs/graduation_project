# 本机/训练机运行问题记录

> 结构：问题 -> 原因 -> 解决 -> 复现命令
> 更新时间：2026-05-16

## 问题 1：`ModuleNotFoundError: No module named 'src'`

- 原因：项目根目录未加入 Python 模块搜索路径。
- 解决：统一用 `PYTHONPATH=.` 启动。
- 复现命令：

```bash
PYTHONPATH=. python3 tools/run_predict.py --config configs/models.yaml --image outputs/test.jpg --mode single --model yolo
```

## 问题 2：`ModuleNotFoundError: No module named 'yaml'`

- 原因：缺少 `PyYAML`。
- 解决：安装依赖。
- 复现命令：

```bash
python3 -m pip install --user pyyaml
```

## 问题 3：推理失败（依赖缺失）

- 原因：`ultralytics`/`mmdet` 未安装或版本不匹配。
- 解决：按模型分框架安装依赖，不混装无关组件。
- 复现命令：

```bash
python3 -m pip install --user ultralytics
# MMDet 环境
python3 -m pip install --user openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

## 问题 4：训练机中断（抢占/断电/重启）

- 原因：训练进程未持续运行或实例被回收。
- 解决：
  1. 固定 checkpoint 间隔（例如每 N epoch）
  2. 使用最近 checkpoint 续训
  3. 续训日志记录“旧实验ID -> 新实验ID”
- 复现命令（示例）：

```bash
python train.py --resume /path/to/last_checkpoint.pth
```

## 问题 6：显存不足（OOM）

- 原因：模型/输入尺寸/批次超出 3060 显存能力。
- 处理优先级：
  1. 降低 `batch`
  2. 降低 `image_size`
  3. 开启 `AMP`（自动混合精度）
- 复现命令（示例）：

```bash
python train.py --batch-size 2 --img-size 512 --amp
```

## 问题 7：DRENet 对非正方形/非 512×512 图片推理崩溃

- 现象：`RuntimeError: The size of tensor a (144) must match the size of tensor b (256) at non-singleton dimension 1`
- 原因：DRENet 的 `BottleneckResAtnMHSA` 模块（[`experiments/drenet/DRENet/models/common.py:119`](../../experiments/drenet/DRENet/models/common.py:119)）在初始化时将位置编码 `rel_h`/`rel_w` 固定为训练时的特征图空间尺寸（512×512 输入 → 某层 16×16=256 个位置）。当输入图片宽高比不同时，letterbox 后有效区域变化，导致该层特征图变为 12×12=144 个位置，`content_content`（dim1=144）与 `content_position`（dim1=256）维度不匹配。
- 影响范围：所有非 512×512 正方形输入（包括任意非数据集图片）。
- 修复方案（待实施）：
  - **方案 A（推荐）**：取消注释 [`common.py:136-141`](../../experiments/drenet/DRENet/models/common.py:136) 中的 resolution-agnostic 位置编码，用 `F.interpolate` 动态适配特征图尺寸。参考 [DRENet Issue #10](https://github.com/WindVChen/DRENet/issues/10)。
  - **方案 B（简单）**：在 [`tools/drenet_local_plugin.py`](../../tools/drenet_local_plugin.py) 中强制将输入 resize 为正方形 512×512（不使用 letterbox），但有损宽高比。
- 复现命令：

```bash
# 用任意非 512×512 正方形图片触发
PYTHONPATH=. python3 -c "
from tools.drenet_local_plugin import build_predictor
p = build_predictor('experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights/best.pt', '', 'cpu')
p('outputs/tasks/37/input/4559992950.png', 0.25, 0.5)
"
```

## 预防措施

- 所有入口统一加 `PYTHONPATH=.`。
- 每轮实验写日志：环境、命令、权重、指标、异常。
- 训练完成立即回传，并按固定顺序回填结果文档。
