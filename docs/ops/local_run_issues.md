# 本机/训练机运行问题记录

> 结构：问题 -> 原因 -> 解决 -> 复现命令
> 更新时间：2026-03-03

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

## 问题 4：`tkinter` 桌面界面无法启动
- 原因：本机 Tk GUI 与当前 macOS patch level 兼容性问题。
- 解决：使用 Qt 桌面 UI（PySide6）。
- 复现命令：
```bash
python3 -m pip install --user pyside6
PYTHONPATH=. python3 tools/desktop_ui_qt.py
```

## 问题 5：训练机中断（抢占/断电/重启）
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

## 预防措施
- 所有入口统一加 `PYTHONPATH=.`。
- 每轮实验写日志：环境、命令、权重、指标、异常。
- 训练完成立即回传，并按固定顺序回填结果文档。
