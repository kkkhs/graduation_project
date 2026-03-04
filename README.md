# 毕业设计仓库

题目：《遥感场景下的微小船舶检测系统设计与实现》

本仓库用于支撑“论文 + 实验 + 系统 Demo”一体化交付，当前已具备：
- 三模型统一推理入口（`drenet` / `mmdet_fcos` / `yolo`）
- 多模型融合推理（IoU 聚类 + 加权融合，`ensemble`）
- 结果可视化与 JSON 落盘
- Python 桌面 UI（Qt）
- 单元测试与兼容导入层

## 快速开始

### 1) 环境准备

推荐 Python 3.10，在仓库根目录执行：

```bash
python3 -m pip install -U pip
python3 -m pip install pyyaml pillow
```

按需安装：

```bash
# YOLO 推理
python3 -m pip install ultralytics

# 可视化窗口（--show）与桌面 UI
python3 -m pip install matplotlib pyside6

# MMDetection / DRENet(mmdet模式)
python3 -m pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

### 2) 准备模型配置

```bash
cp configs/models.example.yaml configs/models.yaml
```

然后编辑 `configs/models.yaml`：
- 必改：每个模型的 `weight_path`
- 按模式改：`config_path`
  - MMDetection / DRENet(mmdet模式)：填写配置文件绝对路径
  - DRENet 自定义模式：`/abs/path/to/plugin.py:build_predictor`
- 可选：默认阈值 `default_conf_threshold` / `default_iou_threshold`

> DRENet 插件模板见：`tools/drenet_plugin_template.py`

### 3) 命令行推理（JSON 输出）

单模型示例：

```bash
PYTHONPATH=. python3 tools/run_predict.py \
  --config configs/models.yaml \
  --image outputs/demo_input.jpg \
  --mode single \
  --model yolo \
  --conf 0.25 \
  --iou 0.50
```

融合示例：

```bash
PYTHONPATH=. python3 tools/run_predict.py \
  --config configs/models.yaml \
  --image outputs/demo_input.jpg \
  --mode ensemble \
  --models drenet,mmdet_fcos,yolo \
  --conf 0.25 \
  --iou 0.50 \
  --fusion-iou 0.55 \
  --min-votes 1
```

### 4) 可视化输出（图片 + JSON）

```bash
PYTHONPATH=. python3 tools/visualize_predict.py \
  --config configs/models.yaml \
  --image outputs/demo_input.jpg \
  --mode ensemble \
  --models drenet,mmdet_fcos,yolo \
  --vis-out outputs/visualizations/vis_result.jpg \
  --json-out outputs/predictions/pred_result.json
```

如需弹窗显示结果，可额外加 `--show`（需要 `matplotlib`）。

### 5) 桌面 UI（Qt）

```bash
PYTHONPATH=. python3 tools/desktop_ui_qt.py
```

默认输出：
- 可视化图：`outputs/ui_visualizations/`
- JSON：`outputs/predictions/`

### 6) 运行测试

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -p "test_*.py"
```

## 输出格式

推理结果遵循统一字段：
- `image_id`
- `model_name`（融合结果固定为 `ensemble`）
- `bbox`（`[x, y, w, h]`）
- `score`
- `category_id`
- `inference_time`（毫秒）

完整 schema 见：`src/contracts/result_schema.json`

## 目录概览

```text
configs/    模型注册与运行配置（models.yaml）
src/        分层核心代码（application/domain/infrastructure）
tools/      CLI、可视化入口、桌面 UI
tests/      单元与兼容性测试
docs/       规范、实验手册、结果模板、系统设计文档
outputs/    推理输出（json/可视化）
scripts/    训练机结果回传脚本
```

## 文档导航

- 任务主清单：`docs/spec.md`
- 执行看板：`docs/spec_todo.md`
- 实验总手册：`docs/experiments/README.md`
- 3060 训练机手册：`docs/experiments/3060_execution_playbook.md`
- 结果模板说明：`docs/results/README.md`
- 架构说明：`docs/system/architecture.md`
- 推理接口契约：`docs/system/predict_api_contract.md`
- 本机 MVP 跑通说明：`docs/system/local_mvp_run.md`
- 常见问题：`docs/ops/local_run_issues.md`
- 开题材料入口：`docs/kaiti/kaiti.md`

## 常见问题（最短路径）

- `ModuleNotFoundError: No module named 'src'`
  - 用 `PYTHONPATH=.` 启动所有入口脚本。
- 模型依赖未安装（`ultralytics` / `mmdet`）
  - 按上面的“按需安装”分框架补齐依赖。
- 配置报错（权重或配置文件不存在）
  - 先检查 `configs/models.yaml` 中路径是否为本机可访问的绝对路径。
