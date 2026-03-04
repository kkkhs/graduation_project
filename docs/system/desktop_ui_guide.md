# Python 桌面 UI 使用说明（非 Web）

## 1. 安装依赖
```bash
python3 -m pip install --user pyyaml pillow pyside6 ultralytics
# 若使用 mmdet / drenet(mmdet模式) 还需安装 mmdet 依赖
# pip install openmim && mim install mmengine && mim install \"mmcv>=2.0.0\" && mim install mmdet
```

## 2. 准备配置
```bash
cp configs/models.example.yaml configs/models.yaml
```

## 3. 启动桌面界面
```bash
cd /Users/khs/codes/graduation_project
PYTHONPATH=. python3 tools/desktop_ui_qt.py
```

## 4. 交互流程
1. 点击“选择图片”。
2. 选择模型（`ensemble_all` 或单模型 `drenet / mmdet_fcos / yolo`）。
3. 设置 `conf` 和 `iou`。
4. 点击“执行推理”。

## 5. 输出位置
- 可视化图：`outputs/ui_visualizations/`
- JSON 结果：`outputs/predictions/`

## 6. 说明
- UI 已支持 `ensemble_all` 综合推理。
- 若结果为空，优先检查：权重路径、框架依赖、阈值设置与模型输出质量。
