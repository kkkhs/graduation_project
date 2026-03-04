# 纯 Python 可视化运行方式

不使用 Web，直接命令行执行推理并输出可视化图片 + JSON。

## 1. 安装依赖
```bash
python3 -m pip install --user pyyaml pillow matplotlib ultralytics
# 若使用 mmdet / drenet(mmdet模式) 还需安装 mmdet 依赖
# pip install openmim && mim install mmengine && mim install \"mmcv>=2.0.0\" && mim install mmdet
```

## 2. 准备配置
```bash
cp configs/models.example.yaml configs/models.yaml
```

## 3. 运行
```bash
cd /Users/khs/codes/graduation_project
PYTHONPATH=. python3 tools/visualize_predict.py \
  --config configs/models.yaml \
  --image /absolute/path/to/test.jpg \
  --mode ensemble \
  --models drenet,mmdet_fcos,yolo \
  --conf 0.25 \
  --iou 0.5 \
  --fusion-iou 0.55 \
  --min-votes 1 \
  --vis-out outputs/visualizations/vis_result.jpg \
  --json-out outputs/predictions/pred_result.json
```

单模型模式：
```bash
PYTHONPATH=. python3 tools/visualize_predict.py \
  --config configs/models.yaml \
  --image /absolute/path/to/test.jpg \
  --mode single \
  --model yolo
```

## 4. 可选弹窗显示
加 `--show` 参数可用 matplotlib 弹窗：
```bash
PYTHONPATH=. python3 tools/visualize_predict.py ... --show
```

## 5. 输出文件
- 可视化图：`outputs/visualizations/*.jpg`
- 结构化结果：`outputs/predictions/*.json`
