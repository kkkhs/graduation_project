# 本机 MVP 运行说明（含 PowerShell 示例）

## 1. 复制配置文件
```bash
cp configs/models.example.yaml configs/models.yaml
```

## 2. 修改模型路径
在 `configs/models.yaml` 中修改：
- `weight_path`
- `config_path`
- `input_size`
- `default_conf_threshold`
- `default_iou_threshold`

DRENet 非 mmdet 格式时可用插件：
`config_path: "/abs/path/to/tools/drenet_plugin_template.py:build_predictor"`

## 3. 统一入口运行（bash）
```bash
PYTHONPATH=. python3 tools/run_predict.py \
  --config configs/models.yaml \
  --image /absolute/path/to/test.jpg \
  --mode ensemble \
  --models drenet,mmdet_fcos,yolo \
  --conf 0.25 \
  --iou 0.5 \
  --fusion-iou 0.55 \
  --min-votes 1
```

## 4. 统一入口运行（PowerShell）
```powershell
cd E:\work\graduation_project
$env:PYTHONPATH='.'
python tools/run_predict.py --config configs/models.yaml --image E:/datasets/LEVIR-Ship/test/images/xxx.png --mode ensemble --models drenet,mmdet_fcos,yolo --conf 0.25 --iou 0.5 --fusion-iou 0.55 --min-votes 1
```

## 5. 当前状态说明
- 配置读取、模型路由、阈值校验、输出标准化：已实现。
- DRENet/MMDet/YOLO Adapter：已接入真实推理入口（依赖对应框架环境与权重文件）。
- 输出 schema：`src/contracts/result_schema.json`。
- 兼容层保留：`src/core/*`、`src/adapters/*`（仅 re-export）。
