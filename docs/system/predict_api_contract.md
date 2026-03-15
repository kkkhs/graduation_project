# 推理接口契约（实现标准）

> 说明：本文同时覆盖 Python 内部推理接口与 Web API v1（FastAPI）。

## 1. 接口清单

### 1.0 Web API v1
- `POST /api/v1/tasks/infer`
- `GET /api/v1/tasks`
- `GET /api/v1/tasks/{task_id}`
- `GET /api/v1/tasks/{task_id}/results`
- `GET /api/v1/models`
- `PATCH /api/v1/models/{model_key}`
- `GET /api/v1/health`

### 1.1 单模型推理
```python
predict(image_path, model_name, conf_threshold=None, iou_threshold=None) -> list[dict]
```

### 1.2 多模型融合推理
```python
predict_ensemble(
    image_path,
    model_names=None,
    conf_threshold=None,
    iou_threshold=None,
    fusion_iou_threshold=0.55,
    min_votes=1,
) -> list[dict]
```

## 2. 输入参数与约束
- `image_path: str`
  - 必填，图像路径（绝对/相对均可）。
  - 不存在时报错。
- `model_name: str`
  - `predict` 必填。
  - 必须在 `configs/models.yaml` 中注册。
- `model_names: list[str] | None`
  - `predict_ensemble` 可选。
  - `None` 表示使用所有已注册模型。
- `conf_threshold: float | None`
  - 范围 `[0,1]`。
- `iou_threshold: float | None`
  - 范围 `[0,1]`。
- `fusion_iou_threshold: float`
  - 范围 `[0,1]`。
- `min_votes: int`
  - `>= 1`。

## 3. 错误码/异常文本约定
| 场景 | 异常类型 | 异常文本（关键字） |
|---|---|---|
| 模型名不存在 | `ValueError` | `unknown model_name` / `unknown model_names` |
| 图像路径无效 | `ValueError` | `invalid image_path` |
| 阈值越界 | `ValueError` | `threshold out of range` |
| 融合阈值越界 | `ValueError` | `fusion_iou_threshold out of range` |
| 最小投票非法 | `ValueError` | `min_votes must be >= 1` |
| 权重不存在 | `FileNotFoundError` | `weight file not found` |
| 依赖缺失 | `RuntimeError` | `not installed` |
| 推理失败 | `RuntimeError` | `inference failed` |

## 4. 输出结构（统一 schema）
每个元素必须包含：
- `image_id: str`
- `model_name: str`（融合结果固定为 `ensemble`）
- `bbox: [x, y, w, h]`
- `score: float`（0~1）
- `category_id: int`
- `inference_time: float`（ms）

## 5. 输出示例
```json
[
  {
    "image_id": "000123.jpg",
    "model_name": "ensemble",
    "bbox": [122.4, 87.1, 34.5, 18.2],
    "score": 0.82,
    "category_id": 0,
    "inference_time": 19.37
  }
]
```

## 6. 兼容层说明（旧路径可调）
保留旧导入路径，仅做 re-export：
- `from src.core.predictor import UnifiedPredictor`
- `from src.adapters.yolo_adapter import YOLOAdapter`

说明：兼容层不承载新业务逻辑，真实实现在 `src/application` 与 `src/infrastructure`。

## 7. DRENet 自定义插件模式
当 DRENet 不是 mmdet 兼容格式时，可使用插件模式：
- `configs/models.yaml` 中设置：
  - `framework_type: drenet`
  - `config_path: "/abs/path/to/plugin.py:build_predictor"`
- 插件模板：`tools/drenet_plugin_template.py`
