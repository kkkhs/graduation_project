# src 目录详解（推理内核）

`src/` 是本项目的算法推理内核，采用 DDD 分层架构，与 Web 服务层（`backend/`）解耦，可被后端、CLI 工具和脚本复用。

---

## 目录总览

```
src/
├── domain/                    # 领域层：实体 + 抽象接口
│   ├── entities.py            # PredictRequest / PredictionResult
│   └── interfaces.py          # DetectorAdapter 抽象基类
├── application/               # 应用层：业务编排
│   ├── dto.py                 # RuntimeConfig / ModelRuntimeConfig
│   ├── bootstrap.py           # build_predict_service() 工厂函数
│   ├── predict_service.py     # PredictService 推理编排器
│   └── fusion.py              # fuse_predictions() WBF 融合算法
└── infrastructure/            # 基础设施层：具体实现
    ├── config_loader.py       # YAML 配置加载 → RuntimeConfig
    ├── adapter_factory.py     # AdapterFactory 按 framework_type 创建适配器
    ├── visualization.py       # render_detections() PIL 可视化
    └── adapters/              # 三个模型的适配器实现
        ├── base.py            # BaseAdapter 抽象基类
        ├── drenet_adapter.py  # DRENetAdapter（支持 MMDet / 插件两种模式）
        ├── mmdet_adapter.py   # MMDetAdapter（FCOS）
        ├── yolo_adapter.py    # YOLOAdapter（Ultralytics）
        └── parsing.py         # 预测结果归一化工具
```

---

## 一、domain/ — 领域层

> 不依赖任何第三方框架（PyTorch/MMDet/Ultralytics），只依赖 Python 标准库。

### [`entities.py`](src/domain/entities.py)

定义两个核心 dataclass：

| 类                 | 用途           | 关键字段                                                                         |
| ------------------ | -------------- | -------------------------------------------------------------------------------- |
| `PredictRequest`   | 推理请求值对象 | `image_path`, `model_name`, `conf_threshold`, `iou_threshold`, `override_imgsz`  |
| `PredictionResult` | 推理结果值对象 | `image_id`, `model_name`, `bbox`(xywh), `score`, `category_id`, `inference_time` |

### [`interfaces.py`](src/domain/interfaces.py)

定义 `DetectorAdapter` 抽象基类（ABC），规定所有模型适配器必须实现的两个方法：

- `ensure_loaded()` — 确保模型已加载
- `infer(image_path, conf_threshold, iou_threshold, override_imgsz)` — 执行推理，返回统一格式的 `List[Dict]`

---

## 二、application/ — 应用层

> 业务编排层，依赖 domain 接口 + infrastructure 实现，不依赖具体 UI 框架。

### [`dto.py`](src/application/dto.py)

配置数据传输对象：

| 类                    | 用途                                                                   |
| --------------------- | ---------------------------------------------------------------------- |
| `ModelRuntimeConfig`  | 单个模型的运行时配置（名称、框架类型、权重路径、输入尺寸、默认阈值等） |
| `GlobalRuntimeConfig` | 全局配置（device、默认阈值、输出目录）                                 |
| `RuntimeConfig`       | 聚合 `global_config` + `models` 字典                                   |

### [`bootstrap.py`](src/application/bootstrap.py)

工厂函数 `build_predict_service(config_path: str) -> PredictService`：

1. 调用 `load_runtime_config()` 解析 YAML
2. 创建 `PredictService` 实例

所有入口（Web 后端、CLI 工具、脚本）都通过此函数获取推理服务。

### [`predict_service.py`](src/application/predict_service.py)

核心编排器 `PredictService`：

| 方法                                             | 功能                                   |
| ------------------------------------------------ | -------------------------------------- |
| `available_models()`                             | 返回已注册模型名列表                   |
| `predict(image_path, model_name, ...)`           | 单模型推理，返回 `List[Dict]`          |
| `predict_ensemble(image_path, model_names, ...)` | 多模型推理 + WBF 融合                  |
| `predict_by_request(request)`                    | 基于 `PredictRequest` 值对象的推理入口 |

内部流程：参数校验 → 阈值解析 → 适配器获取/加载 → 推理计时 → 结果归一化。

### [`fusion.py`](src/application/fusion.py)

`fuse_predictions(records, iou_threshold=0.55, min_votes=1)` — IoU-based Weighted Box Fusion：

1. 所有模型的预测框按置信度降序排列
2. 对每个框，检查与已有聚类簇的加权平均框 IoU ≥ 阈值
3. 匹配则加入该簇并重新计算加权平均框（以置信度为权重）
4. 不匹配则新建簇
5. 最终每个簇输出一个融合框，置信度取簇内均值

---

## 三、infrastructure/ — 基础设施层

> 依赖第三方框架的具体实现。

### [`config_loader.py`](src/infrastructure/config_loader.py)

`load_runtime_config(config_path: str) -> RuntimeConfig`：

- 读取 `configs/models.yaml`
- 解析 `global` 段 → `GlobalRuntimeConfig`
- 解析 `models` 列表 → `Dict[str, ModelRuntimeConfig]`
- 返回聚合的 `RuntimeConfig`

### [`adapter_factory.py`](src/infrastructure/adapter_factory.py)

`AdapterFactory` — 按 `framework_type` 创建并缓存适配器实例：

| framework_type | 适配器类        |
| -------------- | --------------- |
| `drenet`       | `DRENetAdapter` |
| `mmdetection`  | `MMDetAdapter`  |
| `ultralytics`  | `YOLOAdapter`   |

使用 `get_or_create()` 确保同一模型只加载一次（实例缓存）。

### [`visualization.py`](src/infrastructure/visualization.py)

`render_detections(image_path, predictions, output_path)` — 使用 PIL 在原图上绘制检测框：

- 按模型名 MD5 哈希分配颜色（同一模型颜色一致）
- 自适应线宽和字体大小
- 绘制 bbox + 标签（模型名:置信度）

---

## 四、infrastructure/adapters/ — 模型适配器

### [`base.py`](src/infrastructure/adapters/base.py)

`BaseAdapter(DetectorAdapter, ABC)` — 所有适配器的公共基类：

- `ensure_loaded()` — 懒加载守卫
- `load_model()` — 抽象方法，子类实现
- `infer()` — 抽象方法，子类实现
- `validate_image_path()` — 图片路径校验

### [`drenet_adapter.py`](src/infrastructure/adapters/drenet_adapter.py)（182 行）

`DRENetAdapter` — 支持两种集成模式：

| 模式       | config_path 格式                      | 说明                                    |
| ---------- | ------------------------------------- | --------------------------------------- |
| MMDet 兼容 | `path/to/config.py`                   | 复用 MMDetection 推理管线               |
| 自定义插件 | `/abs/path/to/plugin.py:factory_name` | 通过 `importlib` 动态加载外部 predictor |

### [`mmdet_adapter.py`](src/infrastructure/adapters/mmdet_adapter.py)（141 行）

`MMDetAdapter` — FCOS 等 MMDetection 模型的适配器：

- 处理 Torch 2.6 `weights_only` 兼容性
- 支持 `override_imgsz` 动态调整推理尺寸（通过临时修改 config 的 `test_pipeline`）
- 使用 `inference_detector` 执行推理

### [`yolo_adapter.py`](src/infrastructure/adapters/yolo_adapter.py)（61 行）

`YOLOAdapter` — Ultralytics YOLO 模型适配器：

- 加载 `.pt` 权重文件
- 调用 `model.predict()` 执行推理
- 支持 `override_imgsz` 动态调整推理尺寸

### [`parsing.py`](src/infrastructure/adapters/parsing.py)（134 行）

预测结果归一化工具：

| 函数                              | 用途                                                           |
| --------------------------------- | -------------------------------------------------------------- |
| `xyxy_to_xywh()`                  | 坐标格式转换                                                   |
| `clamp_score()`                   | 置信度裁剪到 [0,1]                                             |
| `normalize_record()`              | 单条记录归一化 + 阈值过滤                                      |
| `normalize_raw_predictions()`     | 批量归一化（支持 list[dict] / dict / list[list] 三种输入格式） |
| `rows_from_mmdet_result()`        | MMDetection 输出 → 统一格式                                    |
| `rows_from_ultralytics_results()` | Ultralytics 输出 → 统一格式                                    |

---

## 数据流全景

```
configs/models.yaml
       │
       ▼
config_loader.load_runtime_config()
       │
       ▼
bootstrap.build_predict_service()
       │
       ▼
PredictService.predict() / predict_ensemble()
       │
       ├─► AdapterFactory.get_or_create()
       │       │
       │       ▼
       │   DRENetAdapter / MMDetAdapter / YOLOAdapter
       │       │
       │       ▼
       │   parsing.normalize_raw_predictions()
       │
       ▼
  (ensemble 模式) fusion.fuse_predictions()
       │
       ▼
  List[Dict] 统一输出 → visualization.render_detections()
```
