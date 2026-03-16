# 配置目录

本目录用于保存系统与模型的配置文件。

## 关键文件
- `models.yaml`：系统运行时实际读取的模型注册表。
- `models.example.yaml`：模板配置，需复制为 `models.yaml` 并填写本机路径。

## 使用建议
1. 修改 `weight_path` 与 `config_path` 为本机真实路径。
2. 确保模型 key 与系统一致（`drenet / mmdet_fcos / yolo`）。
