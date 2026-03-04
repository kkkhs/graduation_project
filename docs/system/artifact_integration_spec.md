# 训练产物接入规范（可机器执行）

## 1. 目标
统一训练产物命名、目录和配置更新方式，确保结果可追溯、可复现、可回填论文。

## 2. 命名规范

### 2.1 权重文件
```text
{model}_{dataset}_{input}_{metric}-{value}_{date}.{ext}
```
示例：
- `drenet_levirship_640_ap50-0.721_20260310.pth`
- `mmdet_fcos_levirship_640_ap50-0.703_20260312.pth`
- `yolo_levirship_640_ap50-0.695_20260313.pt`

### 2.2 指标文件
```text
metrics_{model}_{exp_id}.json
```
示例：`metrics_yolo_exp-20260312-01-yolo.json`

### 2.3 可视化文件
```text
{model}_{exp_id}_{case_type}_{image_id}.jpg
```
`case_type`: `success | false_positive | miss | hard_scene`

## 3. 目录规范
```text
artifacts/
  drenet/
    weights/
    logs/
    metrics/
    figures/
  mmdet/
    weights/
    logs/
    metrics/
    figures/
  yolo/
    weights/
    logs/
    metrics/
    figures/
```

## 4. `configs/models.yaml` 更新规则
每次接入新权重，只允许更新以下字段：
- `weight_path`
- `config_path`
- `input_size`
- `default_conf_threshold`
- `default_iou_threshold`

禁止在接入阶段改动接口层代码。

## 5. 版本追踪字段（必须记录）
每次接入在实验日志中追加：
- `experiment_id`
- `git_commit`
- `dataset_version`
- `model_name`
- `old_weight`
- `new_weight`
- `updated_at`

## 6. 最小交付清单
每个模型至少提交：
1. 最优权重（best）
2. 训练配置
3. 指标文件（AP50/P/R/F1）
4. 推理可视化样例（>= 3 张）
5. 实验日志

## 7. 接入后的回填动作（固定顺序）
1. `docs/results/baselines.md`
2. `docs/results/qualitative.md`
3. `docs/results/ablation.md`
