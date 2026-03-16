# 定性结果整理

> 目标：把现有可复用的可视化结果按模型归档，直接服务论文中的“成功检测 / 漏检 / 误检”分析。

## 1. 归档规则
- 一级按模型分类：`DRENet`、`YOLO26`、`FCOS`
- 二级按现象分类：`success`、`miss`、`false_positive`
- 当前统一落点：`assets/figures/qualitative/<model>/<category>/`

## 2. 当前可直接用于论文的样例

### 2.1 DRENet
| 模型 | 场景类型 | 图像路径 | 现象 | 论文可用结论 |
|---|---|---|---|---|
| DRENet | success | `assets/figures/qualitative/drenet/success/drenet_success_case_01.jpg` | 预测框与标注框基本重合，目标位置准确 | 说明 DRENet 在低纹理海面小目标场景下有较稳定的检出能力 |
| DRENet | miss | `assets/figures/qualitative/drenet/miss/drenet_miss_case_01.jpg` | 图中存在标注目标，但预测结果覆盖不完整 | 说明 DRENet 在稀疏小目标或局部低对比区域仍存在漏检风险 |
| DRENet | false_positive | `assets/figures/qualitative/drenet/false_positive/drenet_false_positive_case_01.jpg` | 近岸复杂背景中出现预测框，存在将岸线/近岸纹理误判为目标的风险 | 说明 DRENet 在复杂背景与岸线干扰场景下仍可能产生误检 |

### 2.2 YOLO26
| 模型 | 场景类型 | 图像路径 | 现象 | 论文可用结论 |
|---|---|---|---|---|
| YOLO26 | success | `assets/figures/qualitative/yolo/success/yolo_success_case_01.jpg` | 多个测试拼图块中预测结果与标注基本一致 | 说明 YOLO26 在主流测试样本上具有较好的整体检出稳定性 |
| YOLO26 | miss | `assets/figures/qualitative/yolo/miss/yolo_miss_case_01_labels.jpg`、`assets/figures/qualitative/yolo/miss/yolo_miss_case_01_pred.jpg` | 对照 labels/pred 可见部分标注目标未被输出 | 说明 YOLO26 在局部弱目标场景下存在漏检 |
| YOLO26 | false_positive | `assets/figures/qualitative/yolo/false_positive/yolo_false_positive_case_01_labels.jpg`、`assets/figures/qualitative/yolo/false_positive/yolo_false_positive_case_01_pred.jpg` | 对照 labels/pred 可见部分预测框数量多于标注目标 | 说明 YOLO26 在纹理复杂区域可能产生额外框或重复框 |

### 2.3 FCOS
| 模型 | 场景类型 | 图像路径 | 现象 | 说明 |
|---|---|---|---|---|
| FCOS (global-best@ep14) | success | `assets/figures/qualitative/fcos/global_best/success/fcos_global_best_success_01.jpg` | 预测框与标注框一致 | 论文口径代表图（global-best） |
| FCOS (global-best@ep14) | miss | `assets/figures/qualitative/fcos/global_best/miss/fcos_global_best_miss_01.jpg` | 标注存在但未完全检出 | 论文口径误差样例 |
| FCOS (global-best@ep14) | false_positive | `assets/figures/qualitative/fcos/global_best/false_positive/fcos_global_best_false_positive_01.jpg` | 存在多余预测框 | 论文口径误检样例 |
| FCOS (stable@ep102) | success | `assets/figures/qualitative/fcos/stable/success/fcos_stable_success_01.jpg` | 预测与标注一致 | 系统口径代表图（stable） |
| FCOS (stable@ep102) | miss | `assets/figures/qualitative/fcos/stable/miss/fcos_stable_miss_01.jpg` | 标注存在但漏检 | 系统口径误差样例 |
| FCOS (stable@ep102) | false_positive | `assets/figures/qualitative/fcos/stable/false_positive/fcos_stable_false_positive_01.jpg` | 存在误检框 | 系统口径误检样例 |

## 3. 当前目录结构
```text
assets/figures/qualitative/
├── drenet/
│   ├── success/
│   ├── miss/
│   └── false_positive/
├── yolo/
│   ├── success/
│   ├── miss/
│   └── false_positive/
└── fcos/
    ├── global_best/
    │   ├── success/
    │   ├── miss/
    │   └── false_positive/
    └── stable/
        ├── success/
        ├── miss/
        └── false_positive/
```

## 4. 论文插图顺序清单
1. DRENet success：`assets/figures/qualitative/drenet/success/drenet_success_case_01.jpg`
2. DRENet miss：`assets/figures/qualitative/drenet/miss/drenet_miss_case_01.jpg`
3. DRENet false positive：`assets/figures/qualitative/drenet/false_positive/drenet_false_positive_case_01.jpg`
4. YOLO26 success：`assets/figures/qualitative/yolo/success/yolo_success_case_01.jpg`
5. YOLO26 miss：`assets/figures/qualitative/yolo/miss/yolo_miss_case_01_pred.jpg`（配合 labels 说明）
6. YOLO26 false positive：`assets/figures/qualitative/yolo/false_positive/yolo_false_positive_case_01_pred.jpg`（配合 labels 说明）
7. FCOS global-best success：`assets/figures/qualitative/fcos/global_best/success/fcos_global_best_success_01.jpg`
8. FCOS global-best miss：`assets/figures/qualitative/fcos/global_best/miss/fcos_global_best_miss_01.jpg`
9. FCOS global-best false positive：`assets/figures/qualitative/fcos/global_best/false_positive/fcos_global_best_false_positive_01.jpg`
10. FCOS stable success（系统口径）：`assets/figures/qualitative/fcos/stable/success/fcos_stable_success_01.jpg`

## 5. 写论文时的使用建议
- 主文优先放 DRENet 与 YOLO26 的 success/miss，说明主方法与轻量基线的实际表现。
- FCOS 作为 anchor-free 参考基线，放在对比段落或补充说明中，强调其参考意义而非严格同口径。
- stable 样例用于系统章节或“系统默认权重”说明，不与 global-best 混为同一口径。

## 6. 当前缺口
- 若论文版面允许，建议再补 1 组“同场景不同模型对照图”。

## 7. FCOS 图像获取方式
若后续需要新增 FCOS 样例，建议直接在已有 MMDetection 环境的机器上执行单模型导出，而不是重新训练。最短路径如下：

```bash
cd /Users/khs/codes/graduation_project
PYTHONPATH=. python3 tools/visualize_predict.py \
  --config configs/models.yaml \
  --image /absolute/path/to/test_image.png \
  --mode single \
  --model mmdet_fcos \
  --vis-out outputs/visualizations/fcos_vis_result.jpg \
  --json-out outputs/predictions/fcos_pred_result.json
```

前提：
- `configs/models.yaml` 中启用 `mmdet_fcos`
- `weight_path` 指向 `experiment_assets/checkpoints/mmdet/fcos_main_fixedcfg_20260315_160824_global_best_ep14.pth`
- `config_path` 指向 FCOS 正式配置文件
- 运行环境已安装 `mmengine + mmcv + mmdet`
