# Generated Figures Manifest

本目录仅保留**当前论文正文实际引用**的图片文件。

口径说明：
- 本目录中的图片是论文正文最终引用的成稿资产，而不是全部原始样例。
- 第 4 章三模型定性图当前统一采用自动生成的 `3x2` 拼板图，服务于论文成稿展示。
- FCOS 的原始定性样例仍保留在 `assets/figures/qualitative/fcos/`，其中论文主对比口径与系统/消融口径分开管理；本目录不再重复区分两套原始样例。

## 1) 论文在用图片清单（当前）

- `ch1_pipeline.png`
- `ch3_dataset_examples.png`
- `ch3_drenet_overview.png`
- `ch3_fcos_pipeline.png`
- `ch3_yolo_pipeline.png`
- `ch4_success_cases.png`
- `ch4_miss_cases.png`
- `ch4_false_positive_cases.png`
- `ch4_main_comparison_chart.png`
- `ch4_threshold_trend.png`
- `ch4_size_trend.png`
- `ch5_architecture.png`
- `ch5_er.png`
- `ch5_flow.png`
- `ch5_pages.png`

## 2) 每张图的生成脚本/来源标准

### `ch3_dataset_examples.png`
- 用途：第3章数据集样例图（3x2）
- 生成脚本：`/Users/khs/codes/graduation_project/tools/build_ch3_dataset_examples_3x2.py`
- 生成命令：
  - `cd /Users/khs/codes/graduation_project`
  - `.venv/bin/python tools/build_ch3_dataset_examples_3x2.py`
- 备注：脚本会读取 `experiment_assets/datasets/LEVIR-Ship/test/images` 与 `labels` 自动叠框并输出到本目录。

### `ch4_success_cases.png` / `ch4_miss_cases.png` / `ch4_false_positive_cases.png`
- 用途：第4章三模型定性对比图（3列x2行，列=模型，行=Label/Prediction）
- 生成脚本：`/Users/khs/codes/graduation_project/tools/build_ch4_panels_aligned.py`
- 生成命令：
  - `cd /Users/khs/codes/graduation_project`
  - `.venv/bin/python tools/build_ch4_panels_aligned.py`
- 备注：脚本会从三模型 `predictions.json` 自动选样并渲染。

### `ch3_drenet_overview.png` / `ch3_fcos_pipeline.png` / `ch3_yolo_pipeline.png`
- 用途：第3章三模型方法示意图
- 生成脚本：`/Users/khs/codes/graduation_project/tools/build_thesis_method_and_metric_figures.py`
- 生成命令：
  - `cd /Users/khs/codes/graduation_project`
  - `.venv/bin/python tools/build_thesis_method_and_metric_figures.py`
- 备注：脚本采用黑白灰论文框图风格，统一使用衬线字体、细线箭头与自绘流程结构，不直接复用原论文插图。

### `ch4_main_comparison_chart.png` / `ch4_threshold_trend.png` / `ch4_size_trend.png`
- 用途：第4章主对比与消融趋势图
- 生成脚本：`/Users/khs/codes/graduation_project/tools/build_thesis_method_and_metric_figures.py`
- 生成命令：
  - `cd /Users/khs/codes/graduation_project`
  - `.venv/bin/python tools/build_thesis_method_and_metric_figures.py`
- 备注：主对比图使用正文主表口径；阈值与尺寸趋势图使用 `experiment_assets/ablation/` 对应结果回填。

### `ch1_pipeline.png`
- 用途：第1章技术路线图
- 生成脚本：`/Users/khs/codes/graduation_project/tools/build_thesis_method_and_metric_figures.py`
- 生成命令：
  - `cd /Users/khs/codes/graduation_project`
  - `.venv/bin/python tools/build_thesis_method_and_metric_figures.py`
- 备注：当前已统一改为黑白灰论文框图风格，技术路线图与第3章方法图保持一致。

### `ch5_architecture.png`
- 用途：第5章系统总体架构图
- 生成脚本：`/Users/khs/codes/graduation_project/tools/build_thesis_method_and_metric_figures.py`
- 生成命令：
  - `cd /Users/khs/codes/graduation_project`
  - `.venv/bin/python tools/build_thesis_method_and_metric_figures.py`
- 备注：脚本采用与第1章、第3章一致的分层框图风格，和 `docs/system/architecture.md` 的系统语义保持一致。

### `ch5_er.png`
- 用途：第5章数据库 ER 图
- 生成脚本：`/Users/khs/codes/graduation_project/tools/build_thesis_method_and_metric_figures.py`
- 生成命令：
  - `cd /Users/khs/codes/graduation_project`
  - `.venv/bin/python tools/build_thesis_method_and_metric_figures.py`
- 备注：脚本将数据库表结构转为论文中的静态实体关系图，和 `docs/system/database_er.md` 保持一致。

### `ch5_flow.png`
- 用途：第5章系统关键流程图
- 生成脚本：`/Users/khs/codes/graduation_project/tools/build_thesis_method_and_metric_figures.py`
- 生成命令：
  - `cd /Users/khs/codes/graduation_project`
  - `.venv/bin/python tools/build_thesis_method_and_metric_figures.py`
- 备注：脚本输出论文化的单列流程框图，与 `docs/system/architecture.mmd` 中的关键流程语义一致。

### `ch5_pages.png`
- 用途：第5章系统页面三联图
- 当前来源：固定成稿资产（由稳定页面截图拼板后定稿）
- 脚本状态：`N/A`（仓库内暂无对应自动生成脚本）

## 3) 更新与清理规则

- 仅保留在 `thesis_overleaf/chapters/*.tex` 中通过 `\includegraphics{generated/...}` 实际引用的图片。
- 中间产物（例如以 `_` 开头的临时图、早期调参图）不应保留在本目录。
- 如新增图片，需同时在本文件补充：
  - 输出文件名
  - 对应章节用途
  - 生成脚本路径
  - 一键生成命令
  - 是否为固定成稿资产（若非脚本化）
