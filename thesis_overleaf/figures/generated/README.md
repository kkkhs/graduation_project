# Generated Figures Manifest

本目录仅保留**当前论文正文实际引用**的图片文件。

口径说明：
- 本目录中的图片是论文正文最终引用的成稿资产，而不是全部原始样例。
- 第 4 章三模型定性图当前统一采用自动生成的 `3x2` 拼板图，服务于论文成稿展示。
- FCOS 的原始定性样例仍保留在 `assets/figures/qualitative/fcos/`，其中论文主对比口径与系统/消融口径分开管理；本目录不再重复区分两套原始样例。

## 1) 论文在用图片清单（当前）

- `ch1_pipeline.png`
- `ch3_dataset_examples.png`
- `ch4_success_cases.png`
- `ch4_miss_cases.png`
- `ch4_false_positive_cases.png`
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

### `ch1_pipeline.png`
- 用途：第1章技术路线图
- 当前来源：固定成稿资产（手工整理后保留）
- 脚本状态：`N/A`（仓库内暂无对应自动生成脚本）
- 建议：若后续需要脚本化，建议补一个基于 Mermaid 的统一导出脚本。

### `ch5_architecture.png`
- 用途：第5章系统总体架构图
- 当前来源：固定成稿资产（与 `docs/system/architecture.md` 中 Web 架构语义一致）
- 脚本状态：`N/A`（仓库内暂无对应自动生成脚本）

### `ch5_er.png`
- 用途：第5章数据库 ER 图
- 当前来源：固定成稿资产（与 `docs/system/database_er.md` 语义一致）
- 脚本状态：`N/A`（仓库内暂无对应自动生成脚本）

### `ch5_flow.png`
- 用途：第5章系统关键流程图
- 当前来源：固定成稿资产（与 `docs/system/architecture.mmd` 流程语义一致）
- 脚本状态：`N/A`（仓库内暂无对应自动生成脚本）

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
