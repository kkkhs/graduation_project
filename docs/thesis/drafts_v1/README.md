# 第1-3章初稿（V1）使用说明

## 文件列表
- `chapter1_background.tex`
- `chapter2_related_work.tex`
- `chapter3_dataset_metric_method.tex`

## 放入 Overleaf 模板的方式
1. 用 VSCode Overleaf Workshop 把项目同步到本地目录（建议：`thesis_overleaf/`）。
2. 找到模板中的章节目录（常见为 `chapters/`、`body/` 或 `contents/`）。
3. 将本目录中的三个文件内容分别粘贴到对应章节文件。
4. 若模板使用 `\input{...}` 或 `\include{...}`，确认章节入口顺序为第1章到第3章。

## 首轮引用策略
- 使用占位 key，详见：`docs/thesis/citation_keys_v1.md`。
- 第二轮统一补齐 `.bib`。

## 术语与格式约束
- 模板约束清单：`docs/thesis/template_constraints_checklist.md`
- 逐章定稿流程：`docs/thesis/review_workflow_v1.md`
