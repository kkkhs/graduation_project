# Overleaf 模板写作约束清单（V1）

更新时间：2026-03-15

## 1. 当前状态
- 目标模板目录约定：`/Users/khs/codes/graduation_project/thesis_overleaf/`
- 当前仓库尚未检测到 `.tex/.cls/.bib` 模板文件，故本清单先输出“通用约束 + 待锁定项”。
- 已准备可直接粘贴的第1-3章正文初稿：`docs/thesis/drafts_v1/`

## 2. 已锁定约束（先行执行）
- 章节命名：中文学术风格，使用“问题-方法-证据-小结”结构。
- 引用策略：首轮使用占位 key，统一格式 `author_year_topic`。
- 图表标签：
  - 图：`fig:chX_xxx`
  - 表：`tab:chX_xxx`
- 术语统一：
  - 遥感微小船舶检测
  - LEVIR-Ship
  - AP50 / AP50-95 / Precision / Recall / F1
  - DRENet / FCOS / YOLO26

## 3. 待模板同步后锁定项
- `main.tex` 入口文件与章节 include 方式（`\input` 或 `\include`）。
- 正文层级命令（`\chapter` / `\section` 的具体层级策略）。
- 参考文献命令与样式（BibTeX/BibLaTeX，`\cite`/`\parencite` 等）。
- 图表标题格式（中文/英文、标题位置、caption 样式）。
- 学校模板对摘要、关键词、符号表、致谢、附录的专用环境要求。

## 4. 模板接入后执行动作
1. 扫描 `main.tex` 与章节入口。
2. 对 `docs/thesis/drafts_v1/*.tex` 做一次语法映射（命令与环境对齐模板）。
3. 合并进模板章节文件并进行编译自检。
4. 输出 V2“模板已锁定版本”清单。
