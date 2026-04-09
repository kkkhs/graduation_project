# 仓库目录说明（项目导航）

本文档用于快速说明仓库各目录职责，重点回答“该看哪里、哪些可忽略、哪些是交付核心”。

## 核心交付目录

- `backend/`：Web 后端（FastAPI、任务调度、SQLite、API 实现）
- `frontend/`：Web 前端（React + Vite + TS）
- `src/`：算法与推理内核（适配器、融合、应用层服务）
- `thesis_overleaf/`：毕业论文 LaTeX 工程
- `docs/`：实验、系统、论文与答辩文档

## 工程支撑目录

- `tests/`：Python 自动化测试（见 `tests/README.md`）
- `tools/`：脚本工具入口（推理、可视化、数据处理、插件）
- `scripts/`：启动与运维脚本（数据库初始化、前后端启动、一键开发）
- `configs/`：模型与系统配置（本地路径由用户按环境维护）

## 资产与产物目录

- `experiment_assets/`：训练产物、实验配置、运行日志归档
- `assets/`：长期保留图件（论文/答辩素材）
- `outputs/`：运行时输出（任务结果、可视化、临时截图）
- `pdfs/`：论文 PDF 与翻译相关 PDF 资料

## 历史与辅助目录

- `experiments/`：历史实验代码与复现材料
- `patches/`：临时补丁文件（用于特定环境修复）
- `images/`：预留图片目录（当前使用较少）

## 建议阅读路径

1. 先看仓库总览：`README.md`
2. 再看系统文档入口：`docs/system/README.md`
3. 再看实验结果入口：`docs/results/README.md`
4. 论文相关看：`thesis_overleaf/README.md`
5. 需要跑回归测试看：`tests/README.md`

## 常见疑问

### 为什么同时有 `backend/` 和 `src/`

- `backend/` 负责 HTTP 服务、数据库、任务调度、文件路由。
- `src/` 负责模型推理能力本身（可被后端和脚本复用）。
- 两者是“服务层”和“能力层”的关系，不是重复目录。

### 哪些目录答辩时重点展示

建议优先展示：

- `backend/`
- `frontend/`
- `src/`
- `tests/`
- `docs/results/`
- `thesis_overleaf/`

