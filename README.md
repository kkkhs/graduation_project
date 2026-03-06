# 毕设项目总览

题目：《遥感场景下的微小船舶检测系统设计与实现》

这个仓库不是对外产品仓库，而是我自己做毕设时用的工作台。  
它的目标很明确：把“论文 + 实验 + 系统 Demo”三件事放在一个仓库里推进，并且保证每一步都能回填到论文和答辩材料。

## 1. 当前项目在做什么

当前主线已经从“开题准备”切到“实验落地”：
- 第一优先级：训练机跑通 DRENet 首轮训练 / 评测 / 可视化
- 第二优先级：扩展到 MMDetection 和 YOLO，形成三模型对比
- 第三优先级：把实测结果回填到论文、图表和系统演示材料

当前训练机计划支持两类环境：
- Windows 3060 笔记本
- Linux 云 GPU（例如 AutoDL）

## 2. Done / Todo

### 已完成
- 开题相关主体材料已完成，基础文献整理框架已建立
- 实验规范、训练手册、结果模板、日志模板已落地
- 三模型统一推理骨架已完成
- 融合推理、可视化输出、Qt 桌面 UI 已完成
- 单元测试已补齐，当前系统骨架可运行
- 云端与本地两套训练文档都已建立

### 还没完成
- LEVIR-Ship 在训练机上的实际下载、转换、统计落盘
- DRENet 首轮正式训练 / 评测 / 可视化结果
- MMDet 与 YOLO 的正式结果
- 主对比表、消融表、定性分析表的真实数据回填
- 论文第四章、第五章用实测结果补齐
- 系统最终演示样例整理

### 当前最该做
1. 选定训练环境：3060 或 AutoDL
2. 按对应执行手册完成环境预检、数据预检
3. 先跑 DRENet 首轮
4. 结果回传后更新 `docs/results/` 和论文材料

## 3. Todo 看板入口

总看板在：
- [spec_todo.md](/Users/khs/codes/graduation_project/docs/spec_todo.md)

这份文件是当前最重要的进度来源。  
判断“现在该做什么”，优先看它，不优先看零散笔记。

对应关系：
- `B`：数据集与评价指标
- `C`：DRENet 复现
- `D`：三模型对比实验
- `E`：系统实现
- `F`：论文写作回填
- `G`：训练机跑实验时，本机可以并行做的事

## 4. 目录说明

### 核心目录
- `docs/`
  - 项目文档中心。论文、实验、系统设计、答辩材料说明都在这里。
- `src/`
  - 系统代码主体，已经按分层方式组织。
- `tools/`
  - 直接可运行的入口脚本，比如推理、可视化、桌面 UI。
- `configs/`
  - 模型注册与统一推理配置。
- `tests/`
  - 单元测试与兼容性测试。
- `scripts/`
  - 辅助脚本，比如结果回传。
- `assets/`
  - 适合长期保留的图片、图表、论文插图素材。

### 文档目录
- `docs/experiments/`
  - 训练和评测相关手册、日志、执行计划。
- `docs/results/`
  - 论文第四章最直接的来源：主对比、消融、定性分析。
- `docs/system/`
  - 系统需求、架构、接口、集成规范。
- `docs/literature/`
  - 文献阅读、翻译、文献表格。
- `docs/kaiti/`
  - 开题报告、开题答辩相关材料。
- `docs/thesis/`
  - 论文大纲与章节材料映射。
- `docs/ops/`
  - 本地运行问题和运维说明。

### 运行与产物目录
- `outputs/`
  - 运行产物目录。主要是推理 JSON、可视化图片、UI 导出结果。
  - 现在已忽略，不作为版本库长期内容。
- `pdfs/`
  - 论文 PDF 与相关参考材料。

## 5. 重要文档入口

### 总控文档
- [spec.md](/Users/khs/codes/graduation_project/docs/spec.md)
  - 毕设总体任务说明，偏“全局规划”。
- [spec_todo.md](/Users/khs/codes/graduation_project/docs/spec_todo.md)
  - 当前进度看板，偏“现在做什么”。

### 实验执行
- [experiments/README.md](/Users/khs/codes/graduation_project/docs/experiments/README.md)
  - 实验总规范，总入口。
- [3060_execution_playbook.md](/Users/khs/codes/graduation_project/docs/experiments/3060_execution_playbook.md)
  - Windows 3060 训练机手册。
- [cloud_execution_playbook.md](/Users/khs/codes/graduation_project/docs/experiments/cloud_execution_playbook.md)
  - Linux 云 GPU 手册，适合 AutoDL。
- [drenet_reproduction_guide.md](/Users/khs/codes/graduation_project/docs/experiments/drenet_reproduction_guide.md)
  - DRENet 专项复现手册。
- [exp_log_template.md](/Users/khs/codes/graduation_project/docs/experiments/exp_log_template.md)
  - 每次实验记录模板。

### 结果回填
- [baselines.md](/Users/khs/codes/graduation_project/docs/results/baselines.md)
  - 三模型主对比结果表。
- [ablation.md](/Users/khs/codes/graduation_project/docs/results/ablation.md)
  - 消融实验记录。
- [qualitative.md](/Users/khs/codes/graduation_project/docs/results/qualitative.md)
  - 成功/误检/漏检/难例分析。

### 系统实现
- [architecture.md](/Users/khs/codes/graduation_project/docs/system/architecture.md)
  - 系统架构说明。
- [predict_api_contract.md](/Users/khs/codes/graduation_project/docs/system/predict_api_contract.md)
  - 统一推理接口契约。
- [artifact_integration_spec.md](/Users/khs/codes/graduation_project/docs/system/artifact_integration_spec.md)
  - 训练产物接入规范。
- [result_sync_flow.md](/Users/khs/codes/graduation_project/docs/system/result_sync_flow.md)
  - 训练结果回传流程。

### 开题与论文
- [kaiti.md](/Users/khs/codes/graduation_project/docs/kaiti/kaiti.md)
  - 开题阶段任务说明。
- [outline.md](/Users/khs/codes/graduation_project/docs/thesis/outline.md)
  - 论文大纲。

## 6. 代码目录作用

### `src/`
- `src/application/`
  - 业务流程层，负责预测服务、融合逻辑、DTO。
- `src/domain/`
  - 领域层，定义实体和接口。
- `src/infrastructure/`
  - 基础设施层，负责适配器工厂、配置读取、可视化等。
- `src/adapters/`
  - 兼容导入层/适配层。
- `src/core/`
  - 预测器、配置、注册表等底层核心逻辑。
- `src/contracts/`
  - 输出 JSON schema 等契约文件。

### `tools/`
- `run_predict.py`
  - 统一推理命令行入口。
- `visualize_predict.py`
  - 推理并输出可视化结果。
- `desktop_ui_qt.py`
  - 桌面 UI。
- `convert_yolo_to_coco.py`
  - 数据格式转换脚本。
- `drenet_plugin_template.py`
  - DRENet 接入模板。

## 7. 我现在该怎么用这个仓库

如果当前目标是跑实验，建议只按这条链路走：

1. 看 [spec_todo.md](/Users/khs/codes/graduation_project/docs/spec_todo.md)
   - 确认当前阶段仍然是 `B/C/D/G3`
2. 选训练环境
   - AutoDL：看 [cloud_execution_playbook.md](/Users/khs/codes/graduation_project/docs/experiments/cloud_execution_playbook.md)
   - 3060：看 [3060_execution_playbook.md](/Users/khs/codes/graduation_project/docs/experiments/3060_execution_playbook.md)
3. 先跑 DRENet 首轮
4. 结果回传后更新：
   - [baselines.md](/Users/khs/codes/graduation_project/docs/results/baselines.md)
   - [qualitative.md](/Users/khs/codes/graduation_project/docs/results/qualitative.md)
5. 再开始 MMDet / YOLO

如果当前目标是改系统，不先碰训练：
- 看 `src/` + `tools/`
- 先跑测试：

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -p "test_*.py"
```

## 8. 当前默认约定

- 主数据集：LEVIR-Ship
- 主基线：DRENet
- 目标：至少 3 个模型对比
- 主指标：`AP50`
- 辅助指标：`Precision / Recall / F1`
- 训练产物应最终统一整理到 `artifacts/`
- 可长期保留的图片放 `assets/figures/`
- `outputs/` 只是临时运行结果，不作为正式交付目录

## 9. 当前下一步

- 如果走 AutoDL：先补一份 AutoDL 专用手册，或直接按云端手册改路径执行
- 先完成 DRENet 首轮训练、评测、可视化
- 然后回填结果表和日志
- 再进入三模型主对比阶段
