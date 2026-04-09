# Spec TODO（执行看板）

> 说明：
> - `[x]` 本机已完成（文档/模板/脚本已落地）
> - `[ ]` 需在 3060 笔记本或云 GPU 执行
> - 时间基准：2026-03-18

## A. 文献调研与开题
- [x] A1 确定调研范围与关键词
- [x] A2 建立文献清单与分类模板（`docs/literature/literature_table.md`）
- [x] A3 完成开题报告主体（`docs/kaiti/毕设开题报告-正文.docx`）

## B. 数据集与评价指标
- [x] B4 数据文档模板与执行说明已准备（见实验手册）
- [x] B4 在训练机完成 LEVIR-Ship 下载、结构核验和统计落盘（云端链路已验证可用）
- [x] B5 数据统一方案已明确（COCO 中间格式）
- [x] B5 在训练机完成实际转换脚本与产物（2026-03-09 已生成 `annotations/{train,val,test}.json` 并核验）
- [x] B6 指标口径已统一（AP50/Precision/Recall/F1）
- [x] B6 在训练机跑通评测脚本并产出首版结果（DRENet：AP50=0.7949，AP50:95=0.2919）

## C. DRENet 复现
- [x] C7 复现文档已完成（`docs/experiments/archive/drenet_reproduction_guide.md`）
- [x] C7 在训练机完成 DRENet 训练/测试/可视化首轮结果（cloud formal run 到 299/299）
- [x] C8 实验日志模板已完成（`docs/experiments/formal_experiment_log_template_v1.md`）
- [x] C8 每次实验填报日志到 `docs/experiments/logs/`（已沉淀 2026-03-07~03-08 关键日志）

## D. 三模型对比实验
- [x] D9 模型1：DRENet 正式结果
- [x] D10 模型2：mmdetection（FCOS）历史正式结果（`lvxs9xhk`，120/120，last AP50=0.770，AP50:95=0.285，best AP50=0.798@14；实际口径 `1333x800`）
- [x] D10 FCOS `512x512` 统一重训尝试已完成，但两次新 run 数值失稳，最终保留历史默认口径结果作为参考基线
- [x] D11 模型3：YOLO26 首轮正式结果（EarlyStopping at 263/300，best epoch=186）
- [x] D12 主对比表模板已完成（`docs/results/baselines.md`）
- [x] D12 填入三模型实测结果并形成论文可用结论（其中 FCOS 以历史默认口径作为参考基线）
- [x] D13 消融表模板已完成（`docs/results/ablation.md`）
- [x] D13 完成消融结果回填：
  - 三模型统一阈值消融
  - FCOS/YOLO26 输入尺寸敏感性分析
  - DRENet 尺寸限制说明留痕
- [x] D14 定性结果模板已完成（`docs/results/qualitative.md`）
- [x] D14 整理成功/漏检/误检难例图到 `assets/figures/`（DRENet/YOLO/FCOS 已补齐，含 FCOS global-best 与 stable 口径）

## E. 系统需求、设计与实现
- [x] E15 需求文稿已在开题稿中固定
- [x] E16 系统架构图代码已准备（Mermaid/PlantUML）
- [x] E17 系统 MVP 代码实现（统一 `predict(...)` 接口，含 Web 主链路与异步任务闭环）
- [x] E17-本机骨架已完成（配置读取、Adapter 路由、统一输出，支持真实推理接入）
- [x] E17-可视化界面原型已完成（Python 桌面 UI、模型切换、可视化输出、JSON 展示）
- [x] E17-纯 Python 可视化入口已完成（命令行 + matplotlib + 文件输出）
- [x] E17-纯 Python 桌面 UI 已完成（Qt 交互界面）
- [x] E17-多层可维护架构已落地（presentation/application/domain/infrastructure）
- [x] E17-三模型综合推理已接入（ensemble_all + 结果融合）
- [x] E17-单元/回归测试已补齐（fusion、参数校验、兼容导入）
- [x] E18 演示用例准备（易/中/难三类样例已固定到 `assets/demo_cases/`）

## F. 论文写作与答辩材料
- [x] F19 论文大纲与材料映射已具备基础
- [x] F19 用实测结果回填第四章与第五章内容（已形成可交初稿，持续润色中）

## G. 最终收口清单（当前仍需保留的真实缺口）

- [x] G1 固定演示样例池已落地（`assets/demo_cases/{easy,medium,hard}` + `docs/system/demo_case_catalog.md`）
- [ ] G2 统一参考文献格式与最终 citation key 检查
- [x] G3 FCOS 特殊权重口径说明已收口：仅通过文档明确论文口径与系统/消融口径边界，不改 DRENet/YOLO 配置
- [ ] G4 论文最终提交检查：封面、目录、图表分页、参考文献与模板页最终肉眼核验
- [x] G5 系统最终交付检查已留档：启动命令、主链路、测试状态、前端回归与兜底截图已整理

---

## 已在本机落地的关键文件
- 实验总手册：`docs/experiments/README.md`
- 3060一步一步：`docs/experiments/3060_execution_playbook.md`
- 实验记录模板：`docs/experiments/formal_experiment_log_template_v1.md`
- 计划目录：`docs/experiments/plans/`
- 主对比模板：`docs/results/baselines.md`
- 消融模板：`docs/results/ablation.md`
- 定性分析模板：`docs/results/qualitative.md`
- 结果回传脚本：`scripts/sync_results_from_laptop.sh`
- 推理配置样例：`configs/models.example.yaml`
- 结果 schema：`src/contracts/result_schema.json`
- 接口契约文档：`docs/system/predict_api_contract.md`
- 产物接入规范：`docs/system/artifact_integration_spec.md`
- 回传落盘流程：`docs/system/result_sync_flow.md`
- MVP 代码骨架：`src/application/`、`src/domain/`、`src/infrastructure/`、`tools/run_predict.py`
- 本机运行说明：`docs/system/local_mvp_run.md`
- 本机问题记录：`docs/ops/local_run_issues.md`
- Python 可视化入口：`tools/visualize_predict.py`
- Python 可视化说明：`docs/system/python_visualization_guide.md`
- 桌面 UI 说明：`docs/system/desktop_ui_guide.md`
- Qt 桌面 UI（推荐）：`tools/desktop_ui_qt.py`
- 架构说明：`docs/system/architecture.md`
- 架构图：`docs/system/architecture.mmd`

## 下一步（你现在就做）
1. 完成参考文献格式与 citation key 的最后一次统一检查。
2. 完成 FCOS 特殊权重口径的文档收口，确保论文、结果文档与系统文档一致。
3. 按最终提交清单逐项检查论文 PDF 与系统交付材料。
