# Spec TODO（执行看板）

> 说明：
> - `[x]` 本机已完成（文档/模板/脚本已落地）
> - `[ ]` 需在 3060 笔记本或云 GPU 执行
> - 时间基准：2026-03-08

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
- [x] D10 模型2：mmdetection（FCOS）正式结果（`lvxs9xhk`，120/120，last AP50=0.770，AP50:95=0.285，best AP50=0.798@14）
- [x] D11 模型3：YOLO26 首轮正式结果（EarlyStopping at 263/300，best epoch=186）
- [x] D12 主对比表模板已完成（`docs/results/baselines.md`）
- [ ] D12 填入三模型实测结果并得出结论（已填 DRENet、YOLO26，待 FCOS）
- [x] D13 消融表模板已完成（`docs/results/ablation.md`）
- [ ] D13 完成至少 1-2 组消融并填写结果
- [x] D14 定性结果模板已完成（`docs/results/qualitative.md`）
- [ ] D14 整理成功/漏检/误检难例图到 `assets/figures/`

## E. 系统需求、设计与实现
- [x] E15 需求文稿已在开题稿中固定
- [x] E16 系统架构图代码已准备（Mermaid/PlantUML）
- [ ] E17 系统 MVP 代码实现（统一 `predict(...)` 接口）
- [x] E17-本机骨架已完成（配置读取、Adapter 路由、统一输出，支持真实推理接入）
- [x] E17-可视化界面原型已完成（Python 桌面 UI、模型切换、可视化输出、JSON 展示）
- [x] E17-纯 Python 可视化入口已完成（命令行 + matplotlib + 文件输出）
- [x] E17-纯 Python 桌面 UI 已完成（Qt 交互界面）
- [x] E17-多层可维护架构已落地（presentation/application/domain/infrastructure）
- [x] E17-三模型综合推理已接入（ensemble_all + 结果融合）
- [x] E17-单元/回归测试已补齐（fusion、参数校验、兼容导入）
- [ ] E18 演示用例准备（易/中/难三类样例）

## F. 论文写作与答辩材料
- [x] F19 论文大纲与材料映射已具备基础
- [ ] F19 用实测结果回填第四章与第五章内容

## G. 本机可异步执行清单（不依赖 GPU，可与训练并行）

### G1 并行于训练机（训练在跑时，本机立刻可做）
- [ ] G1-1 完成论文第1-3章终稿文字打磨（统一术语、删除重复表述）
- [ ] G1-2 完成“实验方法说明”终稿（只写方法，不写结果）
- [ ] G1-3 完成“评测指标定义”终稿（AP50/Precision/Recall/F1/IoU）
- [ ] G1-4 完成“风险与进度管理”终稿（周计划+里程碑）
- [ ] G1-5 完成答辩PPT文字版（课题意义/综述/难点/方案/条件）
- [ ] G1-6 按模块生成PPT配图（AI生成或论文截图）并存放到 `assets/figures/ppt/`
- [ ] G1-7 完成图注清单（每张图来源、用途、放置页码）保存到 `docs/ppt_figure_list.md`
- [ ] G1-8 统一参考文献格式（作者、年份、会议/期刊名一致）

### G2 纯本机工程准备（不训练也可完成）
- [x] G2-1 建立系统目录骨架（`src/`、`configs/`、`outputs/`、`tools/`）
- [x] G2-2 编写统一配置文件样例（3模型：DRENet/MMDet/YOLO）
- [x] G2-3 编写结果标准化 schema（`image_id/model_name/bbox/score/category_id/inference_time`）
- [x] G2-4 编写推理主流程伪代码与接口文档（`predict(...)` 输入输出契约）
- [x] G2-5 编写“训练机产物接入规范”（权重命名、目录结构、版本号）
- [x] G2-6 编写“结果回传后自动落盘流程”说明（对应 `scripts/sync_results_from_laptop.sh`）
- [ ] G2-7 规范化插件配置入口（将 `config_path=/.../drenet_local_plugin.py:build_predictor` 改为模块化导入路径），并补齐部署依赖说明（避免新机器缺少 DRENet 代码导致推理失败）
- [ ] G2-8 完成系统默认权重策略落地：区分 `global-best`（论文）与 `stable-best`（系统默认），并在 `configs/models.yaml` 与接入记录中固化

### G3 训练后立即衔接（结果一回传即可执行）
- [x] G3-1 将 DRENet 首轮结果填入 `docs/results/baselines.md`
- [ ] G3-2 生成第一版误检/漏检示例页并更新 `docs/results/qualitative.md`
- [ ] G3-3 依据首轮结果调整下一轮实验参数计划（形成 `docs/experiments/plans/plan-next-run.md`）
- [ ] G3-4 回填论文实验章节草稿（先写“现象与原因”，后补最终数字）
- [ ] G3-5 若 `best` 出现在波动期：补做邻域复核（`best_epoch±5`）与后段稳定区间统计（如最后20个epoch均值/方差），并写入论文实验章节“结果稳健性说明”
- [ ] G3-6 对 `global-best` 与 `stable-best` 做固定样本集 A/B 校验，并确定系统默认模型权重

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
1. 切到有卡实例，先完成 FCOS 冒烟与正式首轮，补 anchor-free 学术对比。
2. 回填 `docs/results/baselines.md`（补齐 FCOS）与 `docs/results/qualitative.md`。
3. 产出/更新 `docs/experiments/plans/plan-next-run.md`（FCOS 后续与 YOLO/DRENet 二阶段计划）。
