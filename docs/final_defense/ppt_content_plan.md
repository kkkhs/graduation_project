# 毕业答辩 PPT 逐页内容方案

> **论文题目**：《遥感场景下的微小船舶检测系统设计与实现》
> **答辩时长**：10–12 分钟
> **PPT 总页数**：12 页
> **生成日期**：2026-05-16

---

## 一、整体设计原则

### 1.1 配色方案

| 用途           | 色值                  | 说明                          |
| :------------- | :-------------------- | :---------------------------- |
| 主色（深蓝）   | `#1A365D`             | 标题、强调文字、图表主系列    |
| 辅色（浅蓝）   | `#3182CE`             | 二级标题、超链接、图表次系列  |
| 强调色（橙色） | `#DD6B20`             | 关键数字高亮、DRENet 数据系列 |
| 对比色（绿色） | `#38A169`             | YOLO26 数据系列、正向结论     |
| 对比色（紫色） | `#805AD5`             | FCOS 数据系列                 |
| 背景色         | `#FFFFFF` / `#F7FAFC` | 主背景 / 浅灰卡片背景         |
| 正文色         | `#2D3748`             | 正文文字                      |

### 1.2 字体建议

- **标题**：思源黑体 Bold / 微软雅黑 Bold，28–36pt
- **正文**：思源黑体 Regular / 微软雅黑，16–20pt
- **表格/代码**：等宽字体（Consolas / Source Code Pro），12–14pt
- **图表标注**：10–12pt，与正文同字体

### 1.3 图表风格

- 所有图表统一使用无边框风格，浅灰网格线
- 柱状图/折线图使用上述配色方案中的模型对应色
- 表格使用三线表风格（仅顶线、底线、表头线）
- 定性分析图使用 3×2 拼板布局，统一加 (a)(b)(c) 子图标注
- 系统截图加 1px 灰色边框 + 轻微阴影，模拟卡片效果

### 1.4 页面通用布局

- 顶部：页码 + 章节标题（左对齐）
- 中部：核心内容区（图文混排）
- 底部：学校 Logo + 答辩日期（右对齐）

---

## 二、逐页内容方案

---

### 第 1 页：封面

**页面布局**：

- 居中排列，自上而下：校徽 → 论文题目 → 答辩人信息 → 导师信息 → 日期

**核心文字内容**：

- 论文题目：遥感场景下的微小船舶检测系统设计与实现
- 答辩人：邝黄硕
- 导师：XXX 教授
- 学院：计算机与信息学院
- 日期：2026 年 5 月

**需要的图表/图片**：

- 学校校徽（从开题 PPT 中提取或学校官网下载）

**口播要点**：

- 无需口播，答辩开始时直接翻页

---

### 第 2 页：目录

**页面布局**：

- 左侧：纵向排列 6 个章节编号 + 标题
- 右侧：对应每章 1–2 行关键词/要点预览

**核心文字内容**：

| 编号 | 章节               | 要点预览                           |
| :--- | :----------------- | :--------------------------------- |
| 1    | 研究背景与意义     | 遥感船舶检测应用场景、微小目标挑战 |
| 2    | 国内外研究现状     | 四类方法归纳、20+ 论文调研         |
| 3    | 数据集与方法方案   | LEVIR-Ship、AP50、三种检测范式     |
| 4    | 实验设计与结果分析 | 主对比、消融、定性可视化           |
| 5    | 系统设计与实现     | 四层架构、适配器模式、异步任务     |
| 6    | 总结与展望         | 贡献、不足、未来方向               |

**需要的图表/图片**：

- 无

**口播要点**：

- "本次答辩将从以上六个方面展开，重点汇报实验对比结果和系统实现。"

---

### 第 3 页：研究背景与意义

**页面布局**：

- 上半部分（60%）：左侧 3 个应用场景图标 + 文字，右侧 1 张遥感船舶示例图
- 下半部分（40%）：3 个核心挑战 bullet points + 本课题定位

**核心文字内容**：

**应用场景**：

- 海事监控（Maritime Surveillance）：非法捕捞、走私监控
- 渔业管理：渔船动态监测与作业合规检查
- 海上交通：航道安全、船舶流量统计

**核心挑战**：

- 目标像素极少（多数仅几十像素），特征信息稀疏
- 背景复杂多变：海况、光照、云层、近岸纹理干扰
- 成像退化：遥感传感器模糊、低对比度、噪声

**本课题定位**：

- 在 LEVIR-Ship 公开数据集上，系统性对比三种不同范式的微小船舶检测方法
- 构建完整的前后端 Web 推理系统，支持单模型/融合模式切换

**需要的图表/图片**：

- 遥感船舶示例图：使用 [`assets/demo_cases/easy/GF6_WFV_E131.2_N35.8_20200910_L1A1120034678-3_1536_16384.png`](assets/demo_cases/easy/GF6_WFV_E131.2_N35.8_20200910_L1A1120034678-3_1536_16384.png) 的裁剪缩略图
- 或从论文 [`thesis_overleaf/figures/generated/ch1_pipeline.png`](thesis_overleaf/figures/generated/ch1_pipeline.png) 中截取

**口播要点**：

- "遥感船舶检测在海事监控、渔业管理、海上交通等领域有重要应用价值。"
- "但遥感图像中的船舶目标通常只有几十个像素，加上海面背景复杂多变，检测难度很大。"
- "本课题基于 LEVIR-Ship 公开数据集，对比三种不同范式的检测方法，并构建完整的 Web 推理系统。"

---

### 第 4 页：国内外研究现状

**页面布局**：

- 左侧（50%）：四象限/四列方法分类图，每类 2–3 个代表性方法名
- 右侧（50%）：调研统计（20+ 论文） + 本课题方法选择逻辑

**核心文字内容**：

**四类方法归纳**：

| 类别           | 代表方法                 | 特点                         |
| :------------- | :----------------------- | :--------------------------- |
| 通用目标检测   | Faster R-CNN, YOLO, DETR | CNN/Transformer 基础架构演进 |
| 小目标专门方法 | FPN, PANet, 超分增强     | 多尺度特征、注意力机制       |
| 遥感船舶检测   | LEVIR-Ship, DRENet       | 成像退化建模、海面背景适应   |
| 工具链与工程化 | MMDetection, Ultralytics | 标准化训练与评测框架         |

**本课题方法选择**：

- DRENet：退化重建增强，代表"专门方法"
- FCOS（MMDetection）：Anchor-free，代表"通用检测器范式"
- YOLO26n（Ultralytics）：One-stage 轻量级，代表"工业级部署基线"

**需要的图表/图片**：

- 无特殊图片，可用 SmartArt 或四象限布局呈现

**口播要点**：

- "调研了 20 余篇论文，按四个维度分类：通用检测、小目标方法、遥感船舶专门方法、工具链。"
- "基于调研，选择了三种不同范式的代表性方法进行对比：DRENet 代表专门方法，FCOS 代表 anchor-free 范式，YOLO26n 代表轻量化工业级方法。"
- "三者覆盖了专门方法、通用检测器、轻量部署三种定位，对比维度丰富。"

---

### 第 5 页：数据集与评价指标

**页面布局**：

- 上半部分（50%）：LEVIR-Ship 数据集关键统计（表格 + 示例图）
- 下半部分（50%）：评价指标定义（AP50 为主，AP50:95 为辅）

**核心文字内容**：

**LEVIR-Ship 数据集**（Chen et al., IEEE TGRS 2022）：

- 图像数量：3000+ 张高分辨率遥感图像（Google Earth）
- 目标特点：船舶目标极小（多数仅几十像素），单类别（ship）
- 标注格式：水平边界框（Horizontal Bounding Box）
- 数据划分：固定 train/val/test 划分，保证可复现

**评价指标**：

- **AP50（主指标）**：IoU=0.50 下的 Average Precision
  - 微小目标场景下 IoU=0.75 过于苛刻，AP50 更能反映检出能力
- **AP50:95（辅助指标）**：IoU 从 0.50 到 0.95 步长 0.05 的均值
- **Precision / Recall / F1**：辅助分析精度-召回 trade-off
- **FPS / Params**：效率与复杂度度量

**需要的图表/图片**：

- 数据集示例图：[`thesis_overleaf/figures/generated/ch3_dataset_examples.png`](thesis_overleaf/figures/generated/ch3_dataset_examples.png)
- 或从 LEVIR-Ship 论文中引用数据集统计图

**口播要点**：

- "LEVIR-Ship 是专门针对微小船舶场景的公开数据集，3000+ 张高分辨率遥感图像，固定划分保证可复现。"
- "评价指标以 AP50 为主——因为微小目标检测中 IoU=0.75 过于严格，几个像素偏移就可能导致匹配失败。"
- "同时报告 AP50:95、Precision、Recall、F1 作为辅助分析维度。"

---

### 第 6 页：模型方法概述

**页面布局**：

- 三列等宽布局，每列一个模型
- 每列包含：模型名称、范式标签、核心思想（2–3 句）、关键参数、网络结构示意图

**核心文字内容**：

|              | DRENet                                             | FCOS                                  | YOLO26n                            |
| :----------- | :------------------------------------------------- | :------------------------------------ | :--------------------------------- |
| **范式**     | Degradation-Reconstruction 增强                    | Anchor-Free 检测器                    | One-Stage 轻量检测器               |
| **框架**     | 自定义 PyTorch                                     | MMDetection                           | Ultralytics                        |
| **核心思想** | 模拟成像退化 → 逆向重建增强 → 提升微小目标特征表达 | 逐像素直接回归边界框，无需预设 anchor | CSPNet + 高效 Neck，平衡精度与速度 |
| **输入尺寸** | 512×512                                            | 1333×800（历史默认）                  | 512×512                            |
| **参数量**   | 4.79M                                              | 32.17M                                | 2.50M                              |

**需要的图表/图片**：

- DRENet 结构图：[`thesis_overleaf/figures/generated/ch3_drenet_overview.png`](thesis_overleaf/figures/generated/ch3_drenet_overview.png)
- FCOS 结构图：[`thesis_overleaf/figures/generated/ch3_fcos_pipeline.png`](thesis_overleaf/figures/generated/ch3_fcos_pipeline.png)
- YOLO 结构图：[`thesis_overleaf/figures/generated/ch3_yolo_pipeline.png`](thesis_overleaf/figures/generated/ch3_yolo_pipeline.png)
- 三张图缩小后并排或上下排列

**口播要点**：

- "三种模型代表了三种不同的检测范式。"
- "DRENet 通过退化重建增强模块提升微小目标特征，参数量 4.79M。"
- "FCOS 是 anchor-free 检测器，无需预设 anchor，使用 MMDetection 框架，参数量 32M。"
- "YOLO26n 是 Ultralytics 最新轻量模型，参数量仅 2.5M，适合部署。"
- "需要说明的是，FCOS 使用历史默认 1333×800 训练口径，与 DRENet/YOLO26 的 512 口径不同，论文中将其作为 anchor-free 参考基线而非严格同口径对比。"

---

### 第 7 页：主对比实验结果

**页面布局**：

- 上半部分（60%）：三模型主对比表格（突出关键数字）
- 下半部分（40%）：3 条核心结论（bullet points + 关键数字高亮）

**核心文字内容**：

**三模型主对比（LEVIR-Ship 测试集）**：

| 指标        |   DRENet   | FCOS¹  |  YOLO26n   |
| :---------- | :--------: | :----: | :--------: |
| **AP50 ↑**  |   0.7949   | 0.7700 | **0.7950** |
| AP50:95 ↑   |   0.2919   | 0.2850 | **0.3170** |
| Precision ↑ |   0.4927   |   —    | **0.8430** |
| Recall ↑    | **0.8511** | 0.4050 |   0.7220   |
| F1 ↑        |   0.6241   |   —    | **0.7778** |
| FPS ↑       | **121.9**  |  ~46   |    69.1    |
| Params(M) ↓ |    4.79    | 32.17  |  **2.50**  |

> ¹ FCOS 使用历史默认口径 1333×800 训练，作为 anchor-free 参考基线

**核心结论**：

- **AP50 持平**：DRENet（0.7949）≈ YOLO26（0.7950），两者在检测精度上接近
- **召回 vs 精度 trade-off**：DRENet 召回最高（0.8511）但 Precision 最低（0.4927）；YOLO26 Precision 最高（0.8430）且 F1 最优（0.7778）
- **轻量化优势**：YOLO26n 参数量仅 2.50M，DRENet 推理速度最快（121.9 FPS）

**需要的图表/图片**：

- 主对比柱状图：[`thesis_overleaf/figures/generated/ch4_main_comparison_chart.png`](thesis_overleaf/figures/generated/ch4_main_comparison_chart.png)
- 放在表格下方或右侧

**口播要点**：

- "这是三模型在 LEVIR-Ship 测试集上的主对比结果。"
- "AP50 上 DRENet 和 YOLO26 非常接近，都在 0.795 左右。"
- "但两者有明显差异：DRENet 召回率最高 0.851，适合强调不漏检的场景；YOLO26 精度最高 0.843，误检更少，F1 也最优。"
- "FCOS 作为 anchor-free 参考基线，AP50 为 0.770，但参数量最大、速度最慢。"
- "综合来看，DRENet 适合作为强调召回的主方法，YOLO26 适合轻量化部署。"

---

### 第 8 页：消融实验

**页面布局**：

- 左侧（50%）：阈值消融结果表 + 趋势箭头
- 右侧（50%）：尺寸敏感性结果表 + 趋势箭头
- 底部：2 条消融结论

**核心文字内容**：

**阈值消融**（固定 iou=0.50，扫描 conf=0.15/0.25/0.35）：

| 模型        | conf=0.15   | conf=0.25   | conf=0.35   | 趋势         |
| :---------- | :---------- | :---------- | :---------- | :----------- |
| DRENet AP50 | 0.6616      | 0.6616      | 0.6616      | → 持平       |
| DRENet P/R  | 0.644/0.794 | 0.733/0.766 | 0.778/0.736 | P↑ R↓        |
| FCOS AP50   | 0.7389      | 0.7389      | 0.7389      | → 持平       |
| FCOS P/R    | 0.728/0.835 | 0.762/0.830 | 0.776/0.823 | P↑ R↓        |
| YOLO26 AP50 | 0.6661      | 0.6661      | 0.5914      | ↓ 高阈值下降 |
| YOLO26 P/R  | 0.641/0.790 | 0.753/0.761 | 0.825/0.699 | P↑ R↓        |

**尺寸敏感性**（推理阶段，conf=0.25, iou=0.50）：

| 模型        | 512    | 640    | 800    | 趋势       |
| :---------- | :----- | :----- | :----- | :--------- |
| FCOS AP50   | 0.4490 | 0.6452 | 0.7389 | ↑ 高度敏感 |
| YOLO26 AP50 | 0.6661 | 0.6737 | 0.6564 | 640 最优   |

> DRENet 仅支持 512×512，不纳入尺寸敏感性主表

**消融结论**：

- 三模型均表现出"阈值升高 → Precision↑、Recall↓"的典型趋势；DRENet 在 0.25–0.35 区间最平衡
- FCOS 对输入尺寸高度敏感（512→800 时 AP50 从 0.449 升至 0.739）；YOLO26 在 640 时最均衡

**需要的图表/图片**：

- 阈值趋势折线图：[`thesis_overleaf/figures/generated/ch4_threshold_trend.png`](thesis_overleaf/figures/generated/ch4_threshold_trend.png)
- 尺寸趋势折线图：[`thesis_overleaf/figures/generated/ch4_size_trend.png`](thesis_overleaf/figures/generated/ch4_size_trend.png)

**口播要点**：

- "两组离线消融实验揭示了模型行为差异。"
- "阈值消融：三模型都表现出阈值升高精度上升、召回下降的典型趋势。DRENet 在 0.25 到 0.35 区间最平衡。"
- "尺寸敏感性：FCOS 对输入尺寸高度敏感，从 512 到 800 持续提升；YOLO26 在 640 最均衡，再增大反而下降。"
- "DRENet 因网络结构限制仅支持 512，论文中将其写为实现边界而非结果缺失。"

---

### 第 9 页：定性分析

**页面布局**：

- 3×2 拼板图占据页面主体（80%）
- 底部简短文字说明

**核心文字内容**：

**定性可视化**：覆盖 success / miss / false_positive 三类场景，三模型各一列

|                    | DRENet               | FCOS                 | YOLO26                 |
| :----------------- | :------------------- | :------------------- | :--------------------- |
| **Success**        | 预测框与标注基本重合 | 预测框与标注一致     | 多测试块预测与标注一致 |
| **Miss**           | 稀疏小目标覆盖不完整 | 标注存在但未完全检出 | 局部弱目标漏检         |
| **False Positive** | 近岸纹理误判         | 存在多余预测框       | 纹理复杂区域额外框     |

**需要的图表/图片**：

- 成功检测拼板图：[`thesis_overleaf/figures/generated/ch4_success_cases.png`](thesis_overleaf/figures/generated/ch4_success_cases.png)
- 漏检拼板图：[`thesis_overleaf/figures/generated/ch4_miss_cases.png`](thesis_overleaf/figures/generated/ch4_miss_cases.png)
- 误检拼板图：[`thesis_overleaf/figures/generated/ch4_false_positive_cases.png`](thesis_overleaf/figures/generated/ch4_false_positive_cases.png)
- 建议将三张拼板图缩小后并排展示，或选取最具代表性的一张放大

**口播要点**：

- "定性分析使用 3×2 拼板图，覆盖成功检测、漏检、误检三类场景。"
- "DRENet 在低纹理海面小目标场景下检出较稳定，但近岸复杂背景处可能产生误检。"
- "YOLO26 整体检出稳定性好，但在局部弱目标场景存在漏检。"
- "FCOS 作为参考基线，在典型海面场景表现可接受。"
- "这些定性结果与前面的定量指标相互印证。"

---

### 第 10 页：系统设计与实现

**页面布局**：

- 上半部分（55%）：系统架构图（四层架构 + 数据流）
- 下半部分（45%）：3 个设计要点（适配器模式、异步任务、融合策略）

**核心文字内容**：

**系统架构**（四层）：

- **Presentation**：React + Vite + TypeScript + Ant Design，4 页面
- **Application**：FastAPI 路由 + 任务编排 + 融合调度
- **Domain**：实体定义 + 接口抽象（不依赖任何框架）
- **Infrastructure**：模型适配器 + 配置加载 + 可视化渲染

**关键设计**：

- **适配器模式**：统一 `infer()` 接口封装 DRENet/FCOS/YOLO26，通过 [`adapter_factory.py`](src/infrastructure/adapter_factory.py) 动态创建
- **异步任务调度**：ThreadPoolExecutor + 状态机（queued → running → done/failed），前端 4 秒轮询
- **融合策略**：IoU 加权框融合（WBF），默认 ensemble 模式，综合三模型结果

**技术栈**：React + FastAPI + SQLite + PyTorch + MMDetection + Ultralytics

**需要的图表/图片**：

- 系统架构图：[`thesis_overleaf/figures/generated/ch5_architecture.png`](thesis_overleaf/figures/generated/ch5_architecture.png)
- 系统流程图：[`thesis_overleaf/figures/generated/ch5_flow.png`](thesis_overleaf/figures/generated/ch5_flow.png)
- 数据库 ER 图：[`thesis_overleaf/figures/generated/ch5_er.png`](thesis_overleaf/figures/generated/ch5_er.png)（可选，放附录）
- 页面截图拼板：[`thesis_overleaf/figures/generated/ch5_pages.png`](thesis_overleaf/figures/generated/ch5_pages.png)

**口播要点**：

- "系统采用四层架构：前端 React、后端 FastAPI、领域层纯抽象、基础设施层封装模型推理。"
- "核心设计之一是适配器模式——三个模型实现统一的 infer 接口，通过工厂类动态创建和缓存。"
- "异步任务使用线程池，状态从 queued 到 running 到 done，前端轮询实时更新。"
- "默认使用三模型融合模式，采用 IoU 加权框融合策略。"
- "接下来进行现场演示。"

---

### 第 11 页：系统演示

**页面布局**：

- 此页为过渡页/标题页
- 居中大字："系统演示" + 副标题"单模型推理 → 融合模式 → 模型管理"
- 底部提示："以下为现场操作，预计 3–4 分钟"

**核心文字内容**：

- 系统演示
- 演示路径：🟢 易（YOLO26 单模型）→ 🟡 中（DRENet 单模型）→ 🔴 难（FCOS 单模型，可选）

**需要的图表/图片**：

- 无（此页为过渡页）

**口播要点**：

- "下面进行现场系统演示，展示从提交任务到查看结果的完整闭环。"

> **详细演示操作脚本见第三章。**

---

### 第 12 页：总结与展望

**页面布局**：

- 左侧（50%）：工作总结（4 条 bullet points）
- 右侧（50%）：不足与展望（4–5 条 bullet points）

**核心文字内容**：

**工作总结**：

- 调研 20+ 篇论文，覆盖通用检测、小目标、遥感船舶、工具链四个维度
- 在 LEVIR-Ship 上完成 DRENet / FCOS / YOLO26 三种范式的系统性对比实验
- 完成阈值消融与输入尺寸敏感性两组离线分析，揭示模型行为差异
- 设计并实现了基于适配器模式的多模型 Web 推理系统，支持单模型/融合模式切换与异步任务调度

**不足与展望**：

- 未提出新算法，工作以对比分析和系统集成为主
- 三模型输入尺寸不完全统一，FCOS 对比存在口径差异
- 当前仅支持水平边界框，未来可引入旋转框检测（OBB）
- 系统在 CPU 上推理，实际部署需 GPU 环境 + Docker 容器化
- 可尝试 Transformer-based 检测器（DETR/DINO）作为第四种范式

**需要的图表/图片**：

- 无特殊图片，可用图标装饰

**口播要点**：

- "总结一下，本课题完成了从文献调研、模型对比实验、消融分析到系统实现的完整链路。"
- "实验证据充分，系统具备单模型/融合模式切换、异步任务调度、结果可视化等能力。"
- "不足之处包括：未提出新算法、输入尺寸不完全统一、仅支持水平框等。"
- "未来可以引入旋转框检测、尝试 Transformer 检测器、完善 GPU 部署方案。"
- "我的答辩到此结束，请各位老师批评指正。"

---

## 三、第 11 页系统演示详细操作脚本

### 3.1 演示前准备（答辩开始前 5 分钟）

```bash
cd /Users/khs/codes/graduation_project
bash scripts/init_db.sh          # 初始化数据库
bash scripts/start_backend.sh    # 启动后端 :8000
bash scripts/start_frontend.sh   # 启动前端 :5173
```

**验证清单**：

- [ ] 前端 `http://127.0.0.1:5173` 可访问
- [ ] 健康接口 `http://127.0.0.1:8000/api/v1/health` 返回 200
- [ ] 模型管理页确认 3 个模型均已启用
- [ ] 三张 demo 图片存在于 `assets/demo_cases/`
- [ ] 兜底截图已准备（见 3.4 节）

---

### 3.2 阶段 1：YOLO26 单模型快速闭环（约 1.5 分钟）

| 步骤 | 操作                                                                                              | 口播词                                                                       |
| :--- | :------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------- |
| 1    | 打开浏览器，进入「任务提交」页                                                                    | "首先看系统的主界面——任务提交页。"                                           |
| 2    | 选择：`单图` + `单模型` + `YOLO26`，阈值保持 `0.25`                                               | "选择单图推理、单模型模式、YOLO26，阈值使用默认的 0.25。"                    |
| 3    | 上传样例图：`assets/demo_cases/easy/GF6_WFV_E131.2_N35.8_20200910_L1A1120034678-3_1536_16384.png` | "上传一张典型的遥感海面图像，这是 GF6 卫星的 WFV 传感器数据。"               |
| 4    | 点击「提交」→ 自动跳转任务列表页                                                                  | "提交后自动跳转到任务列表，可以看到任务状态从 queued 变为 running。"         |
| 5    | 等待任务完成（约 5–15 秒），状态变为 done                                                         | "系统采用异步任务架构，后台线程执行推理，前端每 4 秒轮询状态，不阻塞界面。"  |
| 6    | 点击进入「任务详情」页                                                                            | "进入详情页，可以看到三个区域：可视化检测结果图、结构化结果表、任务元信息。" |
| 7    | 指认结果图上的检测框                                                                              | "YOLO26 在这张典型海面场景上检出效果稳定，框的位置和置信度都合理。"          |
| 8    | 滚动到结构化结果表                                                                                | "结果表列出每个检测框的坐标、置信度和类别，支持数据导出。"                   |

---

### 3.3 阶段 2：DRENet 单模型展示第二算法（约 1 分钟）

| 步骤 | 操作                                                                                                | 口播词                                                                                                                                |
| :--- | :-------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | 回到「任务提交」页                                                                                  | "回到提交页，展示第二个算法。"                                                                                                        |
| 2    | 切换模型为 `DRENet`，其余参数不变                                                                   | "切换为 DRENet，这是基于退化重建增强的专门方法。"                                                                                     |
| 3    | 上传中等样例：`assets/demo_cases/medium/GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_8704_1536.png` | "换一张 GF1 卫星的中等难度图像。"                                                                                                     |
| 4    | 提交 → 等待 → 进入详情页                                                                            | "同样走一遍提交到完成的闭环。"                                                                                                        |
| 5    | 对比两个模型的结果差异                                                                              | "可以看到 DRENet 的检测风格与 YOLO26 有所不同——DRENet 倾向于检出更多候选框，召回更高，但也可能伴随更多误检，这与前面的定量分析一致。" |

---

### 3.4 阶段 3：模型管理页展示（约 0.5 分钟）

| 步骤 | 操作                 | 口播词                                                                                     |
| :--- | :------------------- | :----------------------------------------------------------------------------------------- |
| 1    | 切换到「模型管理」页 | "最后看一下模型管理页。"                                                                   |
| 2    | 指认三个模型卡片     | "系统当前接入 DRENet、FCOS、YOLO26 三个模型，每个模型显示名称、框架、权重路径和启用状态。" |
| 3    | 说明启用/禁用开关    | "支持运行时启用或禁用模型，无需重启服务。这个设计方便在演示时灵活切换可用模型。"           |

---

### 3.5 阶段 4（可选）：FCOS 难例展示（约 0.5 分钟，仅在老师追问时展开）

| 步骤 | 操作                                                                                           | 口播词                                                                                |
| :--- | :--------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| 1    | 回到提交页，切换为 FCOS                                                                        | "FCOS 作为 anchor-free 参考基线，我们来看一下它在复杂场景下的表现。"                  |
| 2    | 上传难例：`assets/demo_cases/hard/GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_10976_9728.png` | "这是一张大尺寸复杂场景图像，适合展示系统边界和难例分析能力。"                        |
| 3    | 提交并查看结果                                                                                 | "FCOS 在这类复杂场景下可能出现漏检或误检，这也说明了为什么需要多模型对比和融合策略。" |

---

### 3.6 兜底方案（服务异常时）

若现场推理失败，立即切换静态截图：

| 截图               | 路径                                                                                                 |
| :----------------- | :--------------------------------------------------------------------------------------------------- |
| 提交页             | [`outputs/ui-submit-after-polish-20260315.png`](outputs/ui-submit-after-polish-20260315.png)         |
| 列表页             | [`outputs/ui-tasks-after-polish-20260315.png`](outputs/ui-tasks-after-polish-20260315.png)           |
| 详情页（融合结果） | [`outputs/ui-detail-fused-last-20260315.png`](outputs/ui-detail-fused-last-20260315.png)             |
| 详情页（进度）     | [`outputs/ui-detail-progress-unified-20260315.png`](outputs/ui-detail-progress-unified-20260315.png) |
| 模型管理页         | [`outputs/pw-models-desktop-after.png`](outputs/pw-models-desktop-after.png)                         |

**兜底口播**："演示流程与系统在线流程一致，展示的是同一版本同一数据链路。接下来看静态截图说明同样的闭环。"

---

## 四、附录建议

以下为备用 slides，建议放在 PPT 末尾，仅在老师追问时翻出：

### 附录 A：完整指标对比表

与第 7 页相同但补充更多细节（FLOPs、训练 epoch、W&B run ID 等），数据来源：[`docs/results/baselines.md`](docs/results/baselines.md)。

### 附录 B：消融实验完整数据

阈值消融 9 行 + 尺寸敏感性 6 行的完整表格，数据来源：[`docs/results/ablation.md`](docs/results/ablation.md)。

### 附录 C：系统技术细节

- 数据库 ER 图：[`thesis_overleaf/figures/generated/ch5_er.png`](thesis_overleaf/figures/generated/ch5_er.png)
- 全链路流程图（Mermaid）：参见 [`docs/system/architecture.md`](docs/system/architecture.md) 第 6.1 节
- API 接口列表（RESTful，`/api/v1/` 前缀）

### 附录 D：参考文献精选（5–8 篇）

| 编号 | 文献                                    | 说明             |
| :--- | :-------------------------------------- | :--------------- |
| [1]  | Chen et al., LEVIR-Ship, IEEE TGRS 2022 | 数据集来源       |
| [2]  | Chen et al., DRENet, IEEE TGRS 2022     | DRENet 方法参考  |
| [3]  | Tian et al., FCOS, ICCV 2019            | FCOS 方法参考    |
| [4]  | Ultralytics, YOLO26, 2026               | YOLO26 框架      |
| [5]  | MMDetection Contributors, 2020          | MMDetection 框架 |
| [6]  | Solovyev et al., WBF, 2021              | 加权框融合方法   |

### 附录 E：工作量统计

- 文献调研：20+ 篇论文
- 模型复现与训练：3 种模型（DRENet / FCOS / YOLO26）
- 消融实验：2 组（阈值消融 + 尺寸敏感性）
- 系统开发：前端 4 页面 + 后端 RESTful API + 推理内核
- 核心代码量：约 3000+ 行（不含实验脚本和配置）
- 论文：6 章，约 XX 页

### 附录 F：论文结构速览

```
第一章  研究背景与意义
第二章  国内外研究现状（四类方法归纳）
第三章  数据集、评价指标与方法方案
第四章  实验设计与结果分析（主对比 + 消融 + 定性）
第五章  系统需求分析、设计与实现
第六章  总结与展望
```

---

## 五、图片资源清单

### 5.1 项目中已有的图片（可直接使用）

| 用途             | 路径                                                                                                                                                                         | 对应 PPT 页码   |
| :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- |
| 遥感船舶示例图   | [`assets/demo_cases/easy/GF6_WFV_E131.2_N35.8_20200910_L1A1120034678-3_1536_16384.png`](assets/demo_cases/easy/GF6_WFV_E131.2_N35.8_20200910_L1A1120034678-3_1536_16384.png) | 第 3 页         |
| 数据集示例图     | [`thesis_overleaf/figures/generated/ch3_dataset_examples.png`](thesis_overleaf/figures/generated/ch3_dataset_examples.png)                                                   | 第 5 页         |
| DRENet 结构图    | [`thesis_overleaf/figures/generated/ch3_drenet_overview.png`](thesis_overleaf/figures/generated/ch3_drenet_overview.png)                                                     | 第 6 页         |
| FCOS 结构图      | [`thesis_overleaf/figures/generated/ch3_fcos_pipeline.png`](thesis_overleaf/figures/generated/ch3_fcos_pipeline.png)                                                         | 第 6 页         |
| YOLO 结构图      | [`thesis_overleaf/figures/generated/ch3_yolo_pipeline.png`](thesis_overleaf/figures/generated/ch3_yolo_pipeline.png)                                                         | 第 6 页         |
| 主对比柱状图     | [`thesis_overleaf/figures/generated/ch4_main_comparison_chart.png`](thesis_overleaf/figures/generated/ch4_main_comparison_chart.png)                                         | 第 7 页         |
| 阈值趋势图       | [`thesis_overleaf/figures/generated/ch4_threshold_trend.png`](thesis_overleaf/figures/generated/ch4_threshold_trend.png)                                                     | 第 8 页         |
| 尺寸趋势图       | [`thesis_overleaf/figures/generated/ch4_size_trend.png`](thesis_overleaf/figures/generated/ch4_size_trend.png)                                                               | 第 8 页         |
| 成功检测拼板图   | [`thesis_overleaf/figures/generated/ch4_success_cases.png`](thesis_overleaf/figures/generated/ch4_success_cases.png)                                                         | 第 9 页         |
| 漏检拼板图       | [`thesis_overleaf/figures/generated/ch4_miss_cases.png`](thesis_overleaf/figures/generated/ch4_miss_cases.png)                                                               | 第 9 页         |
| 误检拼板图       | [`thesis_overleaf/figures/generated/ch4_false_positive_cases.png`](thesis_overleaf/figures/generated/ch4_false_positive_cases.png)                                           | 第 9 页         |
| 系统架构图       | [`thesis_overleaf/figures/generated/ch5_architecture.png`](thesis_overleaf/figures/generated/ch5_architecture.png)                                                           | 第 10 页        |
| 系统流程图       | [`thesis_overleaf/figures/generated/ch5_flow.png`](thesis_overleaf/figures/generated/ch5_flow.png)                                                                           | 第 10 页        |
| 页面截图拼板     | [`thesis_overleaf/figures/generated/ch5_pages.png`](thesis_overleaf/figures/generated/ch5_pages.png)                                                                         | 第 10 页        |
| 数据库 ER 图     | [`thesis_overleaf/figures/generated/ch5_er.png`](thesis_overleaf/figures/generated/ch5_er.png)                                                                               | 附录 C          |
| 技术路线图       | [`thesis_overleaf/figures/generated/ch1_pipeline.png`](thesis_overleaf/figures/generated/ch1_pipeline.png)                                                                   | 第 3 页（可选） |
| 系统截图（兜底） | [`outputs/ui-submit-after-polish-20260315.png`](outputs/ui-submit-after-polish-20260315.png)                                                                                 | 第 11 页兜底    |
| 系统截图（兜底） | [`outputs/ui-tasks-after-polish-20260315.png`](outputs/ui-tasks-after-polish-20260315.png)                                                                                   | 第 11 页兜底    |
| 系统截图（兜底） | [`outputs/ui-detail-fused-last-20260315.png`](outputs/ui-detail-fused-last-20260315.png)                                                                                     | 第 11 页兜底    |
| 系统截图（兜底） | [`outputs/pw-models-desktop-after.png`](outputs/pw-models-desktop-after.png)                                                                                                 | 第 11 页兜底    |

### 5.2 需要额外生成或准备的图片

| 用途                    | 说明                                                                 | 优先级 |
| :---------------------- | :------------------------------------------------------------------- | :----- |
| 学校校徽                | 从开题 PPT 或学校官网获取高清 PNG                                    | **高** |
| 第 3 页应用场景图标     | 海事监控/渔业管理/海上交通三个图标（可从 iconfont 或 flaticon 获取） | 中     |
| 第 4 页四象限方法分类图 | 可用 PPT SmartArt 直接制作，无需外部图片                             | 低     |
| 第 7 页主对比表美化版   | 建议在 PPT 中直接用表格功能制作（而非截图），便于高亮关键数字        | 低     |
| 第 12 页图标装饰        | 工作总结/不足/展望的图标（可从 PPT 内置图标库获取）                  | 低     |
| 现场演示录屏（备选）    | 若担心现场网络/服务不稳定，可提前录制 3 分钟演示视频作为终极兜底     | 中     |

---

## 六、时间分配建议

| 页码     | 内容             | 建议时间              |
| :------- | :--------------- | :-------------------- |
| 1        | 封面             | —                     |
| 2        | 目录             | 10 秒                 |
| 3        | 研究背景与意义   | 1 分钟                |
| 4        | 国内外研究现状   | 1 分钟                |
| 5        | 数据集与评价指标 | 1 分钟                |
| 6        | 模型方法概述     | 1 分钟                |
| 7        | 主对比实验结果   | 1.5 分钟              |
| 8        | 消融实验         | 1 分钟                |
| 9        | 定性分析         | 0.5 分钟              |
| 10       | 系统设计与实现   | 1.5 分钟              |
| 11       | 系统演示         | 3–4 分钟              |
| 12       | 总结与展望       | 1 分钟                |
| **合计** |                  | **约 11.5–12.5 分钟** |

---

> **参考资料**：
>
> - 答辩 QA 指南：[`docs/final_defense/defense_qa_guide.md`](docs/final_defense/defense_qa_guide.md)
> - 主对比结果：[`docs/results/baselines.md`](docs/results/baselines.md)
> - 消融实验：[`docs/results/ablation.md`](docs/results/ablation.md)
> - 定性分析：[`docs/results/qualitative.md`](docs/results/qualitative.md)
> - 系统架构：[`docs/system/architecture.md`](docs/system/architecture.md)
> - 演示样例：[`docs/system/demo_case_catalog.md`](docs/system/demo_case_catalog.md)
> - 中期演示脚本：[`docs/midterm/midterm_demo_script_20260315.md`](docs/midterm/midterm_demo_script_20260315.md)
