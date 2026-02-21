# 本科毕业设计开题报告

## 0. 基本信息

- **题目**：遥感场景下的微小船舶检测系统设计与实现
- **学生**：\<填写\>
- **学号**：\<填写\>
- **专业/班级**：\<填写\>
- **指导教师**：\<填写\>
- **日期**：2026-02-07

---

## 一、研究背景与意义

### 1.1 研究背景

遥感影像具备覆盖范围大、获取频率高、可在复杂海域/港口环境中持续观测等优势，在海上交通监管、海事执法、港口调度、海上安全与应急救援等场景具有重要应用价值。船舶检测作为遥感智能解译的重要任务之一，目标是从遥感影像中自动定位并识别船舶目标，为后续的目标跟踪、态势分析、风险预警与应急响应提供基础数据支撑。

近年来，随着卫星遥感技术的快速发展，高分辨率、中分辨率遥感影像的获取成本不断降低，数据量呈指数级增长。然而，在实际应用中，中分辨率遥感影像（如16m/pixel）因其覆盖范围广、获取效率高，更适合大范围海域的快速监测。但中分辨率影像中的船舶目标往往只占据少量像素（可能仅20像素左右），且背景复杂（如海面波纹、碎云、港口设施等），给微小目标检测带来了巨大挑战。

### 1.2 问题定义

**输入**：光学遥感图像（卫星/航空），背景包含海面波纹、浪花、港口设施、云雾等复杂干扰。

**输出**：船舶目标的检测框（bbox）与置信度（必要时导出结构化结果，如 JSON/CSV）。

**核心挑战**：

- **微小目标特征弱**：船舶在中分辨率遥感图中可能仅占少量像素，细节信息不足，易漏检。
- **尺度变化与多样背景**：船舶尺寸变化大且与背景纹理相近，易误检。
- **目标稀疏与类别不平衡**：目标在大幅面影像中分布稀疏，正负样本极不均衡。
- **成像退化与噪声**：成像模糊、压缩、雾霾等影响可见特征，降低检测稳定性。
- **碎云干扰**：碎云容易被误认为船舶目标，导致误检率上升。

### 1.3 研究意义

本课题以公开遥感船舶数据集（LEVIR-Ship）为基础，通过复现与对比多种检测模型，形成可复现的实验结论，并进一步完成检测系统原型集成与可视化输出，实现从算法到工程应用的闭环验证。研究成果可为遥感场景下微小目标检测的模型选型、工程落地与复现实验提供参考。

具体意义包括：

1. **学术价值**：系统性地对比多种检测方法在微小船舶检测任务上的性能，为该领域的研究提供实验依据。
2. **工程价值**：实现可运行的检测系统原型，为实际应用提供技术参考。
3. **复现价值**：提供完整的实验记录、配置和代码，便于其他研究者复现和验证。

---

## 二、国内外研究现状与评述

> 要求：整理 20–30 篇论文。此处给出分类框架与写作方法，具体论文已整理到 `docs/literature/literature_table.md` 中。

### 2.1 通用目标检测方法演进

通用目标检测从两阶段（Two-stage）方法到单阶段（One-stage）方法，再到 Anchor-Free 检测器以及 Transformer 检测器持续发展。

**两阶段方法**以 Faster R-CNN（Ren et al., 2015）为代表，先通过区域建议网络（RPN）生成候选区域，再对每个区域进行分类和边界框回归。这类方法通常具有较高的定位精度和稳定性，但推理速度相对较慢。

**单阶段方法**以 YOLO 系列（Redmon et al., 2016; Redmon & Farhadi, 2018）和 SSD 为代表，直接在图像上进行密集预测，推理速度快，适合实时应用。RetinaNet（Lin et al., 2017）通过 Focal Loss 解决了正负样本不平衡问题，进一步提升了单阶段检测器的性能。

**Anchor-Free 方法**如 FCOS（Tian et al., 2019）摒弃了预设锚框，采用逐像素预测的方式，简化了检测流程，减少了超参数调优的工作量。

**Transformer 检测器**以 DETR（Carion et al., 2020）为代表，将目标检测建模为集合预测问题，利用注意力机制建模全局上下文。后续的 Deformable DETR（Zhu et al., 2021）和 DINO（Zhang et al., 2023）通过可变形注意力和去噪训练等技术，进一步提升了 Transformer 检测器的性能和收敛速度。

### 2.2 小目标/微小目标检测相关研究

小目标检测是目标检测领域的难点问题。Cheng et al.（2022）对小目标检测进行了系统性综述，总结了该领域的主要挑战和解决方案。Zhang et al.（2025）提出了 FDM-YOLO，一种基于 YOLOv8 的轻量级小目标检测改进模型。

针对小目标检测的常用技术包括：

1. **多尺度特征融合**：FPN（Lin et al., 2017）和 PAFPN（Liu et al., 2018）通过融合不同尺度的特征，提升模型对不同大小目标的检测能力。
2. **注意力机制**：CBAM（Woo et al., 2018）和 SENet（Hu et al., 2018）通过通道注意力和空间注意力，帮助模型关注重要特征。
3. **数据增强**：Mosaic、Copy-Paste 等方法通过增加小目标样本，提升模型的泛化能力。
4. **损失函数改进**：Focal Loss（Lin et al., 2017）通过调整损失权重，解决正负样本不平衡问题。

### 2.3 遥感/船舶检测与数据集研究

遥感船舶检测具备高分辨率大图、目标稀疏、背景复杂、尺度变化大的典型特征。围绕公开数据集的研究工作逐步增多。

**数据集方面**：

- LEVIR-Ship（Chen et al., 2022）：首个公开的中分辨率遥感船舶检测数据集，包含 3876 张图像和 3219 个船舶实例，空间分辨率为 16m/pixel。
- HRSC2016：高分辨率遥感船舶检测数据集，包含旋转框标注。
- DOTA：大规模旋转目标检测数据集，包含多个类别。

**方法方面**：
Chen et al.（2022）提出了 DRENet，通过退化重建增强（Degraded Reconstruction Enhancement）方法，引导网络关注微小船舶目标，在 LEVIR-Ship 数据集上达到了 82.4 AP 的性能。Wu et al.（2022）提出了 MTU-Net，用于红外微小船舶检测。

针对遥感检测的工程化问题，Qu et al.（2020）研究了主动学习以减少标注成本，Kim et al.（2024）提出了处理噪声边界框的方法。Li et al.（2024）提出了 LR-FPN，通过位置精炼特征金字塔提升遥感目标检测性能。

### 2.4 小结与不足

综合现有研究，可以观察到：

1. 不同方法常在不同数据与训练设定下报告结果，缺少统一的可复现对比。
2. 论文方法在工程集成与落地演示方面资料不完备，影响复现实验与应用验证。
3. 针对中分辨率遥感微小船舶检测的研究相对较少，尤其是碎云等复杂背景下的检测问题。

因此，本课题拟围绕统一数据与指标、复现强基线、开展多模型公平对比与可视化分析，并实现可运行的检测系统原型，形成可交付、可复现的完整成果链路。

---

## 三、研究内容与目标

### 3.1 研究内容

结合任务要求与工程可交付性，本课题研究内容包括：

1. **数据集与评价体系构建**：基于 LEVIR-Ship 数据集梳理目录结构与标注格式，转换为统一的中间格式（COCO），明确评价指标与评测流程。

2. **基线复现**：复现参考论文方法 DRENet 的训练、评估与推理可视化流程。

3. **多模型对比实验**：在同一数据集与评测脚本下，完成 DRENet + 1 个通用检测器（mmdetection 生态）+ 1 个 YOLO（ultralytics）模型的训练测试对比，输出指标表与定性可视化。

4. **消融实验与误差分析**：围绕影响微小目标检测效果的关键因素（输入分辨率、多尺度训练、数据增强或阈值/NMS 等），完成 1–2 组消融实验，并整理难例分析。

5. **系统设计与实现**：实现可切换 ≥3 模型的推理系统，支持单图/批量推理、结果可视化与结构化导出，形成可演示 Demo。

### 3.2 研究目标

**实验目标**：

- 在 LEVIR-Ship 上完成 ≥3 种模型的统一训练/测试与对比分析；
- 形成至少 1 张"主对比表"（mAP、AP50、Recall 等）与 1 份可视化样例集（成功/失败/难例）。

**系统目标**：

- 支持模型选择、推理与结果导出；
- 形成可演示的最小可用系统（MVP），用于答辩展示。

**文档与论文目标**：

- 完成 20–30 篇论文的分类调研与"研究现状"成章；
- 论文写作具备可复现证据链（实验记录、配置、指标与可视化）。

---

## 四、技术路线与方法方案

### 4.1 数据集与预处理方案

**数据集**：LEVIR-Ship（3876 张图像，3219 个船舶实例，空间分辨率 16m/pixel）。

**数据集划分**：

- 训练集：2320 张图像（2002 个目标）
- 验证集：788 张图像（665 个目标）
- 测试集：788 张图像（552 个目标）

**标注格式**：水平边界框（bbox），单类别（船舶）。

**统一格式策略**：优先转换为 COCO 作为中间格式，分别适配 mmdetection 与 ultralytics YOLO 的数据输入要求，降低多框架协作成本。

### 4.2 评价指标与评测流程

**检测指标**：

- AP50（IoU > 0.5 的平均精度，与 DRENet 论文保持一致）
- Precision、Recall、F1-Score

**效率指标**：

- FPS（每秒帧数）
- Params（参数量）
- FLOPs（浮点运算次数）

**评测流程**：

1. 在测试集上推理
2. 计算预测结果
3. 与真实标注对比
4. 计算评价指标
5. 生成可视化结果

### 4.3 模型方案

**模型 A（参考论文方法）**：DRENet（Chen et al., 2022）

- 基于 YOLOv5s 的轻量级骨干网络
- 退化重建增强器（DRE）：仅在训练阶段工作
- 跨阶段多头注意力（CRMA）模块
- 性能：AP 82.4，FPS 85，Params 4.79M

**模型 B（通用检测器）**：Faster R-CNN（Ren et al., 2015）

- 两阶段检测器
- 使用 mmdetection 框架实现
- 作为通用检测器的对比基线

**模型 C（工程与部署友好）**：YOLOv8（ultralytics）

- 单阶段检测器
- 推理速度快，适合实际部署
- 作为实时检测的对比基线

### 4.4 对比实验设计

**公平性保证**：

- 使用同一训练/验证/测试划分（官方划分）
- 统一评价脚本与指标定义；输出同格式结果
- 训练设置尽可能统一（输入尺寸 512×512、batch_size=16、epochs=500）
- 固定随机种子，记录数据版本、环境依赖与训练资源消耗

**对比维度**：

- 检测精度：mAP、AP50、Precision、Recall、F1
- 检测速度：FPS
- 模型复杂度：Params、FLOPs
- 不同场景下的性能：平静海面、薄云、厚云、强浪、碎云

### 4.5 消融实验计划

**消融实验 1：退化函数 F(·) 的选择**

- 对比线性函数、对数函数、指数函数
- 验证指数函数在退化重建中的优势

**消融实验 2：增强器结构设计**

- 对比不同数量的 RCAB（1个、2个）
- 对比不同的上采样倍数（×2、×4）
- 验证 1个 RCAB + 2倍上采样的最优配置

---

## 五、系统需求分析与总体设计

### 5.1 功能需求

**模型管理**：

- 可切换 ≥3 个模型（DRENet + Faster R-CNN + YOLOv8）
- 支持加载预训练权重

**推理能力**：

- 单张推理
- 批量推理（文件夹输入）

**结果输出**：

- 可视化图片（bbox+置信度叠加）
- 结构化结果导出（JSON/CSV）

**评测功能**：

- 对测试集运行评测
- 输出指标汇总表

### 5.2 非功能需求

**可复现安装**：

- 提供环境依赖说明（requirements.txt）
- 提供一键运行方式（conda/pip）

**运行环境**：

- GPU 优先（推荐 NVIDIA GPU，显存 ≥8GB）
- CPU 兼容（作为备选）

**性能目标**：

- 在推荐 GPU 上单张推理耗时 <0.5 秒（DRENet 和 YOLOv8 目标 <0.1 秒）
- 可视化与导出流程可一键完成

### 5.3 总体架构与模块划分

**数据层**：

- 图像读取
- 尺寸归一化
- 预处理

**模型适配层**：

- 对 DRENet/mmdetection/YOLO 封装统一接口 `predict(image, model_name, conf, iou)`

**推理与后处理**：

- 阈值筛选
- NMS（非极大值抑制）
- 结果格式化

**展示与导出**：

- 可视化绘制
- JSON/CSV 输出

**交互层**：

- CLI 命令行界面
- Web Demo（使用 Gradio，适合快速原型开发和答辩演示）

---

## 六、进度安排与预期成果

### 6.1 进度计划

**2026.02（开题准备）**

- 完成 20–30 篇论文收集与分类笔记
- 明确数据集与评测指标
- 完成开题报告与开题 PPT
- **里程碑**：开题答辩通过

**2026.03（模型跑通与预实验）**

- DRENet 跑通训练/评测/推理可视化
- Faster R-CNN 或 YOLOv8 至少 1 个模型跑通
- 输出首版指标与可视化
- **里程碑**：完成至少 2 个模型的预实验

**2026.04（对比实验）**

- 三模型完整训练/测试
- 完成主对比表与定性可视化集合
- **里程碑**：完成所有模型的对比实验

**2026.05（消融 + 系统集成 + 论文定稿）**

- 完成 1–2 组消融与难例分析
- 系统 MVP 集成三模型并可演示
- 完成论文撰写与查重前自检材料
- **里程碑**：系统 Demo 可演示，论文初稿完成

**2026.06（答辩与归档）**

- 完成答辩 PPT、系统演示
- 材料归档与复盘总结
- **里程碑**：答辩通过，所有材料归档

### 6.2 预期成果

**文献调研**：

- 20–30 篇论文清单与分类总结（可成章）

**实验结果**：

- ≥3 模型对比表
- 消融实验表
- 可视化结果集

**系统原型**：

- 可切换模型的推理与可视化导出 Demo

**论文与附件**：

- 论文定稿
- 开题/中期/答辩材料包
- 代码与文档归档

---

## 七、风险分析与备选方案

**算力不足/训练耗时过长**：

- **备选方案 1**：优先做预实验（小 epoch/轻量模型/降低输入尺寸），在保证可复现的前提下逐步扩大训练设置
- **备选方案 2**：使用云端算力资源（如 Colab、AutoDL），按需租用 GPU 资源
- **备选方案 3**：减少模型数量，优先完成 DRENet + YOLOv8 的对比实验，Faster R-CNN 作为可选补充

**数据格式适配困难**：

- **备选方案 1**：先统一转换为 COCO 作为中间格式，分别适配 mmdetection 与 YOLO
- **备选方案 2**：参考官方文档和开源代码，使用成熟的数据转换工具（如 `labelme2coco`）
- **备选方案 3**：若格式转换困难，直接使用各框架的原生格式，在评测时统一结果格式

**参考代码复现困难**：

- **备选方案 1**：以 mmdetection/YOLO 完成"≥3 模型对比"主线交付，同时保留 DRENet 复现作为增强
- **备选方案 2**：采用论文/仓库提供的预训练权重，直接进行推理和评测
- **备选方案 3**：若 DRENet 复现失败，使用 YOLOv5s + Focal Loss 作为替代基线

**指标提升不明显**：

- **备选方案 1**：突出"统一对比与分析 + 系统实现"的核心成果，不将"刷新 SOTA"作为唯一目标
- **备选方案 2**：增加消融实验和可视化分析，深入分析不同模型的优势和劣势
- **备选方案 3**：增加场景分类评测（平静海面、薄云、厚云、强浪、碎云），展示模型在不同场景下的性能差异

**系统开发时间不足**：

- **备选方案 1**：优先实现 CLI 命令行版本，Web Demo 作为可选扩展
- **备选方案 2**：简化系统功能，仅保留核心推理和可视化功能，批量推理和结果导出作为后续扩展
- **备选方案 3**：使用 Gradio 快速搭建 Web Demo，避免复杂的后端开发

---

## 参考文献

[1] Chen J, Chen K, Chen H, et al. A degraded reconstruction enhancement-based method for tiny ship detection in remote sensing images with a new large-scale dataset[J]. IEEE Transactions on Geoscience and Remote Sensing, 2022, 60: 1-14.

[2] Cheng Y, Yang X, Wang J, et al. Towards large-scale small object detection: Survey and benchmarks[J]. arXiv preprint arXiv:2207.14096, 2022.

[3] Ren S, He K, Girshick R, et al. Faster R-CNN: Towards real-time object detection with region proposal networks[C]//Advances in neural information processing systems. 2015: 91-99.

[4] Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2980-2988.

[5] Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 2117-2125.

[6] Tian Z, Shen C, Chen H, et al. FCOS: Fully convolutional one-stage object detection[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2019: 9627-9636.

[7] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

[8] Redmon J, Farhadi A. YOLOv3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.

[9] Carion N, Massa F, Synnaeve G, et al. End-to-end object detection with transformers[C]//European conference on computer vision. 2020: 213-229.

[10] Zhu X, Su W, Lu L, et al. Deformable detr: Deformable transformers for end-to-end object detection[J]. arXiv preprint arXiv:2010.04159, 2020.

[11] Zhang H, Li F, Liu S, et al. DINO: DETR with improved denoising anchor boxes for end-to-end object detection[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2023: 7948-7957.

[12] Woo S, Park J, Lee J Y, et al. CBAM: Convolutional block attention module[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 3-19.

[13] Hu J, Shen L, Sun G, et al. Squeeze-and-excitation networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132-7141.

[14] Liu S, Qi L, Qin H, et al. Path aggregation network for instance segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 8759-8768.

[15] Qu Y, Li Y, Wang J, et al. Deep active learning for remote sensing object detection[J]. arXiv preprint arXiv:2003.08793, 2020.

[16] Kim H, Lee S, Park J, et al. NBBOX: Noisy bounding box improves remote sensing object detection[J]. arXiv preprint arXiv:2409.09424, 2024.

[17] Li Z, Zhang Y, Liu X, et al. Large selective kernel network for remote sensing object detection[J]. arXiv preprint arXiv:2303.09030, 2023.

[18] Wu Y, Chen L, Zhang H, et al. MTU-Net: Multi-level TransUNet for space-based infrared tiny ship detection[J]. arXiv preprint arXiv:2209.13756, 2022.

[19] Ramos J, Silva R, Santos L, et al. A decade of you only look once (YOLO) for object detection: A review[J]. arXiv preprint arXiv:2504.18586, 2025.

[20] Terven D, Musa A, Kyriacou A, et al. A comprehensive review of YOLO architectures in computer vision: From YOLOv1 to YOLOv8 and YOLO-NAS[J]. arXiv preprint arXiv:2304.00501, 2024.

[21] Li X, Wang Y, Zhang L, et al. LR-FPN: Enhancing remote sensing object detection with location refined feature pyramid network[J]. arXiv preprint arXiv:2404.01614, 2024.

[22] Wang Z, Liu Y, Chen H, et al. Multi-grained angle representation for remote sensing object detection[J]. arXiv preprint arXiv:2209.02884, 2022.

[23] Wang Y, Zhang L, Li X, et al. Learning oriented remote sensing object detection via naive geometric computing[J]. arXiv preprint arXiv:2112.00504, 2021.

[24] He Y, Liu Z, Wang X, et al. Learning remote sensing object detection with single point supervision[J]. arXiv preprint arXiv:2305.14141, 2023.

[25] Park J, Lee S, Kim H, et al. Investigating long-term training for remote sensing object detection[J]. arXiv preprint arXiv:2407.15143, 2024.

[26] Lin Y, Chen Z, Wang H, et al. OBBStacking: An ensemble method for remote sensing object detection[J]. arXiv preprint arXiv:2209.13369, 2022.

[27] Huang Y, Zhang L, Wang X, et al. MutDet: Mutually optimizing pre-training for remote sensing object detection[J]. arXiv preprint arXiv:2407.09920, 2024.

[28] Gao Y, Liu Z, Chen H, et al. Dual-stream spectral decoupling distillation for remote sensing object detection[J]. arXiv preprint arXiv:2512.04413, 2025.

[29] Zhang L, Wang Y, Chen H, et al. FDM-YOLO: A lightweight model FDM-YOLO for small target improvement based on YOLOv8[J]. arXiv preprint arXiv:2503.04452, 2025.

[30] Hussain M, Ahmed K, Ali S, et al. YOLOv5, YOLOv8 and YOLOv10: The go-to detectors for real-time vision[J]. arXiv preprint arXiv:2407.02988, 2024.
