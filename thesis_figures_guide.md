# 毕业论文插图生成指南

> 本文档将论文插图分为两类：
>
> - **Mermaid格式**：流程图、架构图、ER图等可用Mermaid代码直接生成
> - **图像生成Prompt**：包含遥感图像的定性分析图、复杂统计图表等，需使用gpt-image-2等模型生成

---

## 一、Mermaid格式图表（可直接代码转换）

### 1. ch1_pipeline - 第一章整体技术路线图

```mermaid
flowchart TB
    subgraph 顶层流程[" "]
        A[数据准备<br/>核对图像与标注] --> B[模型训练与推理<br/>DRENet / YOLO26 / FCOS]
        B --> C[结果分析<br/>定量与定性分析]
        C --> D[系统集成<br/>Web部署与演示]
    end

    subgraph 底层支撑[" "]
        E[统一协议<br/>数据划分、指标与阈值] --> F[统一推理API<br/>单模型/融合推理]
        F --> G[系统证据链<br/>任务回放、结果复查与历史记录]
    end

    B -.->|约束训练与评估| E
    C -.->|输入统一接口| F
    D -.->|服务化呈现| G

    G --> H[论文结果、系统演示和任务历史共享同一协议]
```

**风格参数：**

- 节点背景色: `#F7F7F7`
- 强调背景色: `#EFEFEF`
- 边框色: `#222222`
- 文字色: `#222222`
- 字体: Times New Roman / Songti SC

---

### 2. ch2_research_map - 第二章研究脉络图

```mermaid
flowchart TB
    subgraph 研究路线["三条研究路线"]
        A[通用目标检测<br/>两阶段/单阶段/无锚点/Transformer]
        B[小目标检测<br/>多尺度融合/损失重加权/后处理]
        C[遥感舰船检测<br/>LEVIR-Ship/专项方法]
    end

    subgraph 具体方法["代表方法"]
        A1[Faster R-CNN] & A2[YOLO / RetinaNet] & A3[FCOS / DETR] <-->|细化| A
        B1[FPN / SCRDet] & B2[Focal Loss] & B3[SNIP / NMS] <-->|技术支撑| B
        C1[LEVIR-Ship数据集] & C2[DRENet专项] <-->|数据集/方法| C
    end

    D[本文切入点<br/>统一协议下的多模型对比+系统实现]

    A -.->|跨范式比较基础| D
    B -.->|性能差距分析| D
    C -.->|小舰船场景聚焦| D
```

---

### 3. ch3_drenet_overview - DRENet模型架构图

```mermaid
flowchart LR
    A[输入图像<br/>中分辨率遥感影像] --> B[骨干网络<br/>多尺度特征提取]
    B --> C[CRMA模块<br/>跨阶段注意力]
    C --> D[检测头<br/>分类与边界框回归]
    D --> E[输出<br/>船舶检测结果]

    subgraph 训练专用分支["训练专用退化重建分支"]
        F[DRE分支<br/>退化重建] --> G[重建特征图<br/>目标感知退化图像]
        G --> H[监督目标<br/>退化图像约束]
    end

    B -.->|共享特征| F
    G -.->|特征引导| C
    F --> G
    G -.->|重建损失| H
```

---

### 4. ch3_fcos_pipeline - FCOS检测流程图

```mermaid
flowchart LR
    A[输入图像] --> B[骨干网络<br/>特征提取]
    B --> C[FPN<br/>多层级特征]
    C --> D[FCOS检测头<br/>共享卷积塔]
    D --> E[解码边界框<br/>无锚点输出]

    D --> F[分类<br/>类别得分]
    D --> G[回归<br/>左/上/右/下]
    D --> H[中心度<br/>质量先验]
```

---

### 5. ch3_yolo_pipeline - YOLO检测流程图

```mermaid
flowchart LR
    A[输入图像<br/>512主设置] --> B[骨干网络<br/>特征提取]
    B --> C[特征融合层<br/>多尺度融合]
    C --> D[检测头<br/>单阶段预测]
    D --> E[输出<br/>NMS结果]

    D --> F[小尺度<br/>精细目标]
    D --> G[中尺度<br/>平衡尺度]
    D --> H[大尺度<br/>上下文支持]
```

---

### 6. ch3_preprocess_flow - 数据预处理与统一格式流程

```mermaid
flowchart LR
    A[核对图像与标注<br/>一一对应检查<br/>空标签/异常坐标] --> B[COCO中间格式<br/>统一标注格式<br/>跨框架通用]
    B --> C[固定数据划分<br/>训练集/验证集/测试集<br/>固定样本列表]
    C --> D[统一输出结构<br/>图像ID/边界框<br/>得分/类别ID]
    D --> E[系统可调用<br/>推理API/数据库<br/>可视化/回放]

    C --> F[统一协议：差异仅来自检测范式与实现策略<br/>而非数据组织方式或脚本细节]
```

---

### 7. ch5_architecture - 系统整体架构图

```mermaid
flowchart TB
    subgraph Frontend["前端层"]
        F1[React + Vite + AntD]
        F2[提交/任务]
        F3[详情/模型]
        F4[轮询/可视化]
    end

    subgraph Backend["后端层"]
        B1[FastAPI路由<br/>/api/v1] --> B2[任务执行器<br/>ThreadPoolExecutor]
        B2 --> B3[推理运行时<br/>统一预测器]
        B3 --> B4[SQLAlchemy服务<br/>任务/结果持久化]
    end

    subgraph Model["模型层"]
        M1[DRENet]
        M2[YOLO26]
        M3[FCOS]
    end

    subgraph Data["数据与输出层"]
        D1[SQLite<br/>模型/任务/结果]
        D2[outputs/tasks/&lt;id&gt;<br/>原始/可视化/JSON产物]
    end

    Frontend --> Backend
    B3 --> Model
    B3 --> Data
    B4 --> D1
```

---

### 8. ch5_flow - 系统业务流程图

```mermaid
flowchart TB
    A[1. 用户提交图像与参数] --> B[2. FastAPI验证请求并创建任务]
    B --> C[3. 任务执行器派发本地推理任务]
    C --> D[4. 运行时调用DRENet/YOLO26/FCOS]
    D --> E[5. 结果写入数据库与输出目录]
    E --> F[6. 前端轮询任务状态与结果接口]
    F --> G[7. 用户查看进度、可视化与结果]

    subgraph StateMachine["状态机"]
        S1[排队中→运行中]
        S2[运行中→完成]
        S3[运行中→失败]
    end

    C -.-> StateMachine

    subgraph Artifacts["产物"]
        Art1[原始图像]
        Art2[可视化图像]
        Art3[JSON与结果行]
    end

    E -.-> Artifacts
```

---

### 9. ch5_er - 数据库ER关系图

```mermaid
erDiagram
    MODELS ||--o{ TASKS : "配置来源"
    TASKS ||--o{ RESULTS : "一对多"
    TASKS ||--o{ TASK_FILES : "一对多"

    MODELS {
        int id PK
        string name
        string key
        string weight_path
        boolean is_enabled
        datetime created_at
    }

    TASKS {
        int id PK
        string type
        string status
        string mode
        string model_key
        float score_thr
        int input_count
        int done_count
        string error_code
        string error_message
        datetime created_at
        datetime started_at
        datetime finished_at
    }

    RESULTS {
        int id PK
        int task_id FK
        string image_name
        string source_model
        boolean is_fused
        float bbox_x1
        float bbox_y1
        float bbox_x2
        float bbox_y2
        float score
        int category_id
    }

    TASK_FILES {
        int id PK
        int task_id FK
        string kind
        string path
    }
```

---

## 二、图像生成Prompt（需使用gpt-image-2等模型）

以下图片包含遥感影像内容或复杂数据可视化，建议使用图像生成模型生成。

### 10. ch3_dataset_examples - 数据集样本展示

**Prompt:**

```
Create a 3x2 grid figure showing satellite remote sensing image samples from LEVIR-Ship dataset.
Each panel shows a 512x512 medium-resolution satellite image (GF-1 WFV sensor) with 1-3 small green bounding boxes marking ship targets.
Style: Academic thesis figure, clean white background, Times New Roman font style.
Top-left corner of each panel has a small label "GT: N" (N=1,2,3 indicating ground truth count).
Images show coastal/maritime scenes with varying water textures.
Color palette: Blue-green ocean water, gray-white ships, bright green (#50DC50) bounding boxes with 2px width.
Layout: 3 columns (horizontal), 2 rows (vertical), with 16px gap between panels.
Professional academic publication quality, high resolution.
```

---

### 11. ch4_main_comparison_chart - 主实验结果对比柱状图

**Prompt:**

```
Create a professional academic bar chart comparing three ship detection models (DRENet, YOLO26, FCOS) across 5 metrics: AP50, AP50-95, Precision, Recall, F1.
Layout: Grouped bar chart, 3 bars per metric group, 5 metric groups on X-axis.
Colors: DRENet in medium gray (#8C8C8C), YOLO26 in light gray (#B0B0B0), FCOS in lighter gray (#D0D0D0).
Y-axis range: 0 to 1.0, labeled "Score".
X-axis labels: "AP50", "AP50-95", "Precision", "Recall", "F1".
Each bar has value labels on top (3 decimal places), rotated 90 degrees.
Legend at top center, frameless, 3 columns.
Note at bottom right: "* FCOS recall uses AR@100 from formal run; Precision and F1 unavailable".
Style: Clean academic chart, white background, Times New Roman style fonts, no top/right borders.
Add light gray horizontal grid lines (-- style, alpha 0.25).
```

---

### 12. ch4_threshold_trend - 置信度阈值影响趋势图

**Prompt:**

```
Create a 1x3 subplot figure showing confidence threshold impact on detection performance for three models.
Each subplot is a line chart with X-axis: threshold values [0.15, 0.25, 0.35], Y-axis: Score [0.60, 0.88].
Three lines per subplot:
- Precision: steel blue (#566D7E) with circle markers
- Recall: sage green (#6E7F63) with circle markers
- F1: brown (#8A6A46) with circle markers
Each data point has value label annotation (3 decimals) with smart offset to avoid overlap.
Subplot titles: "DRENet", "FCOS", "YOLO26" (top center of each subplot).
Common legend at top center of figure (frameless, 3 columns).
Grid: Light gray dashed lines, alpha 0.25.
Style: Academic publication quality, white background, Times New Roman fonts, no top/right spines.
X-axis label (shared): "Confidence threshold", Y-axis label (shared): "Score".
```

---

### 13. ch4_size_trend - 输入尺寸影响趋势图

**Prompt:**

```
Create a 1x2 subplot figure showing inference image size impact on FCOS and YOLO26 performance.
Each subplot: Line chart with X-axis: image sizes [512, 640, 800], Y-axis: Score [0.40, 0.88].
Three lines per subplot:
- AP50: steel blue (#566D7E)
- Recall: sage green (#6E7F63)
- F1: brown (#8A6A46)
All lines have circle markers (size 6) and 2.3px width.
Each point has value label (3 decimals) with offset: Precision(-14,10), Recall(12,10), F1(0,-14).
Labels have white background boxes (alpha 0.95) to ensure readability.
Subplot titles: "FCOS", "YOLO26".
Common legend at top center (frameless, 3 columns).
X-axis ticks at [512, 640, 800], Y-axis shared between subplots.
Grid: Light gray dashed, alpha 0.25. No top/right spines.
Style: Academic quality, Times New Roman, white background.
```

---

### 14. ch4_efficiency_chart - 效率与复杂度对比图

**Prompt:**

```
Create a grouped bar chart comparing three models (DRENet, YOLO26, FCOS) on efficiency metrics.
X-axis: 3 metrics - "FPS", "Params(M)", "FLOPs(G)".
Y-axis: Score values (linear scale, auto-range).
3 bars per metric group:
- DRENet: medium gray (#8C8C8C)
- YOLO26: light gray (#B0B0B0)
- FCOS: lighter gray (#D0D0D0)
Each bar has value label on top: "121.90", "4.79", "4.21" format (2 decimals for <10, 1 for >=10).
Legend: Top center, frameless, 3 columns.
Y-axis label: "Score" (or appropriate units).
Grid: Light gray horizontal dashed lines, alpha 0.25.
No top/right spines. Clean academic style, Times New Roman, white background.
```

---

### 15. ch4_radar_chart - 多维度雷达图

**Prompt:**

```
Create a polar/radar chart comparing three ship detection models across 8 dimensions.
Dimensions (clockwise from top): AP50, AP50-95, Precision, Recall, F1, FPS norm, Params eff, FLOPs eff.
Three polygons:
- DRENet: medium gray (#8C8C8C), line width 2.0, fill alpha 0.15
- YOLO26: light gray (#B0B0B0), line width 2.0, fill alpha 0.15
- FCOS: lighter gray (#D0D0D0), line width 2.0, fill alpha 0.15
Radial axis: 0 to 1.0, ticks at 0.2, 0.4, 0.6, 0.8, 1.0.
Axis labels: Gray (#555555), size 8.
Dimension labels on perimeter: Size 10.
Legend: Upper right, frameless, outside main plot area.
Style: Academic quality, white/light gray background, clean circular grid.
```

---

### 16. ch4_success_cases - 检测成功案例定性分析

**Prompt:**

```
Create a qualitative analysis figure showing successful ship detection results.
Layout: 3 columns (models: DRENet, YOLO26, FCOS) × 2 rows (Label row, Prediction row).
Each panel: 520×520 satellite image tile from LEVIR-Ship dataset.
Row 1 (Label): Original image with GREEN bounding boxes showing ground truth ships.
Row 2 (Prediction): Same image with colored detection boxes:
- DRENet column: Red boxes (#FF5A5A)
- YOLO26 column: Blue boxes (#46AAFF)
- FCOS column: Yellow boxes (#FFC846)
Column headers: Model names centered above top row.
Row labels on left: "Label", "Prediction" vertically centered.
Style: Academic figure, white background, clean layout with 20px gaps.
Images show successful detection cases with minimal false positives/negatives.
Satellite imagery: GF-1 WFV sensor, coastal scenes, blue-green water.
```

---

### 17. ch4_miss_cases - 漏检案例定性分析

**Prompt:**

```
Create a qualitative analysis figure showing ship detection miss cases (false negatives).
Layout: Same 3×2 grid as success cases (3 model columns, Label/Prediction rows).
Each panel: 520×520 satellite image.
Label row: Green boxes showing ALL ground truth ships that SHOULD be detected.
Prediction row: Model-specific colored boxes showing detected ships, with CLEARLY VISIBLE missing targets (undetected ships still visible in image but without boxes).
Column headers: "DRENet", "YOLO26", "FCOS".
Row labels: "Label", "Prediction".
Purpose: Highlight cases where models fail to detect small or ambiguous ships.
Style: Academic figure, clean layout, Times New Roman style text.
Select challenging cases: small ships, low contrast, clustered targets.
```

---

### 18. ch4_false_positive_cases - 误检案例定性分析

**Prompt:**

```
Create a qualitative analysis figure showing false positive ship detection cases.
Layout: 3×2 grid (3 models, Label/Prediction rows), 520×520 panels.
Label row: Green boxes showing actual ships (usually 0-2 targets).
Prediction row: Colored boxes showing model predictions with EXTRA boxes marking false detections (background objects incorrectly classified as ships).
Colors:
- Ground truth: Bright green (#50DC50)
- DRENet predictions: Red (#FF5A5A)
- YOLO26 predictions: Blue (#46AAFF)
- FCOS predictions: Yellow (#FFC846)
Show cases where models detect non-ship objects (waves, clouds, coastlines, artifacts) as ships.
Purpose: Compare false positive patterns across three detection paradigms.
Style: Academic publication quality, clean white background.
```

---

### 19. ch5_pages - 系统界面多页展示

**Prompt:**

```
Create a multi-page screenshot composite showing a web-based ship detection system interface.
Layout: 2×2 or 3×2 grid showing different system pages:
1. Task submission page: Upload area, model selection dropdown, threshold slider
2. Task list page: Table with status (queued/running/done), progress indicators
3. Task detail page: Image grid, detection results overlay, download buttons
4. Visualization page: Side-by-side original/detected comparison, zoom controls
Style: Modern web UI, React + Ant Design aesthetic.
Color scheme: Clean whites, light grays (#F5F5F5), accent blue (#1890FF) for primary actions.
Typography: System fonts, 14px body, 16px headers.
Include realistic content: satellite thumbnails, bounding box overlays, confidence scores.
Academic figure quality with subtle drop shadows and rounded corners (4px radius).
```

---

## 三、图表分类汇总表

| 图号 | 文件名                    | 类型         | 生成方式  | 复杂度 | 状态   |
| ---- | ------------------------- | ------------ | --------- | ------ | ------ |
| 1    | ch1_pipeline              | 技术路线图   | Mermaid   | 中     | 已完成 |
| 2    | ch2_research_map          | 研究脉络图   | Mermaid   | 中     | 已完成 |
| 3    | ch3_drenet_overview       | 模型架构图   | Mermaid   | 高     | 已完成 |
| 4    | ch3_fcos_pipeline         | 检测流程图   | Mermaid   | 中     | 已完成 |
| 5    | ch3_yolo_pipeline         | 检测流程图   | Mermaid   | 中     | 已完成 |
| 6    | ch3_preprocess_flow       | 数据处理流程 | Mermaid   | 中     | 已完成 |
| 7    | ch5_architecture          | 系统架构图   | Mermaid   | 高     | 已完成 |
| 8    | ch5_flow                  | 业务流程图   | Mermaid   | 中     | 已完成 |
| 9    | ch5_er                    | ER关系图     | Mermaid   | 低     | 已完成 |
| 10   | ch3_dataset_examples      | 数据集样本   | Image Gen | 中     | 已完成 |
| 11   | ch4_main_comparison_chart | 对比柱状图   | Image Gen | 中     | 已完成 |
| 12   | ch4_threshold_trend       | 趋势折线图   | Image Gen | 高     | 已完成 |
| 13   | ch4_size_trend            | 趋势折线图   | Image Gen | 中     | 已完成 |
| 14   | ch4_efficiency_chart      | 效率对比图   | Image Gen | 低     | 已完成 |
| 15   | ch4_radar_chart           | 雷达图       | Image Gen | 中     | 已完成 |
| 16   | ch4_success_cases         | 定性分析图   | Image Gen | 高     | 已完成 |
| 17   | ch4_miss_cases            | 定性分析图   | Image Gen | 高     | 已完成 |
| 18   | ch4_false_positive_cases  | 定性分析图   | Image Gen | 高     | 已完成 |
| 19   | ch5_pages                 | 界面截图     | Image Gen | 中     | 已完成 |

---

## 四、使用说明

### Mermaid图表生成

1. 将上述Mermaid代码复制到 [Mermaid Live Editor](https://mermaid.live) 或支持Mermaid的Markdown渲染器
2. 导出为PNG/SVG格式
3. 使用统一的配色方案确保风格一致

### 图像生成模型使用

1. 将Prompt复制到gpt-image-2或其他图像生成模型
2. 建议参数：
   - 分辨率：根据原图尺寸（通常 1200×800 或更高）
   - 风格：Academic/Professional
   - 背景：纯白 (#FFFFFF)
3. 如有需要，可先生成草图再迭代优化

### 输出规范

| 场景         | 推荐格式 | 分辨率      | 备注             |
| ------------ | -------- | ----------- | ---------------- |
| LaTeX (矢量) | SVG/PDF  | 矢量        | 最佳质量，文件小 |
| LaTeX (位图) | PNG      | 300+ DPI    | 需确保高清源文件 |
| Word/PPT     | PNG/EMF  | 150-300 DPI | 兼容性优先       |
| 网页展示     | SVG/PNG  | 72-150 DPI  | 压缩后使用       |

### Mermaid图表生成

1. 将上述Mermaid代码复制到 [Mermaid Live Editor](https://mermaid.live) 或支持Mermaid的Markdown渲染器
2. 导出为PNG/SVG格式
3. 使用统一的配色方案确保风格一致

### 风格统一要求

- **配色**：
  - 主灰色系: #8C8C8C (深), #B0B0B0 (中), #D0D0D0 (浅)
  - 强调色: #566D7E (蓝灰), #6E7F63 (绿灰), #8A6A46 (棕)
  - 背景: #FFFFFF, 次要背景: #F7F7F7
- **箭头**：1.2px，深灰 (#222222)，箭头大小 8-10px
- **字体**：Times New Roman, Songti SC (中文), 10-12pt
- **边框**：1.2px 深灰 (#222222)，圆角 2-4px
- **输出格式**：
  - 矢量图优先：SVG/PDF（无限缩放不失真）
  - 位图备用：PNG，300 DPI 以上
- **分辨率要求**：印刷级 300-600 DPI，屏幕展示 150 DPI 即可
