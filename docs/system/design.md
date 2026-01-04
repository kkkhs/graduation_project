## 系统总体设计（架构与接口）

### 1. 总体架构（建议）
系统分为五层：
1) **输入层**：图像文件/文件夹  
2) **数据预处理层**：读取、尺寸调整、归一化、必要的切片策略（如后续需要大图切片）  
3) **模型适配层**：对接 DRENet / mmdet / YOLO，统一推理接口  
4) **后处理与结果层**：阈值过滤、NMS、结果标准化（COCO-style bbox）、保存  
5) **展示与交互层**：CLI/Web，展示可视化与导出下载

### 2. 模块划分
- `app/io.py`：输入输出、文件遍历、结果保存  
- `app/models/`：各模型适配器（drenet、mmdet、yolo）  
- `app/predictor.py`：统一入口（选择模型、调用、后处理）  
- `app/visualize.py`：绘制 bbox/score  
- `app/cli.py`：命令行入口  
- `app/web.py`（可选）：Web Demo（Gradio/Streamlit）

> 注：上述为建议的落地路径，最终以实现为准。开题阶段写清“接口与流程”即可。

### 3. 关键数据结构（建议统一）
#### 3.1 Detection 结果格式（建议）
- `image_path`: str
- `detections`: list，每个元素包含：
  - `bbox`: [x, y, w, h]（左上角 + 宽高）
  - `score`: float
  - `category_id`: int（若单类可固定为 1）
  - `category_name`: str（可选）

#### 3.2 统一推理接口（建议）
- `predict(image_path, model_name, conf_threshold, iou_threshold) -> result`

### 4. 推理流程（文字版，可直接放论文）
输入图像后，系统完成图像读取与预处理，依据用户选择加载对应模型权重并执行前向推理，随后进行阈值过滤与 NMS 等后处理，得到最终检测框与置信度。系统将结果以可视化图片与结构化文件两种形式输出，并记录本次推理参数用于复现。

