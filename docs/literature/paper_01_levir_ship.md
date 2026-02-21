# 论文阅读笔记：LEVIR-Ship

## 基本信息

| 项目           | 内容                                                                                                                               |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **论文标题**   | A Degraded Reconstruction Enhancement-based Method for Tiny Ship Detection in Remote Sensing Images with A New Large-scale Dataset |
| **中文标题**   | 基于退化重建增强的遥感图像微小船舶检测方法及大规模数据集                                                                           |
| **作者**       | Jianqi Chen, Keyan Chen, Hao Chen, Zhengxia Zou, Zhenwei Shi                                                                       |
| **作者单位**   | 北京航空航天大学 图像处理中心                                                                                                      |
| **发表期刊**   | IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)                                                                     |
| **发表时间**   | 2022年                                                                                                                             |
| **期刊级别**   | 中科院1区TOP期刊，CCF-B类，影响因子约8.6                                                                                           |
| **IEEE链接**   | https://ieeexplore.ieee.org/document/9791363                                                                                       |
| **数据集**     | LEVIR-Ship                                                                                                                         |
| **数据集链接** | https://github.com/WindVChen/LEVIR-Ship                                                                                            |
| **代码链接**   | https://github.com/WindVChen/DRENet                                                                                                |
| **实验室**     | 北京航空航天大学 LEVIR Lab                                                                                                         |

---

## 一句话问题

如何在中分辨率（MR，约16m/pixel）遥感图像中有效检测微小船舶目标，解决目标尺寸小（可能仅占20像素）、背景复杂（如碎云）、分辨率有限等问题？

---

## 核心方法

### 1. DRENet 网络架构

DRENet 由三个主要部分组成：

#### (1) Backbone（骨干网络）

- 基于 YOLOv5s（轻量级版本）
- 高效提取多尺度图像特征
- 输出特征图尺寸：16×16、32×32、64×64、64×64

#### (2) Degraded Reconstruction Enhancer（退化重建增强器）

- 仅在训练阶段工作，推理时不产生计算开销
- 通过重建退化图像引导骨干网络关注微小船舶目标
- 使用单个 RCAB（Residual Channel Attention Block）
- 从 64×64 特征图重建 128×128 退化标签

#### (3) Detector（检测器）

- 引入 CRMA（CRoss-stage Multi-head Attention）模块
- 利用自注意力机制提升特征判别能力
- 获得大感受野以准确定位船舶
- 节省计算量

### 2. Selective Degradation（选择性退化）

#### 核心思想

生成一个"伪显著性图"，其中船舶保持完整，背景被模糊处理。

#### 算法流程

```
输入：图像 I，所有真实边界框集合 G
输出：退化图像 Î

对于图像中每个像素 (i, j)：
1. 计算该像素到所有船舶目标的最小距离 Dmin
2. 使用递增函数 F(Dmin) 调整均值滤波核大小
3. 应用相应大小的均值滤波
4. 对结果进行最近邻插值调整大小
```

#### 关键参数

- F(·)：递增函数，论文中使用指数函数效果最佳
- 核大小随距离增加而增大，离船舶越远模糊程度越高

### 3. CRMA 模块（跨阶段多头注意力）

#### 结构组成

- 多头自注意力（MHSA）
- 卷积层
- 拼接操作

#### 优势

- 获得大感受野
- 准确定位船舶
- 节省计算量

---

## 数据集

### LEVIR-Ship 数据集

| 项目           | 内容                                        |
| -------------- | ------------------------------------------- |
| **数据集名称** | LEVIR-Ship                                  |
| **发布时间**   | 2021年                                      |
| **发布机构**   | 北京航空航天大学 LEVIR Lab                  |
| **卫星来源**   | GaoFen-1（高分一号）和 GaoFen-6（高分六号） |
| **图像总数**   | 3876 张                                     |
| **图像尺寸**   | 512×512 像素                                |
| **空间分辨率** | 16m/pixel（中分辨率）                       |
| **目标总数**   | 3219 个船舶实例                             |
| **目标特点**   | 微小目标（在 GF-1 图像中可能仅占 20 像素）  |
| **标注格式**   | 边界框标注（BB）                            |

### 数据集划分

| 集合   | 图像数量 | 目标数量 |
| ------ | -------- | -------- |
| 训练集 | 2320     | 2002     |
| 验证集 | 788      | 665      |
| 测试集 | 788      | 552      |

### 数据集特点

1. **中分辨率**：16m/pixel，适合快速大范围监测
2. **复杂场景**：包含不同云量、陆地比例、光照强度、海面特征
3. **微小目标**：船舶在图像中占比很小
4. **挑战性**：碎云、强浪等复杂背景导致检测困难

### 与其他数据集对比

| 数据集      | 图像数量 | 边界框数量 | 来源                 | 分辨率    | 年份 |
| ----------- | -------- | ---------- | -------------------- | --------- | ---- |
| NWPU VHR-10 | 57       | 302        | Google Earth         | 0.5-2m    | 2014 |
| HRSC2016    | 1070     | 2976       | Google Earth         | 0.4-2m    | 2016 |
| DIOR        | 2702     | 62400      | Google Earth         | 0.5-30m   | 2018 |
| HRRSD       | 2165     | 3886       | Google Earth & Baidu | 0.15-1.2m | 2019 |
| LEVIR-SHIP  | 3876     | 3219       | GaoFen-1 & GaoFen-6  | 16m       | 2021 |

---

## 指标/对比

### 评价指标

- **AP**：平均精度（Average Precision）
- **FPS**：每秒帧数（Frames Per Second）
- **Params**：参数量（推理时）
- **FLOPs**：浮点运算次数（推理时）

### 对比方法

#### 通用目标检测方法

- YOLOv3
- YOLOv5s
- RetinaNet (ResNet50)
- SSD (VGG16)
- FasterRCNN (VGG16)
- EfficientDet-D0/D2
- FCOS (ResNet50)
- CenterNet (Hourglass-104)

#### 船舶检测专用方法

- HSFNet
- ImYOLOv3
- MaskRCNN (ResNet50) + DFR + RFE

### 实验结果

#### 整体性能对比

| 方法                      | 参数量（推理） | FLOPs（推理） | AP       | FPS    |
| ------------------------- | -------------- | ------------- | -------- | ------ |
| YOLOv3                    | 61.52M         | 99.2G         | 69.9     | 61     |
| YOLOv5s                   | 7.05M          | 10.4G         | 75.6     | 95     |
| RetinaNet (ResNet50)      | 36.33M         | 104.4G        | 74.9     | 12     |
| SSD (VGG16)               | 24.39M         | 175.2G        | 52.6     | 25     |
| FasterRCNN (VGG16)        | 136.70M        | 299.2G        | 70.8     | 10     |
| EfficientDet-D0           | 3.84M          | 4.6G          | 71.3     | 32     |
| EfficientDet-D2           | 8.01M          | 20.0G         | 80.9     | 21     |
| FCOS (ResNet50)           | 5.92M          | 51.8G         | 75.5     | 37     |
| CenterNet (Hourglass-104) | 191.24M        | 584.6G        | 77.7     | 25     |
| HSFNet                    | 157.59M        | 538.1G        | 73.6     | 7      |
| ImYOLOv3                  | 62.86M         | 101.9G        | 72.6     | 51     |
| MaskRCNN + DFR + RFE      | 24.99M         | 237.8G        | 76.2     | 6      |
| **DRENet (ours)**         | **4.79M**      | **8.3G**      | **82.4** | **85** |

#### 不同背景场景下的性能

| 场景          | 图像数量 | YOLOv5s | EfficientDet-D2 | ImYOLOv3 | Method [12] | DRENet |
| ------------- | -------- | ------- | --------------- | -------- | ----------- | ------ |
| calm sea      | 262      | 76.8    | 83.0            | 75.9     | 78.4        | 82.1   |
| thin cloud    | 238      | 84.3    | 83.9            | 83.7     | 82.4        | 87.3   |
| thick cloud   | 60       | 60.5    | 78.9            | 56.6     | 70.1        | 86.8   |
| strong wave   | 101      | 73.4    | 73.3            | 62.8     | 71.8        | 82.8   |
| fractus cloud | 127      | 72.1    | 74.7            | 61.8     | 64.4        | 76.5   |

### 消融实验

#### 1. 退化函数 F(·) 的选择

对比了线性函数、对数函数和指数函数，**指数函数**效果最佳，能够更明显地进行退化处理，更好地满足增强器突出船舶和模糊背景的需求。

#### 2. 增强器结构设计

| 结构          | n-RCAB | Upsampling | AP   | Params (Train) | FLOPs (Train) |
| ------------- | ------ | ---------- | ---- | -------------- | ------------- |
| YOLOv5s       | -      | -          | 75.6 | 7.05M          | 10.4G         |
| YOLOv5s + DRE | 1      | ×2         | 76.8 | 8.25M          | 20.3G         |
| YOLOv5s + DRE | 1      | ×4         | 76.2 | 8.84M          | 40.0G         |
| YOLOv5s + DRE | 2      | ×2         | 76.6 | 8.55M          | 22.7G         |
| YOLOv5s + DRE | 2      | ×8         | 75.6 | 9.73M          | 121.2G        |

**结论**：1个RCAB + 2倍上采样效果最佳。

#### 3. 不同监督形式

| 标签生成方法                             | AP   |
| ---------------------------------------- | ---- |
| Original (YOLOv5s)                       | 75.6 |
| Down-sampling                            | 76.6 |
| Sudden-blur + Down-sampling              | 75.2 |
| Continuity-blur (α=1.01) + Down-sampling | 76.8 |

**结论**：使用 "Selective Degradation"（Continuity-blur）生成标签效果最好。

---

## 主要结论

1. **方法有效性**：DRENet 在 LEVIR-Ship 数据集上达到 82.4 AP 和 85 FPS，在准确性和速度之间取得了良好平衡
2. **数据集价值**：LEVIR-Ship 是首个公开的中分辨率遥感船舶检测数据集，填补了该领域的空白
3. **性能提升**：相比最接近的竞争对手 EfficientDet-D2，DRENet 的 AP 高出 1.5 个点，同时速度快约 4 倍（85 vs 21 FPS）
4. **鲁棒性**：在复杂背景（如碎云、厚云、强浪）下，DRENet 表现出更强的鲁棒性
5. **轻量化**：DRENet 参数量仅 4.79M，FLOPs 仅 8.3G，适合实际部署

---

## 对本课题的价值

### 1. 作为参考基线复现

- **复现价值**：论文提供了完整的代码实现（DRENet），可直接复现
- **对比基线**：可以作为你论文中的主要对比基线方法
- **实验参考**：实验设置、训练参数、评估指标可作为参考

### 2. 数据集使用

- **直接使用**：LEVIR-Ship 数据集可以直接用于你的实验
- **对比实验**：在相同数据集上对比不同方法
- **评估标准**：使用论文中的评估指标（AP、FPS、Params、FLOPs）

### 3. 方法借鉴

- **退化重建思想**：可以借鉴退化重建增强的思想来提升微小目标检测
- **选择性退化**：Selective Degradation 操作可以应用于其他场景
- **注意力机制**：CRMA 模块的设计思路可以借鉴
- **轻量化设计**：学习如何在保持性能的同时减少计算量

### 4. 论文写作参考

- **研究现状**：可以作为"遥感/船舶检测"部分的重要参考文献
- **实验设计**：参考论文的实验设计、对比方法和消融实验
- **结果展示**：学习如何展示实验结果、可视化对比和性能分析
- **数据集介绍**：参考如何详细介绍新数据集

---

## 下一步行动

### 1. 获取论文全文

- [x] 已获取 PDF 文件：E:\Codes\Githubs\pdfs\2022_jianqi_chen_a.pdf
- [x] 已提取文本内容：E:\Codes\Githubs\graduation_project\pdfs\2022_jianqi_chen_a.txt
- [x] 已整理论文笔记：docs/literature/paper_01_levir_ship.md

### 2. 获取数据集

- [ ] 访问 GitHub：https://github.com/WindVChen/LEVIR-Ship
- [ ] 下载 LEVIR-Ship 数据集
- [ ] 了解数据集结构和标注格式
- [ ] 创建数据集文档：docs/dataset.md

### 3. 获取代码

- [ ] 访问 GitHub：https://github.com/WindVChen/DRENet
- [ ] 下载 DRENet 代码
- [ ] 搭建复现环境
- [ ] 理解代码结构

### 4. 复现实验

- [ ] 配置环境依赖
- [ ] 在 LEVIR-Ship 上训练 DRENet
- [ ] 记录实验结果和指标
- [ ] 可视化检测结果

---

## 关键技术点总结

### 1. 为什么选择中分辨率图像？

- **覆盖范围广**：16m/pixel 的 MR 图像比 1m/pixel 的 HR 图像覆盖面积大 256 倍
- **检测效率高**：对于固定海域，使用 MR 图像只需 1 小时，而 HR 图像需要 256 小时（约 10 天）
- **实际应用需求**：满足大范围海域快速监测和预警的实际需求

### 2. 微小船舶检测的挑战

- **纹理稀缺**：船舶纹理信息少，边缘模糊
- **尺寸极小**：在 GF-1 图像（16m/pixel）中，船舶可能仅占 20 像素
- **背景复杂**：碎云等复杂成像条件导致误检
- **特征不明显**：难以从背景中区分船舶

### 3. DRENet 的创新点

- **退化重建增强器**：仅在训练阶段工作，推理时零计算开销
- **选择性退化**：生成伪显著性图，突出船舶，模糊背景
- **跨阶段多头注意力**：提升特征判别能力，获得大感受野
- **轻量化设计**：参数量仅 4.79M，FLOPs 仅 8.3G

---

## 引用格式

```
@article{chen2022degraded,
  title={A degraded reconstruction enhancement-based method for tiny ship detection in remote sensing images with a new large-scale dataset},
  author={Chen, Jianqi and Chen, Keyan and Chen, Hao and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--14},
  year={2022},
  publisher={IEEE}
}
```

---

## 相关资源

- **LEVIR Lab 官网**：https://levir.buaa.edu.cn/
- **mmdetection**：https://github.com/open-mmlab/mmdetection
- **ultralytics（YOLO）**：https://docs.ultralytics.com/zh/
- **YOLOv5**：https://github.com/ultralytics/yolov5
