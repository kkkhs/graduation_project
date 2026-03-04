# 实验记录模板（论文可引用版）

> 每次训练或评测都复制一份。建议文件名：`exp-YYYYMMDD-xx.md`

## 1. 基本信息
- 实验编号：
- 日期：
- 模型：`DRENet / Faster R-CNN / FCOS / YOLOv8`
- 代码版本（commit）：
- 数据版本：LEVIR-Ship（下载来源+日期）
- 随机种子：

## 2. 训练配置
- 输入尺寸：
- batch size：
- epoch / iters：
- 优化器与学习率策略：
- 数据增强：
- 后处理参数：`conf`、`iou`、NMS 设置

## 3. 环境与资源
- GPU / 显存：
- CPU / 内存：
- 训练总耗时：
- 推理速度（FPS）：

## 4. 结果（定量）
- AP50：
- AP50:95（如有）：
- Precision：
- Recall：
- F1：
- 备注（是否 early stop、是否异常）：

## 5. 结果（定性）
- 成功案例路径：
- 漏检案例路径：
- 误检案例路径：
- 复杂场景说明（碎云/强浪/近岸）：

## 6. 结论
- 本次实验结论（1-3条）：
- 与上次实验差异：
- 下一步动作：

## 7. 复现命令
```bash
# train

# eval

# infer / visualize
```
