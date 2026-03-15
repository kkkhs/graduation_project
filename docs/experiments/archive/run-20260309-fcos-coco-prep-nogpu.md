status: executed
evidence: commands + outputs + blockers

# 运行日志：FCOS 前置准备（无卡实例，2026-03-09）

## 1. 目标
- 在无卡模式下完成 FCOS 训练前最关键的可并行准备：
  - COCO 标注生成（train/val/test）
  - 路径与数据可用性核验
  - 环境可训练性排查（GPU/CUDA/MMDet 依赖）

## 2. 执行环境
- 主机：`ssh -p 29137 root@connect.westb.seetacloud.com`
- 系统：Ubuntu 22.04（AutoDL）
- 资源状态：
  - `GPU: No devices were found`
  - `CPU: 0.5 core`
  - `Memory: 2 GB`

## 3. 已执行事项

### 3.1 检查转换脚本
- 脚本路径：
  - `/root/autodl-tmp/workspace/graduation_project/tools/convert_yolo_to_coco.py`
- 逻辑确认：
  - YOLO `class=0` 映射到 COCO `category_id=1`
  - `categories=[{"id":1,"name":"ship"}]`

### 3.2 生成 COCO 标注（远端）
- 目标目录：
  - `/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/annotations/`
- 执行结果：
  - `train.json`: `images=2321`, `annotations=2002`
  - `val.json`: `images=788`, `annotations=665`
  - `test.json`: `images=788`, `annotations=552`
- 文件大小：
  - `train.json` 约 `502K`
  - `val.json` 约 `168K`
  - `test.json` 约 `167K`

### 3.3 结构与类别校验
- `categories` 校验通过：`[{"id":1,"name":"ship"}]`
- `annotations.category_id` 校验通过：仅出现 `[1]`

### 3.4 MMDet 依赖准备（无卡可执行）
- 在 `shipdet` 环境补齐并校验：
  - `mmengine==0.10.7`
  - `mmcv==2.1.0`（`mmcv-lite`）
  - `mmdet==3.3.0`
- 说明：
  - 无卡阶段采用 `mmcv-lite`，避免 CUDA 扩展编译依赖。

## 4. 遇到的问题与结论

### 4.1 问题：当前实例无卡
- 现象：
  - `nvidia-smi` 无设备
  - `torch.cuda.is_available() == False`
- 结论：
  - 可做数据准备与配置检查，但不能进行 FCOS 冒烟/正式训练

### 4.2 问题：MMDet 依赖未安装
- 现象（系统 Python）：
  - `mmengine/mmcv/mmdet` 均 `ModuleNotFoundError`
- 结论：
  - 切到有卡实例后需先补齐 MMDet 依赖，再开训练

### 4.3 问题：初始依赖安装链路版本冲突
- 现象：
  - 通过 `openmim + mim install mmcv` 首次安装时，`openmim` 相关依赖将 `setuptools` 降级到 `60.2.0`
  - 触发 `mmcv` 构建失败（缺 `pkg_resources`）
  - 后续安装 `mmcv-lite 2.2.0` 与 `mmdet 3.3.0` 冲突（`mmdet` 要求 `<2.2.0`）
- 处理：
  - 升级 `setuptools`
  - 改用 `mmcv-lite` 路线
  - 固定版本 `mmcv-lite==2.1.0`，与 `mmdet==3.3.0` 对齐
- 当前状态：
  - 依赖冲突已解除，可进入有卡实例直接做 FCOS 冒烟

## 5. 下一步计划（切回有卡实例后）
1. 进入有卡实例，确认 `nvidia-smi` 与 `torch.cuda.is_available()` 为可用
2. 复核 `shipdet` 环境版本（`mmengine/mmcv/mmdet`）与数据路径
3. 跑 FCOS `1 epoch` 冒烟
4. 冒烟通过后启动正式训练并写 run 日志

## 6. 产物路径
- COCO 标注目录：
  - `/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/annotations/`
- 文件：
  - `/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/annotations/train.json`
  - `/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/annotations/val.json`
  - `/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/annotations/test.json`
