status: executed
evidence: commands + outputs + artifacts

# 运行日志：FCOS 定性样例导出（2026-03-16）

## 1. 目标
- 导出 FCOS `global-best@ep14` 与 `stable@ep102` 的定性图像（success/miss/false_positive）
- 同步到本地并整理到论文可用目录

## 2. 远端环境
- 主机：`connect.westb.seetacloud.com:29137`
- 环境：`/root/miniconda3/envs/fcos310`
- 数据集：`/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/annotations/test.json`
- 图片：`/root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/test/images`

## 3. 导出脚本
- `/root/autodl-tmp/workspace/graduation_project/tools/export_mmdet_qualitative.py`

## 4. 执行命令
### 4.1 global-best（ep14）
```bash
mkdir -p /root/autodl-tmp/experiment_assets/qualitative/fcos_main_fixedcfg_20260315_160824
/root/miniconda3/bin/conda run -n fcos310 python /root/autodl-tmp/workspace/graduation_project/tools/export_mmdet_qualitative.py \
  --config /root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/fcos_main_fixedcfg_20260315_160824.py \
  --checkpoint /root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/best_coco_bbox_mAP_50_epoch_14.pth \
  --ann /root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/annotations/test.json \
  --image-root /root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/test/images \
  --output-dir /root/autodl-tmp/experiment_assets/qualitative/fcos_main_fixedcfg_20260315_160824 \
  --score-thr 0.25 \
  --iou-thr 0.5 \
  --max-per-class 2 \
  --limit 300
```

### 4.2 stable（ep102）
```bash
mkdir -p /root/autodl-tmp/experiment_assets/qualitative/fcos_main_fixedcfg_20260315_160824_stable
/root/miniconda3/bin/conda run -n fcos310 python /root/autodl-tmp/workspace/graduation_project/tools/export_mmdet_qualitative.py \
  --config /root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/fcos_main_fixedcfg_20260315_160824.py \
  --checkpoint /root/autodl-tmp/experiment_assets/runs/mmdet/fcos_main_fixedcfg_20260315_160824/epoch_102.pth \
  --ann /root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/annotations/test.json \
  --image-root /root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship/test/images \
  --output-dir /root/autodl-tmp/experiment_assets/qualitative/fcos_main_fixedcfg_20260315_160824_stable \
  --score-thr 0.25 \
  --iou-thr 0.5 \
  --max-per-class 1 \
  --limit 300
```

## 5. 导出结果（远端）
- global-best：`/root/autodl-tmp/experiment_assets/qualitative/fcos_main_fixedcfg_20260315_160824/`
- stable：`/root/autodl-tmp/experiment_assets/qualitative/fcos_main_fixedcfg_20260315_160824_stable/`

## 6. 本地整理
- 按论文口径整理至：
  - `assets/figures/qualitative/fcos/global_best/`
  - `assets/figures/qualitative/fcos/stable/`

## 7. 备注
- `mmcv._ext` 缺失导致 `shipdet` 环境无法跑 FCOS，可用的稳定环境是 `fcos310`。
