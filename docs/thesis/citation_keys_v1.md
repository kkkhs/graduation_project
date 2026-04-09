# 引用 Key 清单与审计结果

说明：当前正文使用的引用已不再是“临时占位 key”状态，本文件记录最近一次本地审计结果与当前实际使用的 key 集合。

## 已使用 key
- `carion2020_detr`
- `chen2022_tinyship_drenet`
- `girshick2014_rcnn`
- `lin2017_focalloss`
- `lin2017_fpn`
- `mmdet2018_openmmlab`
- `najibi2019_autofocus`
- `redmon2016_yolo`
- `redmon2018_yolov3`
- `ren2015_fasterrcnn`
- `singh2018_snip`
- `singh2018_sniper`
- `tan2020_efficientdet`
- `tian2019_fcos`
- `ultralytics2024_docs`
- `wang2023_sod_review`
- `yang2019_scrdet`
- `zhang2023_dino`
- `zhao2024_shipsurvey`
- `zhu2021_deformabledetr`

## 统一命名规则
- 格式：`author_year_topic`
- 要求：
  - 小写字母与下划线
  - 不含空格与中文字符
  - 同一文献仅保留一个主 key

## 当前审计结果（2026-04-09）
- 正文实际使用 key 数量：`20`
- `thesis_overleaf/chapters/*.tex` 中所有 `\cite{}` 均能在 `thesis_overleaf/bib/hfut.bib` 中找到
- 当前正文已不再使用历史命名 `levirship2021_dataset`
- 本轮补强后，相关工作与方法部分的引用覆盖范围已从基础模型扩展到小目标检测综述、尺度感知策略和遥感船舶检测综述

## 当前需要注意的点
- `zhao2024_shipsurvey` 当前采用 Remote Sensing 综述条目，用于支撑遥感船舶检测现状与部署趋势描述
- 若最终提交前还要继续扩充文献，优先补“遥感船舶检测”与“统一评测/部署”方向，不建议再大规模改动已有 key
