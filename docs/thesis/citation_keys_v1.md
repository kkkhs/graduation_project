# 引用 Key 清单与审计结果

说明：当前正文已不再是“临时占位 key”状态，本文件同时记录已使用 key 与最近一次本地审计结果。

## 已使用 key
- `chen2022_tinyship_drenet`
- `ren2015_fasterrcnn`
- `lin2017_fpn`
- `lin2017_focalloss`
- `tian2019_fcos`
- `redmon2016_yolo`
- `redmon2018_yolov3`
- `carion2020_detr`
- `zhu2021_deformabledetr`
- `zhang2023_dino`
- `levirship2021_dataset`
- `mmdet2018_openmmlab`
- `ultralytics2024_docs`

## 统一命名规则
- 格式：`author_year_topic`
- 要求：
  - 小写字母与下划线
  - 不含空格与中文字符
  - 同一文献仅保留一个主 key

## 当前审计结果（2026-04-09）
- 正文实际使用 key 数量：`13`
- `thesis_overleaf/chapters/*.tex` 中所有 `\cite{}` 均能在 `thesis_overleaf/bib/hfut.bib` 中找到
- 当前 `hfut.bib` 未发现“正文缺失 key”或“明显未使用模板残留 key”

## 当前需要注意的点
- `levirship2021_dataset` 这一 key 当前仍沿用历史命名，但条目内容与 DRENet / LEVIR-Ship 对应论文一致
- 若最终提交前需要进一步统一风格，优先做“命名注释补充”，不要临时大规模改 citation key，避免牵动全文引用
