# 固定演示样例清单

本文档用于固定答辩与最终演示使用的原图、参数和讲解口径。样例分组按“展示稳定性优先”组织，而不是纯按学术难度划分。

## 1. 使用原则

- 主演示优先使用 `easy`
- 追加强调系统完整链路时使用 `medium`
- `hard` 仅用于问答或展示系统边界，不作为首页首演示
- 所有样例均为可直接上传到 Web 系统的原图，不使用论文拼板图替代

## 2. 样例清单

| case_id | image_path | recommended_model | recommended_mode | score_thr | difficulty_label | expected_result | talking_points | fallback_asset |
|---|---|---|---|---|---|---|---|---|
| demo_easy_01 | `assets/demo_cases/easy/GF6_WFV_E131.2_N35.8_20200910_L1A1120034678-3_1536_16384.png` | `yolo` | `single` | `0.25` | `easy` | 结果直观、适合快速完成“提交 -> 完成 -> 详情”闭环 | 适合开场展示系统主链路，强调界面响应、任务状态流转和结果回看 | `outputs/ui-submit-after-polish-20260315.png` / `outputs/ui-tasks-after-polish-20260315.png` / `outputs/ui-detail-fused-last-20260315.png` |
| demo_medium_01 | `assets/demo_cases/medium/GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_8704_1536.png` | `drenet` | `single` | `0.25` | `medium` | 可用于展示第二个真实算法接入能力 | 适合说明“系统不是只接了一个模型”，并展示结构化结果表和任务详情页 | `outputs/ui-tasks-progress-right-20260315.png` |
| demo_hard_01 | `assets/demo_cases/hard/GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_10976_9728.png` | `mmdet_fcos` | `single` | `0.25` | `hard` | 适合说明复杂场景下的漏检/边界表现，不作为首演示 | 用于回答“系统如何展示失败或难例”“为什么不同模型适合不同场景”等问题 | `assets/figures/qualitative/fcos/stable/miss/fcos_stable_miss_01.jpg` |

## 3. 推荐现场顺序

1. `demo_easy_01`：主演示，优先保证闭环稳定。
2. `demo_medium_01`：补充展示多模型接入能力。
3. `demo_hard_01`：仅在老师追问时展开，说明系统可以承载难例分析。

## 4. 与现有材料的对应关系

- 中期演示脚本：`docs/midterm/midterm_demo_script_20260315.md`
- 系统测试报告：`docs/system/test_report_20260315.md`
- 定性分析素材：`docs/results/qualitative.md`
