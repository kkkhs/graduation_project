# 中期答辩现场演示脚本（3-5分钟）

## 0. 演示目标
演示“提交任务 -> 状态流转 -> 结果查看”的系统闭环，并说明已接入 YOLO26 与 DRENet 两个算法。

## 1. 启动服务（提前2-3分钟）
```bash
cd /Users/khs/codes/graduation_project
bash scripts/init_db.sh
bash scripts/start_backend.sh
bash scripts/start_frontend.sh
```

检查：
- 前端：`http://127.0.0.1:5173`
- 健康接口：`http://127.0.0.1:8000/api/v1/health`

## 2. 主演示路径（YOLO26）
1. 打开“任务提交”页。
2. 选择 `单图`，模式选 `单模型`，模型选 `YOLO26`，阈值保持默认 `0.25`。
3. 上传固定样例：`assets/demo_cases/easy/GF6_WFV_E131.2_N35.8_20200910_L1A1120034678-3_1536_16384.png`，然后点击提交。
4. 跳转“任务列表”，口播：
   - 状态机：`queued -> running -> done`
   - 说明系统具备异步任务能力，不阻塞界面。
5. 打开该任务“详情页”，展示：
   - 可视化检测结果图
   - 结构化结果表（框坐标、分数、类别）
   - 任务信息（创建时间、进度、状态）

## 3. 备选演示（DRENet）
1. 回到“任务提交”页。
2. 切换模型为 `DRENet`，其余参数保持不变。
3. 使用固定样例：`assets/demo_cases/medium/GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_8704_1536.png`
3. 重复提交和查看详情流程，强调“系统已接入至少2个真实算法”。

若老师继续追问复杂场景，可补充展示：
- `assets/demo_cases/hard/GF1_WFV4_E124.0_N33.5_20131122_L1A0000115971_10976_9728.png`
- 说明该样例更适合讲“系统边界与难例分析”，不作为首个演示流程。

## 4. 讲解要点（现场口播短句）
- “中期阶段重点是系统完成度与代码量，不追求最终最优指标。”
- “现在已经具备前端、后端、数据库和算法集成的完整链路。”
- “任务可回放、结果可追溯，满足中期检查要求。”

## 5. 兜底方案（服务波动时）
若现场推理失败，立即切换静态截图讲解同一闭环：
- 提交页：`/Users/khs/codes/graduation_project/outputs/ui-submit-after-polish-20260315.png`
- 列表页：`/Users/khs/codes/graduation_project/outputs/ui-tasks-after-polish-20260315.png`
- 详情页：`/Users/khs/codes/graduation_project/outputs/ui-detail-fused-last-20260315.png`

口播：
“演示流程与系统在线流程一致，展示的是同一版本同一数据链路。”
