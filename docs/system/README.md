# 系统文档总览

本目录集中维护系统实现相关文档。建议按“入口 → 细节 → 证据”的顺序阅读。

## 推荐阅读顺序
1. `web_system_upgrade.md`：Web 系统总体说明与启动入口。
2. `architecture.md`：系统分层架构与数据流说明（含 Mermaid 图）。
3. `system_flow_detailed.md`：端到端细节流程（请求 → 推理 → 落盘 → 展示）。
4. `predict_api_contract.md`：接口契约与字段说明。
5. `database_er.md`：SQLite 数据结构。
6. `result_sync_flow.md`：结果回传与产物同步流程。

## 其他说明
- `local_mvp_run.md`：本机脚本式推理入口（非 Web）。
- `desktop_ui_guide.md`：桌面 UI 使用说明。
- `python_visualization_guide.md`：可视化脚本入口。
- `test_report_20260315.md`：联调验证证据。
