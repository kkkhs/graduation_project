# src 目录说明（分层架构）

- `domain/`：核心实体与抽象接口，不依赖具体框架。
- `application/`：业务用例层，编排预测流程与输出标准化。
- `infrastructure/`：外部依赖实现（配置、适配器、可视化）。
- `contracts/`：结果 schema 等契约文件。

推荐入口：
- CLI：`tools/run_predict.py`
- 命令行可视化：`tools/visualize_predict.py`
- 桌面 UI（Qt）：`tools/desktop_ui_qt.py`
