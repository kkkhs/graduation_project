# 脚本总览

本目录用于存放系统启动、同步、运维类脚本。

## 常用脚本
- `start_backend.sh`：启动后端服务
- `start_frontend.sh`：启动前端服务
- `init_db.sh`：初始化数据库并同步模型配置
- `dev_one_click.sh`：一键启动（后端 + 前端）

## 说明
脚本默认在项目根目录执行，确保 `PYTHONPATH` 与数据库路径正确。
