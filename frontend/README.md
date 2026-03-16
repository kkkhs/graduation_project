# 前端说明

前端基于 React + Vite + AntD，用于任务提交、结果展示与模型管理。

## 启动方式
```bash
./scripts/start_frontend.sh
```

## 页面结构
- `/`：任务提交（TaskSubmitPage）
- `/tasks`：任务列表
- `/tasks/:id`：任务详情与可视化
- `/models`：模型管理

## 关键调用
- `/api/v1/models`
- `/api/v1/tasks/infer`
- `/api/v1/tasks/{id}`
- `/api/v1/tasks/{id}/results`
