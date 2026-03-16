# 实验文档索引

本目录存放实验规范、模板、执行手册与实验日志。

## 保留与使用
- `formal_experiment_spec_v1.md`
- `formal_experiment_log_template_v1.md`
- `3060_execution_playbook.md`
- `cloud_execution_playbook.md`
- `plans/`（待执行计划，未落地）
- `logs/`（已执行事实记录的唯一来源）
- `archive/`（历史归档，仅供参考）

## 冗余清理
- 已移除重复记录：`drenet_smoke_record_20260307.md`
- 所有事实记录统一放在 `logs/`。
- 已将废弃文档移至 `archive/`：
  - `exp_log_template.md`
  - `next_run_plan.md`
  - `drenet_reproduction_guide.md`
  - `drenet_local_reproduction_guide.md`

## 命名规则
- 计划文档：`plans/plan-YYYYMMDD-<topic>.md`
- 运行日志：`logs/run-YYYYMMDD-<topic>.md`
- 审计日志：`logs/audit-YYYYMMDD-<topic>.md`
- 历史文件 `logs/exp-*.md` 与 `logs/local-command-audit-*.md` 保留用于追溯。

## 快速判断规则
- 只有计划命令、无退出码/结果：归入 `plans/`。
- 包含已执行命令、输出与产物：归入 `logs/`。

## 现有日志
- `logs/exp-20260306-01-drenet.md`
- `logs/exp-20260307-02-drenet-smoke-wandb.md`
- `logs/exp-20260307-03-drenet-formal-issues-and-watch.md`
- `logs/exp-20260307-04-cloud-migration-manual-copy-and-next-steps.md`
- `logs/exp-20260307-05-autodl-prep.md`
- `logs/exp-20260308-01-drenet-3080ti-resume-smoke.md`
- `logs/exp-20260308-02-drenet-formal-resume-300.md`
- `logs/run-20260308-yolo26-3060-3080ti.md`
- `logs/run-20260315-fcos-wandb-single-run-epoch-log.md`
- `logs/run-20260315-efficiency-prep-nogpu.md`
- `logs/run-20260316-efficiency-metrics-3080ti.md`
- `logs/run-20260316-fcos-512-clean-start.md`
- `logs/run-20260316-fcos-512-resume19-bs10.md`
- `logs/run-20260316-fcos-qualitative-export.md`

## 现有计划
- `plans/plan-next-run.md`
  - `plans/plan-20260316-fcos-512-clean-run.md` 已归档

## 阶段总结
- 当前：`progress_summary_20260315.md`
- 归档：`archive/progress_summary_20260308.md`
- 额外归档准备记录：`archive/run-20260309-fcos-coco-prep-nogpu.md`
- 归档计划：`archive/plan-20260316-fcos-512-clean-run.md`

## 下一步
1. 完成三模型主对比表（以当前接受的基线为准）。
2. 完成定性页并补至少 1 条消融。
3. 确定默认部署权重策略：
   - 论文：`global-best`
   - 系统：`stable-best`
4. 将 FCOS 作为参考基线，并明确输入口径差异说明。

## 维护规则
- 计划放 `plans/`，事实放 `logs/`，规范类留在 spec/template。
- 文档修改尽量一次只聚焦一件事，方便追溯。
