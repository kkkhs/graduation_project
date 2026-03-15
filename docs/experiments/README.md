# Experiments Index

This folder stores experiment standards, templates, execution playbooks, and experiment logs.

## Keep And Use
- `formal_experiment_spec_v1.md`
- `formal_experiment_log_template_v1.md`
- `3060_execution_playbook.md`
- `cloud_execution_playbook.md`
- `plans/` (pending plans, not executed)
- `logs/` (single source of truth for executed records)
- `archive/` (deprecated docs, reference only)

## Redundancy Cleanup
- Removed duplicated note: `drenet_smoke_record_20260307.md`
- Keep all factual records under `logs/`.
- Moved deprecated docs to `archive/`:
  - `exp_log_template.md`
  - `next_run_plan.md`
  - `drenet_reproduction_guide.md`
  - `drenet_local_reproduction_guide.md`

## Naming Rules
- Plan docs: `plans/plan-YYYYMMDD-<topic>.md`
- Run logs: `logs/run-YYYYMMDD-<topic>.md`
- Audit logs: `logs/audit-YYYYMMDD-<topic>.md`
- Legacy `logs/exp-*.md` and `logs/local-command-audit-*.md` are kept for traceability.

## Quick Judgement Rule
- If the file contains only intended commands and no exit codes/results, it belongs to `plans/`.
- If the file includes executed commands, outputs, and artifacts, it belongs to `logs/`.

## Existing Logs
- `logs/exp-20260306-01-drenet.md`
- `logs/exp-20260307-02-drenet-smoke-wandb.md`
- `logs/exp-20260307-03-drenet-formal-issues-and-watch.md`
- `logs/exp-20260307-04-cloud-migration-manual-copy-and-next-steps.md`
- `logs/exp-20260307-05-autodl-prep.md`
- `logs/exp-20260308-01-drenet-3080ti-resume-smoke.md`
- `logs/exp-20260308-02-drenet-formal-resume-300.md`
- `logs/run-20260308-yolo26-3060-3080ti.md`
- `logs/run-20260315-fcos-wandb-single-run-epoch-log.md`

## Existing Plans
- `plans/plan-next-run.md`

## Progress Summary
- Active: `progress_summary_20260315.md`
- Archived: `archive/progress_summary_20260308.md`
- Additional archived prep log: `archive/run-20260309-fcos-coco-prep-nogpu.md`

## Next Steps
1. Complete cross-model result table finalization (add FPS/Params/FLOPs).
2. Finalize qualitative page and at least one ablation entry.
3. Apply default deployment checkpoint policy:
   - paper: `global-best`
   - system: `stable-best`
4. Keep one new log per run in `logs/` and avoid duplicate standalone notes.

## Maintenance Rules
- Plans go to `plans/`; facts go to `logs/`; standards stay in spec/template files.
- Prefer one focused doc commit at a time for traceability.
