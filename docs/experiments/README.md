# Experiments Index

This folder stores experiment standards, templates, execution playbooks, and experiment logs.

## Keep And Use
- `formal_experiment_spec_v1.md`
- `formal_experiment_log_template_v1.md`
- `docker_portability_runbook.md`
- `3060_execution_playbook.md`
- `cloud_execution_playbook.md`
- `logs/` (single source of truth for run records)
- `archive/` (deprecated docs, reference only)

## Redundancy Cleanup
- Removed duplicated note: `drenet_smoke_record_20260307.md`
- Keep all factual records under `logs/exp-*.md` only.
- Moved deprecated docs to `archive/`:
  - `exp_log_template.md`
  - `next_run_plan.md`
  - `drenet_reproduction_guide.md`
  - `drenet_local_reproduction_guide.md`

## Existing Logs
- `logs/exp-20260306-01-drenet.md`
- `logs/exp-20260307-02-drenet-smoke-wandb.md`
- `logs/exp-20260307-03-drenet-formal-issues-and-watch.md`
- `logs/exp-20260307-04-cloud-migration-manual-copy-and-next-steps.md`

## Next Steps
1. Sync `datasets/runs/checkpoints` to host archive (Mac).
2. Run cloud `resume` smoke test for 1-2 epochs on the same commit.
3. Start formal long-run after smoke passes.
4. Backfill `docs/results/baselines.md` and related result docs.
5. Keep one new log per run in `logs/` and avoid duplicate standalone notes.

## Maintenance Rules
- Facts go to `logs/`; standards stay in spec/template files.
- Prefer one focused doc commit at a time for traceability.
