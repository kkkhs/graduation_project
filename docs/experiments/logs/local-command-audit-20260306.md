# 本机会话命令审计（2026-03-06）

> 说明：本文件记录本次会话中我在本机实际执行过的命令、用途与关键结果（含失败命令）。

## A. 用于“实验准备判断”的依据命令

1. `rg --files /Users/khs/codes/graduation_project | rg 'docs/experiments|configs|tools|tests|outputs/predictions'`
- 结果：确认实验入口、配置、测试与结果模板文件存在。

2. `sed -n '1,260p' /Users/khs/codes/graduation_project/README.md`
- 结果：确认本仓库支持三模型统一推理、融合推理、可视化与测试命令。

3. `sed -n '1,260p' /Users/khs/codes/graduation_project/docs/experiments/README.md`
- 结果：确认实验流程总规范与命令来源。

4. `sed -n '1,320p' /Users/khs/codes/graduation_project/docs/experiments/3060_execution_playbook.md`
- 结果：确认 3060 训练机完整命令链（冒烟/正式/评测/回传）。

5. `sed -n '1,260p' /Users/khs/codes/graduation_project/docs/results/baselines.md`
- 结果：发现结果表为模板，尚未填实测数据。

6. `sed -n '1,260p' /Users/khs/codes/graduation_project/docs/results/ablation.md`
- 结果：发现消融表为模板，尚未填实测数据。

7. `sed -n '1,260p' /Users/khs/codes/graduation_project/docs/results/qualitative.md`
- 结果：发现定性模板已建，但案例未回填。

8. `sed -n '1,260p' /Users/khs/codes/graduation_project/docs/spec_todo.md`
- 结果：确认 D9~D14（正式实验与回填）仍为待完成。

9. `sed -n '1,260p' /Users/khs/codes/graduation_project/configs/models.yaml`
- 结果：`weight_path/config_path` 仍为占位路径，不能直接做真实推理。

10. `sed -n '1,260p' /Users/khs/codes/graduation_project/configs/models.example.yaml`
- 结果：配置结构完整，需训练后替换真实路径。

11. `PYTHONPATH=/Users/khs/codes/graduation_project python3 -m unittest discover -s /Users/khs/codes/graduation_project/tests -p 'test_*.py'`
- 结果：`Ran 12 tests ... OK`。

12. `ls -la /Users/khs/codes/graduation_project/docs/experiments`
- 结果：确认实验文档目录与 logs 子目录存在。

13. `sed -n '1,240p' /Users/khs/codes/graduation_project/docs/experiments/exp_log_template.md`
- 结果：确认日志模板字段完整，可直接填报首轮实验。

## B. 本次会话中其余执行命令（含翻译阶段）

### B1. 文件检索/阅读
- `rg --files`
- `sed -n '1,220p' /Users/khs/.codex/skills/pdf/SKILL.md`
- `sed -n '1,220p' docs/literature/paper_01_levir_ship.md`
- `sed -n '1,260p' docs/literature/paper_01_levir_ship_translation.md`
- `wc -l docs/literature/paper_01_levir_ship.md docs/literature/paper_01_levir_ship_translation.md`
- `sed -n '240,360p' docs/literature/paper_01_levir_ship_translation.md`
- `wc -l pdfs/2022_jianqi_chen_a.txt && sed -n '1,180p' pdfs/2022_jianqi_chen_a.txt`
- `rg -n "^(I|II|III|IV|V|VI|A\\.|B\\.|C\\.|D\\.|E\\.|F\\.)" pdfs/2022_jianqi_chen_a.txt | sed -n '1,140p'`
- `sed -n '260,520p' pdfs/2022_jianqi_chen_a.txt`
- `sed -n '575,960p' pdfs/2022_jianqi_chen_a.txt`
- `sed -n '520,575p' pdfs/2022_jianqi_chen_a.txt`
- `sed -n '1,260p' docs/kaiti/kaiti.md`
- `nl -ba docs/literature/paper_01_levir_ship_translation.md | sed -n '1,220p'`
- `nl -ba docs/literature/paper_01_levir_ship_translation.md | sed -n '220,360p'`
- `sed -n '136,188p' docs/literature/paper_01_levir_ship_translation.md`
- `git diff -- docs/literature/paper_01_levir_ship_translation.md`（结果为空）
- `git status --short`
- `ls -l docs/kaiti/基座论文翻译-DRENet.md && wc -l docs/kaiti/基座论文翻译-DRENet.md`

### B2. 依赖安装与环境检查
- `python3 - <<'PY' ... importlib ... reportlab/pypdf ... PY`
  - 结果：`reportlab False`, `pypdf False`
- `uv pip install reportlab pypdf`
  - 结果：失败，`zsh: command not found: uv`
- `python3 -m pip install --user reportlab pypdf`
  - 结果：成功安装。
- `python3 - <<'PY' ... check fonts ... PY`
  - 结果：`/Library/Fonts/Arial Unicode.ttf True`
- `python3 - <<'PY' ... TTFont(Hiragino...) ... PY`
  - 结果：失败，`postscript outlines are not supported`
- `python3 - <<'PY' ... TTFont('/Library/Fonts/Arial Unicode.ttf') ... PY`
  - 结果：成功，输出 `ok`
- `pdftoppm -png tmp/pdfs/font_test.pdf tmp/pdfs/font_test`
  - 结果：失败，`pdftoppm: command not found`
- `python3 -m pip install --user deep-translator`
  - 结果：成功安装。
- `python3 - <<'PY' from deep_translator ... translate(...) ... PY`
  - 结果：成功返回中文句子。
- `python3 - <<'PY' ... importlib ... pdfplumber ... PY`
  - 结果：`pdfplumber False`
- `python3 -m pip install --user pdfplumber`
  - 结果：成功安装。

### B3. PDF/翻译构建与验证命令
- `python3 - <<'PY' ... 生成 source_copy + translation_pages + merged_pdf ... PY`
  - 结果：生成成功，打印 `orig_pages=14 tran_pages=10 merged_pages=24`。
- `python3 - <<'PY' ... PdfReader(merged) ... PY`
  - 结果：`pages 24`，末页文本可提取。
- `sed -n '1,240p' /Users/khs/.codex/skills/pdf/SKILL.md`
- `sed -n '1,260p' scripts/read_pdf.py`
- `rg -n "pdf|translate|translation|bilingual|docx" -S scripts docs | sed -n '1,240p'`
- `python3 - <<'PY' ... PdfReader(...).extract_text() ... PY`
- `python3 - <<'PY' ... pdfplumber.extract_text_lines ... PY`
- `python3 scripts/build_bilingual_pdf.py --input ... --output ...`
  - 结果：首次执行成功，输出 `paragraphs=124` 与文件路径。
- `ls -lh ...段落对照翻译.pdf ...bilingual.md ...translate_cache.json`
- `python3 - <<'PY' ... 校验段落对照翻译.pdf 页数和文本 ... PY`
  - 结果：`pages 20`，可提取文本。
- `sed -n '1,220p' .../tmp/pdfs/2022_jianqi_chen_a_bilingual.md`
- `python3 - <<'PY' ... pdfplumber 左右栏裁剪查看 ... PY`
- `python3 - <<'PY' ... pdfminer extract_text ... PY`（多次）
- `python3 scripts/build_bilingual_pdf.py ...`
  - 结果：失败，`No such file or directory`（路径错误）。
- `ls -l /Users/khs/codes/graduation_project/scripts | sed -n '1,120p'`
- `rg --files /Users/khs/codes/graduation_project | rg 'build_bilingual_pdf.py'`
  - 结果：未找到。
- `git -C /Users/khs/codes/graduation_project status --short`
- `find /Users/khs -name build_bilingual_pdf.py 2>/dev/null | sed -n '1,20p'`
  - 结果：无输出。

### B4. 状态盘点命令
- `git -C /Users/khs/codes/graduation_project status --short`
- `ls -lh /Users/khs/codes/graduation_project/output/pdf | sed -n '1,120p'`
- `ls -lh /Users/khs/codes/graduation_project/docs/experiments/logs | sed -n '1,120p'`
- `find /Users/khs/codes/graduation_project/outputs -maxdepth 3 -type f | sed -n '1,200p'`
- `find /Users/khs/codes/graduation_project/tmp -maxdepth 3 -type f 2>/dev/null | sed -n '1,200p'`
- `find /Users/khs/codes/graduation_project/pdfs -maxdepth 2 -type f | sed -n '1,200p'`

## C. 本次会话由我新增/修改的文件（确认）

1. 新增：[next_run_plan.md](/Users/khs/codes/graduation_project/docs/experiments/next_run_plan.md)
2. 新增：[exp-20260306-01-drenet.md](/Users/khs/codes/graduation_project/docs/experiments/logs/exp-20260306-01-drenet.md)
3. 新增（审计）：[local-command-audit-20260306.md](/Users/khs/codes/graduation_project/docs/experiments/logs/local-command-audit-20260306.md)

## D. 与“实验工作”直接相关的结论

- 我在本机没有执行三模型训练命令，也没有执行真实权重推理；仅完成了代码与流程可执行性核验（文档检查 + 配置检查 + 单测 12/12）。
- 因此当前实验数值（AP50/P/R/F1）仍需在 3060 训练机按 `docs/experiments/3060_execution_playbook.md` 产出后再回填。

## E. 本轮文档优化与脚本修改（2026-03-06）

### E1. 读取与检查命令
- `sed -n '1,260p' /Users/khs/codes/graduation_project/README.md`
- `sed -n '1,260p' /Users/khs/codes/graduation_project/docs/experiments/3060_execution_playbook.md`
- `sed -n '1,220p' /Users/khs/codes/graduation_project/docs/experiments/drenet_reproduction_guide.md`
- `sed -n '1,220p' /Users/khs/codes/graduation_project/scripts/sync_results_from_laptop.sh`

### E2. 本轮修改目标
- 优化 `README.md`
- 重写 `docs/experiments/3060_execution_playbook.md`
- 重写 `docs/experiments/drenet_reproduction_guide.md`
- 增强 `scripts/sync_results_from_laptop.sh`

### E3. 校验命令
- `bash -n /Users/khs/codes/graduation_project/scripts/sync_results_from_laptop.sh`
  - 结果：通过，无语法错误。
- `rg -n "环境预检|数据目录预检|数据转换验收|冒烟通过判定|指标提取与保存|回传前文件清单|故障分流表|ship.yaml 示例|首轮最小成功定义" /Users/khs/codes/graduation_project/docs/experiments /Users/khs/codes/graduation_project/README.md -S`
  - 结果：新增章节均已命中。
- `git -C /Users/khs/codes/graduation_project diff -- README.md docs/experiments/3060_execution_playbook.md docs/experiments/drenet_reproduction_guide.md scripts/sync_results_from_laptop.sh`
  - 结果：确认 4 个目标文件均已更新。

## F. 云端适配补充（2026-03-06）

### F1. 读取命令
- `sed -n '1,260p' /Users/khs/codes/graduation_project/docs/experiments/README.md`
- `sed -n '1,220p' /Users/khs/codes/graduation_project/README.md`
- `git -C /Users/khs/codes/graduation_project show --stat --oneline HEAD`

### F2. 本轮修改目标
- 新增 `docs/experiments/cloud_execution_playbook.md`
- 更新 `README.md` 的训练入口说明
- 更新 `docs/experiments/README.md` 的双环境说明
