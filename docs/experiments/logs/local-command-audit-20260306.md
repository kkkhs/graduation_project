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

## G. AutoDL 无卡主机环境准备与续训资产迁移（2026-03-07）

> 说明：本段记录 AutoDL 无卡实例上的“云端续训前置准备”。密码认证已使用，但不在日志中记录明文。

### G1. 远端基础核查
- `expect <<'EOF' ... ssh -p 49353 root@connect.westc.gpuhub.com "hostname && pwd && whoami && uname -a && python3 --version && /root/miniconda3/bin/python --version && /root/miniconda3/bin/conda --version && nvidia-smi || true && df -h / /root /root/autodl-tmp" ... EOF`
  - 结果：确认主机为 AutoDL Linux 容器，`/root/autodl-tmp` 为 50G 数据盘，当前实例无 GPU。
- `expect <<'EOF' ... ssh -p 49353 root@connect.westc.gpuhub.com "git ls-remote https://github.com/WindVChen/DRENet.git HEAD && git ls-remote https://github.com/kkkhs/graduation_project.git HEAD" ... EOF`
  - 结果：确认云端可直接访问 GitHub，DRENet 与本仓库均可直接克隆。

### G2. 本地与远端传输策略调整
- `scp -P 49353 /Users/khs/Downloads/train.zip /Users/khs/Downloads/val.zip /Users/khs/Downloads/test.zip root@connect.westc.gpuhub.com:/root/autodl-tmp/transfer/`
  - 结果：3 个分包数据集压缩包传输完成，替代了先前低效的“解压后小文件目录传输”。
- `mkdir -p /Users/khs/codes/graduation_project/tmp_remote_prep && tar -C /Users/khs/codes/graduation_project/experiment_assets -czf /Users/khs/codes/graduation_project/tmp_remote_prep/drenet_resume_bundle_20260307.tar.gz runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights/last.pt runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights/best.pt runs/drenet_levirship_512_bs4_sna_20260307_formal01/results.txt runs/drenet_levirship_512_bs4_sna_20260307_formal01/opt.yaml runs/drenet_levirship_512_bs4_sna_20260307_formal01/hyp.yaml checkpoints/drenet`
  - 结果：生成 `drenet_resume_bundle_20260307.tar.gz`，约 `207M`。
- `scp -P 49353 /Users/khs/codes/graduation_project/tmp_remote_prep/drenet_resume_bundle_20260307.tar.gz root@connect.westc.gpuhub.com:/root/autodl-tmp/transfer/`
  - 结果：续训资产压缩包上传完成。

### G3. 远端目录、数据与代码落位
- `expect <<'EOF' ... ssh -p 49353 root@connect.westc.gpuhub.com "mkdir -p /root/autodl-tmp/transfer /root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship /root/autodl-tmp/experiment_assets/checkpoints/drenet /root/autodl-tmp/experiment_assets/runs /root/autodl-tmp/workspace/experiments/drenet && cd /root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship && unzip -q -o /root/autodl-tmp/transfer/train.zip && unzip -q -o /root/autodl-tmp/transfer/val.zip && unzip -q -o /root/autodl-tmp/transfer/test.zip && find /root/autodl-tmp/experiment_assets/datasets/LEVIR-Ship -maxdepth 2 -type d | sort" ... EOF`
  - 结果：云端数据集目录已展开，包含 `train/val/test` 及各自的 `images/labels/degrade`。
- `expect <<'EOF' ... ssh -p 49353 root@connect.westc.gpuhub.com "set -e; cd /root/autodl-tmp/workspace; test -d graduation_project || git clone https://github.com/kkkhs/graduation_project.git; cd graduation_project; git fetch origin; git checkout f34758aef1814fbd8ff713606a4d955d2020b04b; cd /root/autodl-tmp/workspace/experiments/drenet; test -d DRENet || git clone https://github.com/WindVChen/DRENet.git; cd DRENet; git checkout a187dbe0f623b521a62c6176c7cafaa7322f5f66" ... EOF`
  - 结果：本仓库与 DRENet 均已在云端固定到目标 commit。
- `expect <<'EOF' ... ssh -p 49353 root@connect.westc.gpuhub.com "cd /root/autodl-tmp/workspace && git apply /root/autodl-tmp/transfer/drenet_local_compat_20260307.patch && cd /root/autodl-tmp/workspace/experiments/drenet/DRENet && git status --short" ... EOF`
  - 结果：兼容补丁应用成功，`utils/datasets.py`、`utils/general.py`、`utils/loss.py` 进入修改状态。

### G4. 续训资产展开与清理
- `expect <<'EOF' ... ssh -p 49353 root@connect.westc.gpuhub.com "cd /root/autodl-tmp/experiment_assets && tar -xzf /root/autodl-tmp/transfer/drenet_resume_bundle_20260307.tar.gz && find /root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01 -maxdepth 2 \\( -name last.pt -o -name best.pt -o -name results.txt -o -name opt.yaml -o -name hyp.yaml \\) | sort && find /root/autodl-tmp/experiment_assets/checkpoints/drenet -maxdepth 1 -type f | sort && du -sh /root/autodl-tmp/experiment_assets /root/autodl-tmp/workspace /root/autodl-tmp/envs/shipdet" ... EOF`
  - 结果：`last.pt/best.pt/results.txt/opt.yaml/hyp.yaml` 与 4 个 `ep100` checkpoint 已展开到远端。
- `expect <<'EOF' ... ssh -p 49353 root@connect.westc.gpuhub.com "find /root/autodl-tmp/experiment_assets -name '._*' -type f -delete && find /root/autodl-tmp/experiment_assets/checkpoints/drenet -maxdepth 1 -type f | sort" ... EOF`
  - 结果：已清除 tar 带来的 `._*` macOS 元数据文件。

### G5. 远端环境准备
- `expect <<'EOF' ... ssh -p 49353 root@connect.westc.gpuhub.com "/root/miniconda3/bin/conda create -y -p /root/autodl-tmp/envs/shipdet python=3.10 >/tmp/conda_create_shipdet.log 2>&1 || (cat /tmp/conda_create_shipdet.log && exit 1); tail -n 20 /tmp/conda_create_shipdet.log" ... EOF`
  - 结果：成功创建 `/root/autodl-tmp/envs/shipdet`。
- 交互式 SSH 会话：在云端写入
  - `/root/autodl-tmp/workspace/experiments/drenet/DRENet/data/levir_ship_autodl.yaml`
  - `/root/autodl-tmp/workspace/experiments/drenet/DRENet/scripts/resume_formal01_autodl.sh`
  - 结果：已固化 Linux 路径与 `python train.py --resume "$LAST_PT"` 的续训入口。
- 交互式 SSH 会话：
  - `/root/autodl-tmp/envs/shipdet/bin/pip install -U pip setuptools wheel`
  - `/root/autodl-tmp/envs/shipdet/bin/pip install Cython matplotlib numpy opencv-python-headless Pillow PyYAML scipy tensorboard tqdm wandb seaborn pandas thop pycocotools`
  - 结果：依赖安装已启动；其中 `thop` 会级联安装 `torch` 及相关 CUDA 包，安装过程仍在进行中。

### G6. 已确认的关键事实
- `opt.yaml` 记录的原始正式实验名为 `drenet_levirship_512_bs4_sna_20260307_formal01`。
- `opt.yaml` 中原始 `project/save_dir` 仍是 Windows 路径；云端续训将优先依赖 `--resume <last.pt>` 读取 checkpoint 元数据，必要时使用已写好的 Linux fallback 命令。
- 当前云端无 GPU，因此本轮仅完成环境与资产落位，不启动训练。
- 依赖安装完成后，远端环境已确认：
 
## I. 3080 Ti 云主机正式续训与自动回传（2026-03-08）

### I1. 自动回传脚本改造与验证
- `sed -n '1,220p' /Users/khs/codes/graduation_project/scripts/sync_autodl_experiment_assets.sh`
  - 结果：确认原脚本依赖远端 `rsync`，不适配当前密码登录 AutoDL 主机。
- `bash -n /Users/khs/codes/graduation_project/scripts/sync_autodl_experiment_assets.sh`
  - 结果：脚本改造后语法检查通过。
- `SYNC_SSH_PASSWORD='***' bash /Users/khs/codes/graduation_project/scripts/sync_autodl_experiment_assets.sh root@connect.westb.seetacloud.com --port 29137 --run-name drenet_levirship_512_bs4_sna_20260307_formal01 --no-sync-checkpoints --no-sync-scripts --local-assets-root /Users/khs/codes/graduation_project/experiment_assets`
  - 结果：本机 `experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/results.txt` 已从 `100/299` 更新到 `101/101`；`weights/last.pt`、`weights/best.pt` 和 trace 日志同步成功。

### I2. 完整 checkpoint 与 strip 后权重核验
- `ssh ... "cd /root/autodl-tmp/workspace/experiments/drenet/DRENet && python - <<'PY' ... torch.load('/root/autodl-tmp/experiment_assets/runs/.../weights/last.pt', weights_only=False) ... PY"`
  - 结果：确认当前 run 目录内 `last.pt` 已不再携带 `epoch/wandb_id/opt`。
- `ssh ... "cd /root/autodl-tmp/workspace/experiments/drenet/DRENet && python - <<'PY' ... torch.load('/root/autodl-tmp/experiment_assets/checkpoints/drenet/*ep100*.pt', weights_only=False) ... PY"`
  - 结果：确认完整可续训 checkpoint 为：
    - `drenet_levirship_512_bs4_sna_20260307_formal01_last_ep100_20260307_165755.pt`
    - 元数据：`epoch=100`, `wandb_id=94d4wdmk`

### I3. 提速探测
- `ssh ... "cd /root/autodl-tmp/workspace/experiments/drenet/DRENet && export WANDB_MODE=disabled WANDB_DISABLED=true && /root/autodl-tmp/envs/shipdet/bin/python train.py --weights ...last_ep100_20260307_165755.pt --data .../levir_ship_autodl.yaml --epochs 102 --batch-size 6 --img-size 512 512 --workers 4 --device 0 --project /root/autodl-tmp/experiment_assets/runs --name drenet_probe_bs6_w4_20260308 --exist-ok"`
  - 结果：探测成功完成，无 OOM、无 dataloader 崩溃，峰值显存约 `3.29G`，验证指标约 `P=0.423, R=0.773, mAP@0.5=0.686, mAP@0.5:0.95=0.229`。

### I4. 正式续训到 300 epoch
- 本机 watch：
  - `SYNC_SSH_PASSWORD='***' bash /Users/khs/codes/graduation_project/scripts/sync_autodl_experiment_assets.sh root@connect.westb.seetacloud.com --port 29137 --run-name drenet_levirship_512_bs4_sna_20260307_formal01 --no-sync-checkpoints --no-sync-scripts --watch-pattern 'train.py.*drenet_levirship_512_bs4_sna_20260307_formal01.*--epochs 300' --interval 120 --local-assets-root /Users/khs/codes/graduation_project/experiment_assets`
  - 结果：watch 已启动，并报告 `Remote training still active`。
- 远端后台启动：
  - `ssh ... "cd /root/autodl-tmp/workspace/experiments/drenet/DRENet && mkdir -p /root/autodl-tmp/experiment_assets/runs/trace && export WANDB_RESUME=allow WANDB_RUN_ID=94d4wdmk && nohup /root/autodl-tmp/envs/shipdet/bin/python train.py --weights /root/autodl-tmp/experiment_assets/checkpoints/drenet/drenet_levirship_512_bs4_sna_20260307_formal01_last_ep100_20260307_165755.pt --data /root/autodl-tmp/workspace/experiments/drenet/DRENet/data/levir_ship_autodl.yaml --epochs 300 --batch-size 8 --img-size 512 512 --workers 4 --device 0 --project /root/autodl-tmp/experiment_assets/runs --name drenet_levirship_512_bs4_sna_20260307_formal01 --exist-ok > /root/autodl-tmp/experiment_assets/runs/trace/train_drenet_formal_resume_300_bs8_w4_20260308_101502.log 2>&1 < /dev/null & echo PID=\$!; echo LOG=..."`
  - 结果：远端 Python 训练进程 PID `3046` 已启动。
- 远端运行核验：
  - `ssh ... "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits && tail -n 60 /root/autodl-tmp/experiment_assets/runs/trace/train_drenet_formal_resume_300_bs8_w4_20260308_101502.log"`
  - 结果：
    - GPU 计算进程 PID `3046`
    - 显存占用约 `4.39G`
    - W&B 恢复到 run `94d4wdmk`
    - 日志已进入 `101/299` 主训练循环

### I5. 速度优先参数上调与收尾托管
- `ssh ... \"pkill -f 'train.py --weights .*drenet_levirship_512_bs4_sna_20260307_formal01.*--epochs 300' || true\"`
  - 结果：停止先前 `batch=8, workers=4` 续训。
- `ssh ... \"cd /root/autodl-tmp/workspace/experiments/drenet/DRENet && export WANDB_RESUME=allow WANDB_RUN_ID=94d4wdmk && nohup /root/autodl-tmp/envs/shipdet/bin/python train.py --weights /root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights/last.pt --data /root/autodl-tmp/workspace/experiments/drenet/DRENet/data/levir_ship_autodl.yaml --epochs 300 --batch-size 12 --img-size 512 512 --workers 8 --device 0 --project /root/autodl-tmp/experiment_assets/runs --name drenet_levirship_512_bs4_sna_20260307_formal01 --exist-ok > /root/autodl-tmp/experiment_assets/runs/trace/train_drenet_formal_resume_300_bs12_w8_20260308_101948.log 2>&1 < /dev/null &\"`
  - 结果：`batch=12, workers=8` 正式续训启动成功；W&B 继续接入 `94d4wdmk`。
- `tail -n ... results.txt`（本地/远端）
  - 结果：已推进到 `109/299`（本地）与 `111/299`（远端，持续推进中）。
- 新增脚本：
  - `scripts/watch_sync_then_shutdown_autodl.sh`
  - 作用：训练结束后执行 final sync，校验本地/远端最后一行一致，然后发送远端关机命令。
- 托管执行：
  - `SYNC_SSH_PASSWORD='***' /Users/khs/codes/graduation_project/scripts/watch_sync_then_shutdown_autodl.sh root@connect.westb.seetacloud.com --port 29137 --run-name drenet_levirship_512_bs4_sna_20260307_formal01 --watch-pattern 'train.py.*drenet_levirship_512_bs4_sna_20260307_formal01.*--epochs 300' --interval 120 --local-assets-root /Users/khs/codes/graduation_project/experiment_assets`
  - 结果：收尾脚本已进入 watch 状态，自动同步与自动关机流程生效。

### I6. 最终全量回传与关机收尾
- 手动最终同步（不依赖远端 rsync）：
  - `SYNC_SSH_PASSWORD='***' bash scripts/sync_autodl_experiment_assets.sh root@connect.westb.seetacloud.com --port 29137 --run-name drenet_levirship_512_bs4_sna_20260307_formal01 --local-assets-root /Users/khs/codes/graduation_project/experiment_assets`
  - 结果：`runs` 与 `checkpoints` 回传成功；同步 `scripts` 时远端目录不存在导致任务中断。
- 重试同步（跳过 scripts）：
  - `SYNC_SSH_PASSWORD='***' bash scripts/sync_autodl_experiment_assets.sh root@connect.westb.seetacloud.com --port 29137 --run-name drenet_levirship_512_bs4_sna_20260307_formal01 --local-assets-root /Users/khs/codes/graduation_project/experiment_assets --no-sync-scripts --sync-datasets --sync-wandb`
  - 结果：`runs` 与 `checkpoints` 成功；用户要求“数据集不需要传”后停止 dataset 同步。
- 本地核对：
  - `tail -n 3 /Users/khs/codes/graduation_project/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/results.txt`
  - 结果：本地已到 `299/299`。
- 远端核对：
  - `ssh ... \"tail -n 1 /root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/results.txt && ls -lh /root/autodl-tmp/experiment_assets/runs/drenet_levirship_512_bs4_sna_20260307_formal01/weights\"`
  - 结果：远端同样为 `299/299`，`best.pt/last.pt` 大小均为 `12M`。
- 关机执行：
  - `ssh ... \"shutdown -h now\"`
  - 结果：SSH 连接被远端主动关闭，云主机关机完成。
  - `torch==2.10.0+cu128`
  - `wandb==0.25.0`
  - 额外补装：`torchvision`
- 通过在 DRENet 仓库目录下读取 checkpoint，确认：
  - `epoch=100`
  - `wandb_id=94d4wdmk`
- 远端 wandb 登录态核查结果：
  - 初始状态：`/root/.netrc` 不存在
  - `/root/.config/wandb` / `/root/.wandb` 不存在
  - 已尝试使用用户提供的 key 执行 `wandb login --relogin`
  - 第一次 CLI 返回：`API key must have 40+ characters`
  - 用户补充新 key 后再次执行 `wandb login --relogin`
  - 第二次成功创建 `/root/.netrc`
  - 结论：当前云端已配置有效的 wandb key。

## H. AutoDL 结果同构同步能力补充（2026-03-07）

### H1. 本轮修改目标
- 新增 `scripts/sync_autodl_experiment_assets.sh`
- 更新 `docs/system/result_sync_flow.md`
- 更新 `README.md`

### H2. 关键能力
- 远端 `/root/autodl-tmp/experiment_assets/runs/<run_name>/`
  与本地 `/Users/khs/codes/graduation_project/experiment_assets/runs/<run_name>/`
  保持同名、同层级映射。
- 支持只同步某个 `run_name` 对应的 `runs/` 与 `checkpoints/`。
- 支持 watch 模式轮询远端训练进程，进程结束后自动执行最终一次同步。

### H3. 校验命令
- `bash -n /Users/khs/codes/graduation_project/scripts/sync_autodl_experiment_assets.sh`
  - 结果：通过，无语法错误。

## I. 3080 Ti 云主机续训冒烟（2026-03-08）

### I1. 基础核查
- `expect <<'EOF' ... ssh -p 29137 root@connect.westb.seetacloud.com "hostname && pwd && whoami && uname -a && nvidia-smi && df -h / /root /workspace /root/autodl-tmp 2>/dev/null || true && python3 --version 2>/dev/null || true" ... EOF`
  - 结果：确认主机为 `autodl-container-a2fd40a1b8-db880c0e`，GPU 为 `RTX 3080 Ti 12GB`，数据盘 `/root/autodl-tmp` 可用约 `40G`。
- 交互式 SSH 会话：
  - `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader`
  - `ls -l /root/.netrc`
  - `ls -l .../levir_ship_autodl.yaml .../resume_formal01_autodl.sh .../weights/last.pt`
  - `python ... torch.load(last.pt, weights_only=False)`
  - 结果：确认数据集、续训脚本、`last.pt`、wandb 登录态都已存在，checkpoint 元数据为 `epoch=100`, `wandb_id=94d4wdmk`。

### I2. 续训前环境修复
- 交互式 SSH 会话：
  - `/root/autodl-tmp/envs/shipdet/bin/python -c "import pkg_resources"`
  - 结果：失败，`ModuleNotFoundError: No module named 'pkg_resources'`
- 交互式 SSH 会话：
  - `/root/autodl-tmp/envs/shipdet/bin/pip install 'setuptools<81'`
  - 结果：安装 `setuptools-80.10.2` 后修复。
- 首次冒烟命令：
  - `python train.py --weights .../last.pt --data .../levir_ship_autodl.yaml --epochs 102 --batch-size 4 --img-size 512 512 --workers 2 --device 0 --project /root/autodl-tmp/experiment_assets/runs --name drenet_levirship_512_bs4_sna_20260307_formal01 --exist-ok`
  - 结果：失败，`DistributionNotFound: The 'opencv-python>=4.1.2' distribution was not found`
- 交互式 SSH 会话：
  - `/root/autodl-tmp/envs/shipdet/bin/pip install opencv-python>=4.1.2`
  - 结果：requirements 检查通过。
- 第二次冒烟命令：
  - 同上
  - 结果：失败，`_pickle.UnpicklingError`，原因是 `train.py` 中 `torch.load(weights, map_location=device)` 未显式关闭 `weights_only=True`。
- 交互式 SSH 会话：
  - `perl -0pi -e 's/ckpt = torch\\.load\\(weights, map_location=device\\)  # load checkpoint/ckpt = torch.load(weights, map_location=device, weights_only=False)  # load checkpoint/' train.py`
  - 结果：修复 `train.py` checkpoint 载入兼容性。

### I3. 冒烟成功结果
- 第三次冒烟命令：
  - `python train.py --weights .../last.pt --data .../levir_ship_autodl.yaml --epochs 102 --batch-size 4 --img-size 512 512 --workers 2 --device 0 --project /root/autodl-tmp/experiment_assets/runs --name drenet_levirship_512_bs4_sna_20260307_formal01 --exist-ok`
  - 结果：成功。
- 关键输出：
  - `CUDA:0 (NVIDIA GeForce RTX 3080 Ti, 11910.625MB)`
  - wandb 成功恢复原 run：`94d4wdmk`
  - 训练从 `101/101` 开始，不是从 `0` 重启
  - `1 epochs completed`
  - 验证摘要：`P=0.379`, `R=0.776`, `mAP@0.5=0.681`, `mAP@0.5:0.95=0.242`
- 产物：
  - 远端日志：`/root/autodl-tmp/experiment_assets/runs/trace/train_drenet_resume_smoke_102_20260308_095552.log`
  - 远端权重：`weights/last.pt`, `weights/best.pt` 已更新，时间戳为 `2026-03-08 09:57`

### I4. 自动同步脚本验证
- `bash scripts/sync_autodl_experiment_assets.sh root@connect.westb.seetacloud.com --port 29137 --run-name drenet_levirship_512_bs4_sna_20260307_formal01 --sync-wandb --remote-wandb-root /root/autodl-tmp/workspace/experiments/drenet/DRENet/wandb`
  - 结果：失败。
  - 原因：
    - 当前远端没有安装 `rsync`
    - 本机调用脚本时也未走密码注入逻辑
  - 结论：`experiment_assets` 自动回传链路还需继续补强，当前不能宣称已自动同步完成。
