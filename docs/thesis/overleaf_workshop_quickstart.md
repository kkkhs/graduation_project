# Overleaf 新手快速上手（VSCode + Overleaf Workshop）

## 1. 你只需要做这 5 步
1. 注册并登录 Overleaf（网页先能打开你的论文项目）。
2. 在 VSCode 安装扩展：`Overleaf Workshop`。
3. 在 Overleaf 网页登录状态下，按扩展文档导出浏览器 Cookie（`cookies.txt`）并在扩展设置中填写路径。
4. 在 VSCode 命令面板执行扩展的登录/同步命令，把项目下载到本地目录（建议 `thesis_overleaf/`）。
5. 打开本仓库的初稿目录，把第1-3章内容粘贴进模板对应章节文件并编译。

## 2. 你现在要复制的文件
- 第1章：`docs/thesis/drafts_v1/chapter1_background.tex`
- 第2章：`docs/thesis/drafts_v1/chapter2_related_work.tex`
- 第3章：`docs/thesis/drafts_v1/chapter3_dataset_metric_method.tex`

## 3. 常见问题
- 不能同步：优先检查 `cookies.txt` 路径和是否过期，重新导出一次。
- 同步后看不到章节：先找 `main.tex` 里 `\input` / `\include` 指向的章节路径。
- 引用报错：首轮先保留占位 key，第二轮统一补齐 `.bib`。

## 4. 你完成后告诉我这两件事
1. 本地模板目录路径（例如：`/Users/khs/codes/graduation_project/thesis_overleaf`）
2. `main.tex` 里第1-3章对应的章节文件路径

拿到这两项后，我会直接把当前 V1 初稿精确落到模板文件里。
