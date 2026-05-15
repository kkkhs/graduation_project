# thesis_overleaf2 — 合肥工业大学本科毕业论文（新模板）

> 论文题目：遥感场景下的微小船舶检测系统设计与实现
> 作者：邝黄硕
> 学号：2022217586
> 模板：2025 年官方发布版（`hfut.cls`）

---

## 目录结构

```
thesis_overleaf2/
├── Thesis.tex              # 主入口文件
├── hfut.cls                # 官方模板类（勿修改）
├── ref.bib                 # 参考文献
├── build.sh                # 本地编译脚本
├── build.bat               # Windows 编译脚本
├── fonts/                  # Adobe 字体文件
├── img/                    # 论文图片
│   ├── hfut_logo.png       # 校徽
│   ├── hfut_name.png       # 校名
│   └── ch*.png             # 论文插图（14张）
├── tex/                    # 论文正文
│   ├── info.tex            # 个人信息（题目、姓名、学号、导师等）
│   ├── abstract.tex        # 中英文摘要
│   ├── introduction.tex    # 第1章 绪论
│   ├── related_work.tex    # 第2章 国内外研究现状
│   ├── method.tex          # 第3章 数据集、评价指标与方法方案
│   ├── experiment.tex      # 第4章 实验结果与分析
│   ├── system.tex          # 第5章 系统需求分析、设计与实现
│   ├── conclusion.tex      # 第6章 总结与展望
│   ├── acknowledge.tex     # 致谢
│   └── pages/              # 封面、声明页（勿修改）
└── output2/                # 编译输出目录（所有产物放这里）
    └── Thesis.pdf          # 最终 PDF
```

---

## 编译方式

### 方式一：本地编译（macOS/Linux）

```bash
cd thesis_overleaf2
bash build.sh
```

编译产物输出到 `output2/` 目录。

### 方式二：手动分步编译

```bash
cd thesis_overleaf2
xelatex Thesis.tex
biber Thesis
xelatex Thesis.tex
xelatex Thesis.tex
# 将产物移到 output2/
mv Thesis.pdf Thesis.* tex/pages/*.aux tex/*.aux output2/
```

### 方式三：Overleaf 在线编译

1. 将 `thesis_overleaf2/` 整个目录上传到 Overleaf
2. 编译器选择 **XeLaTeX**
3. 自动完成编译（Overleaf 自带 biber）

---

## 编译规则

1. **所有编译产物输出到 `output2/` 目录**，源目录保持干净
2. 编译命令：`xelatex → biber → xelatex → xelatex`
3. 参考文献后端：`biber`（`hfut.cls` 中已配置 `backend=biber`）
4. 主入口：`Thesis.tex`
5. 图片路径：`img/`

---

## 注意事项

- `hfut.cls` 是官方模板类文件，**不要修改**
- `tex/pages/` 下的封面、声明页文件，**不要修改**
- 个人信息在 `tex/info.tex` 中修改
- 参考文献在 `ref.bib` 中管理
- 编译前先清理旧产物：`rm -rf output2 && mkdir -p output2`
