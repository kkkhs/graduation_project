# 论文最终提交检查清单

## 一次性检查顺序

1. 重新编译 PDF

```bash
cd thesis_overleaf
latexmk -g -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

2. 打开 `build/main.pdf`，只看以下页面：
- 封面页
- 签名页 / 版权页
- 中文摘要与英文摘要
- 目录、图目录、表目录
- 参考文献

3. 重点确认这 6 件事：
- `hfutsetup.tex` 中题目、学号、姓名、专业、导师、日期、入学年级、系名称都正确
- 非本科字段占位内容没有出现在最终 PDF 页面
- 图表没有明显分页异常或跨页断裂
- DRENet / FCOS / YOLO26 名称前后一致
- FCOS 仍只作为参考基线，不与系统/消融口径混写
- 参考文献样式统一、无缺失引用

## 当前已确认通过

- 论文可编译，`build/main.pdf` 可生成
- 正文 `\cite{}` 均能在 `hfut.bib` 中找到
- `\includegraphics{}` 资源未发现缺失
- 模板腔表述已完成统一收口

## 还需要人工确认

- 封面页和模板页的最终视觉效果
- 系名称、专业名称、导师姓名是否与学校最终提交要求完全一致
- 图表分页在最终 PDF 中是否足够自然
