# HKU TPG DS — 我的课程产出

香港大学授课型研究生（数据科学方向）课程中 **我自己做出来的东西**。

课程材料（课件、教程、作业题目）**不放在这里**。那些由本机的 `moodle-sync` 同步维护，
覆盖更全、且逐字节忠于原始文件。这个仓库只放我写的东西。

## 内容

### COMP7404 — NeurIPS Research Showcase

从 NeurIPS 2025 Oral / Spotlight 论文清单出发，抓取摘要、按摘要重判分类标签，
再生成一个可检索的展示页。

- `abstracts/` — 抓取与重打标签的脚本
- `build/` — 分类清单 CSV、构建脚本与模板
- `neurips2025-oral-spotlight/index.html` — 生成的检索页

### DASC7011 — Assignment 1

`Assignment1.qmd` 及渲染出的 pdf / html，含 R 代码与作答。

（DASC7606、STAT8017 目前没有产出，等有作业再往里放。）

## 部署

`.github/workflows/pages.yml` 在 `COMP7404/NeurIPS Showcase/**` 有变动时发布 GitHub Pages。

## 说明

本仓库仅存放个人作业与项目产出。
