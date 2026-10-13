# 文档导航与实验阶段

更新：2026-10-13。目录编号表示研究推进顺序，日期表示阶段起始或主要工作区间，不改变文件正文中原有日期。所有运行命令仍在仓库根目录执行，不能在归档子目录中直接运行。

## 当前应该看什么

- 最新结果：[G4/G5综合分析](07_20260917_strength_control/UIEB_BLUE_STRENGTH_CONTROL_ANALYSIS.md)。

- 当前开发：[逐图独立物理再退化方案](09_20261013_physical_redegradation/UIEB_BLUE_PHYSICAL_REDEGRADATION_PLAN.md)。
- 最近证据：[整图诊断分析](06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_DIAGNOSTIC_ANALYSIS.md)与[路径对照分析](05_20260912_13_factorial/UIEB_BLUE_FACTORIAL_ANALYSIS.md)。
- 版本管理：[Git工作建议](00_project/GIT_WORKFLOW.md)。

历史文件的“待执行”“下一步”描述只代表当时，今天的执行状态以本导航和当前阶段方案为准。尤其不要重跑诊断v1、Fisher探针或已经暂停的第91步扩展。

## 分阶段索引

### 项目维护与跨阶段记录

状态：持续维护；旧日志正文保留当时状态。

- [GIT_WORKFLOW.md](00_project/GIT_WORKFLOW.md)
- [CLEANUP.md](00_project/CLEANUP.md)
- [UIEB_BLUE_EXPERIMENT_LOG.md](00_project/UIEB_BLUE_EXPERIMENT_LOG.md)

### 阶段1：迁移准备（09-06）

状态：历史对接规范，迁移已完成。

- [NU2Net_DCTTA_HANDOFF.md](01_20260906_migration/NU2Net_DCTTA_HANDOFF.md)
- [UIE_MIGRATION.md](01_20260906_migration/UIE_MIGRATION.md)

### 阶段2：源训练与首轮实验（09-07～09）

状态：已完成；发现明显负迁移。

- [UIEB_BLUE_EXPERIMENT_PLAN.md](02_20260907_09_baseline/UIEB_BLUE_EXPERIMENT_PLAN.md)
- [UIEB_BLUE_RUN.md](02_20260907_09_baseline/UIEB_BLUE_RUN.md)

### 阶段3：复盘、Fisher与严格诊断（09-10～11）

状态：已完成；v1被v2替代，Fisher探针无需重跑。

- [UIEB_BLUE_DIAGNOSTIC_RUN.md](03_20260910_11_strict_diagnostic/UIEB_BLUE_DIAGNOSTIC_RUN.md)
- [UIEB_BLUE_DIAGNOSTIC_V2_ANALYSIS.md](03_20260910_11_strict_diagnostic/UIEB_BLUE_DIAGNOSTIC_V2_ANALYSIS.md)
- [UIEB_BLUE_DIAGNOSTIC_V2_RUN.md](03_20260910_11_strict_diagnostic/UIEB_BLUE_DIAGNOSTIC_V2_RUN.md)
- [UIEB_BLUE_FISHER_MISMATCH_20260911.md](03_20260910_11_strict_diagnostic/UIEB_BLUE_FISHER_MISMATCH_20260911.md)
- [UIEB_BLUE_ROUND1_REVIEW_AND_DIAGNOSTIC_PLAN.md](03_20260910_11_strict_diagnostic/UIEB_BLUE_ROUND1_REVIEW_AND_DIAGNOSTIC_PLAN.md)

### 阶段4：真实单步更新诊断（09-12）

状态：已完成；旧分析中第91步扩展已暂停。

- [UIEB_BLUE_UPDATE_DIAGNOSTIC_ANALYSIS.md](04_20260912_update_diagnostic/UIEB_BLUE_UPDATE_DIAGNOSTIC_ANALYSIS.md)
- [UIEB_BLUE_UPDATE_DIAGNOSTIC_RUN.md](04_20260912_update_diagnostic/UIEB_BLUE_UPDATE_DIAGNOSTIC_RUN.md)

### 阶段5：G1/G2/G3路径对照（09-12～13）

状态：已完成；G0复用strict v2。

- [UIEB_BLUE_FACTORIAL_ANALYSIS.md](05_20260912_13_factorial/UIEB_BLUE_FACTORIAL_ANALYSIS.md)
- [UIEB_BLUE_FACTORIAL_EXECUTION_PLAN.md](05_20260912_13_factorial/UIEB_BLUE_FACTORIAL_EXECUTION_PLAN.md)
- [UIEB_BLUE_FACTORIAL_RUN.md](05_20260912_13_factorial/UIEB_BLUE_FACTORIAL_RUN.md)

### 阶段6：整图伪标签与颜色诊断（09-14～15）

状态：已完成；visual_v2含OOM修复，v1命令属于历史。

- [UIEB_BLUE_VISUAL_DIAGNOSTIC_ANALYSIS.md](06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_DIAGNOSTIC_ANALYSIS.md)
- [UIEB_BLUE_VISUAL_DIAGNOSTIC_PLAN.md](06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_DIAGNOSTIC_PLAN.md)
- [UIEB_BLUE_VISUAL_DIAGNOSTIC_RUN.md](06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_DIAGNOSTIC_RUN.md)
- [UIEB_BLUE_VISUAL_SELECTED.csv](06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_SELECTED.csv)
- [分析附件](06_20260914_15_visual_diagnostic/visual_analysis/overview.png)：该目录保存统计JSON/CSV、科学绘图和10张联系表。

### 阶段7：再退化使用强度对照（09-17起）

状态：G4/G5已完成并回传，综合分析已完成。

- [G4/G5综合分析](07_20260917_strength_control/UIEB_BLUE_STRENGTH_CONTROL_ANALYSIS.md)
- [G4/G5程序机运行说明](07_20260917_strength_control/UIEB_BLUE_STRENGTH_CONTROL_RUN.md)
- [UIEB_BLUE_STRENGTH_CONTROL_PLAN.md](07_20260917_strength_control/UIEB_BLUE_STRENGTH_CONTROL_PLAN.md)

### 阶段8：G0 学生原图输出补诊断（09-21起）

状态：visual_yin_v1已回传并完成核验；不新增适应实验组。

- [G0_YIN_PLAN.md](08_20260921_g0_yin/G0_YIN_PLAN.md)
- [统一中间过程图入口与运行说明](08_20260921_g0_yin/G0_UNIFIED_VISUAL_RUN.md)
- [G0_YIN_RUN.md](08_20260921_g0_yin/G0_YIN_RUN.md)（旧版独立入口）

### 阶段9：逐图独立物理再退化（10-13起）

状态：预实验已回传并核验，建议K200待确认；正式G6/G7尚未实现。

- [UIEB_BLUE_PHYSICAL_REDEGRADATION_PLAN.md](09_20261013_physical_redegradation/UIEB_BLUE_PHYSICAL_REDEGRADATION_PLAN.md)
- [物理再退化预实验运行说明](09_20261013_physical_redegradation/UIEB_BLUE_PHYSICAL_PILOT_RUN.md)
- [预实验回传分析](09_20261013_physical_redegradation/UIEB_BLUE_PHYSICAL_PILOT_ANALYSIS.md)

## 如何区分文件用途

- PLAN / EXECUTION_PLAN：当轮实验设计，不等于实现完成。
- RUN：对应阶段的运行说明；复用前先看阶段状态与输入身份。
- ANALYSIS：已回传结果的解释，应优先阅读更新阶段的证据。
- EXPERIMENT_LOG：跨阶段历史记录，不覆盖原叙述；最新状态看本导航。

本次归档仅移动文档与附件、修正链接及CSV示例路径，没有改变数据集、实验产物或算法代码。旧文件名保留，便于搜索。
