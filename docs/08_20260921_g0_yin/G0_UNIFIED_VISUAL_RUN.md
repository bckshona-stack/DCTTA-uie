# G0 统一中间过程图输出：visual_v3

日期：2026-09-21。统一入口：`diagnose_visual_uie.py`。当前用户已有G0结果，优先使用下文的历史补齐模式；完整模式用于需要重新输出全套中间过程图的场景。

## 观测语义

完整模式严格重放G0适配。到第0、14、27、40、54、67、80、94、107、120、133步时，暂停适配，在同一模型状态下逐一处理固定10张原图：

1. 当时教师对原图进行八视图推理，得到teacher_pseudo。
2. 当时生成器对原图再退化，得到x_sd。
3. 当时学生处理x_sd，得到y_sd。
4. 同一个学生处理原图，得到y_in。

观测期间模型处于eval和no_grad，整图原分辨率；不执行优化器更新，不更新教师EMA。观测后恢复完整适配状态（含模型、生成器、优化器、随机状态、模式、梯度等），再继续适配。每步真实适配的损失和完整状态哈希必须与原G0轨迹一致。

第k步表示已完成第k张适配图片后的状态。y_in是学生对原图的输出，不是y_sd的别名，也不是把教师伪标签送入学生。旧科研图红线仍表示y_sd。

## 两种模式

| 模式 | 入口phase | 数据来源 | 新增输出 |
|---|---|---|---|
| 已有实验补齐（当前使用） | yin-preflight、yin-run | 134个历史学生/教师小权重 | 11阶段110张y_in，1340行逐图指标；不运行适配/RDDM |
| 完整中间过程图 | select、preflight、run | 原G0完整状态与精确适配重放 | 四类阶段图和逐样本适配前后图，统一保存在新visual_v3目录 |

两种模式共用student_raw_prediction函数，均为当时学生对同一原图的前向。历史补齐由export_g0_yin.py提供内部实现，该文件需和统一入口一起同步；旧独立入口仍可调用。完整模式的阶段图片目录命名y_in，历史补齐及G4/G5目录student_raw也表示y_in。

## 当前程序机：补齐已有实验

将新版交付包解压到F:\qmh\DCTTA-uie，保持目录结构。在新的PowerShell窗口操作。不要在进程仍使用旧代码时替换其源码；已启动013e188版补导出的任务应使用原版本续跑，或在其完成后再更新。

```powershell
Set-Location F:\qmh\DCTTA-uie
conda activate resshift
nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.used --format=csv
```

下列UUID对应此前空闲的PCI 17显卡；确认当前仍空闲后再设置。Python内cuda:0表示被暴露的这张卡，不是任务管理器GPU0。

```powershell
$env:CUDA_VISIBLE_DEVICES = "GPU-53f5a782-8c8c-958b-006d-c8aad0f4d8a3"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$exp = ".\experiments\UIEBblue_weaktostrong_NU2Net"
$diag = "${exp}_diagnostic_v2"
$visual = "${exp}_visual_v2"
$out = "${exp}_visual_yin_v2"
python diagnose_visual_uie.py --phase yin-preflight --experiment_dir $exp --diagnostic_dir $diag --visual_dir $visual --output_dir $out --device cuda:0
```

预检通过后运行：

```powershell
python diagnose_visual_uie.py --phase yin-run --experiment_dir $exp --diagnostic_dir $diag --visual_dir $visual --output_dir $out --device cuda:0
```

此处yin_v2只是补导出代码版本对应的新输出目录，不是新增实验组。新增110张图、1340行指标、134行摘要，保守磁盘估计约0.316 GB。指标列和检查规则见[G0_YIN_PLAN.md](G0_YIN_PLAN.md)。完成后回传整个$out目录，旧teacher_pseudo/x_sd/y_sd继续使用visual_v2已有结果。

## 完整模式：四类图一次输出

当前补齐任务不需要执行这一节。需要完整重新输出时，使用独立的新visual_v3目录，仍复用旧10张选图：

```powershell
$fullOut = "${exp}_visual_v3"
$selection = Join-Path $visual "selected_samples.csv"
python diagnose_visual_uie.py --phase preflight --experiment_dir $exp --diagnostic_dir $diag --output_dir $fullOut --selection_csv $selection --device cuda:0
```

完整重放预检沿用原严格环境核对，包含可见GPU列表；若原实验暴露两张卡而此时UUID只暴露一张，预检将拒绝。只有要运行完整模式时，应恢复原GPU可见设置，再根据当时映射指定device；不要为通过预检随意改协议。当前历史补齐模式允许可见卡数量改变，不受此限制。

预检通过后将phase改成run，其他参数保持一致。此模式要运行RDDM和精确适配重放，计算成本远高于历史补齐。

四个并列目录：

```text
visual_v3/
  teacher_pseudo/snapshots/<step>/<filename>
  x_sd/snapshots/<step>/<filename>
  y_sd/snapshots/<step>/<filename>
  y_in/snapshots/<step>/<filename>
```

每个目录也含online/<step>/before、after记录；x_sd只保存before一份，after复用同一张x_sd。每个在线样本保存教师前后2张、学生y_sd前后2张、学生y_in前后2张、x_sd一张，共7张。

- 固定10图×11阶段×4类=440张。
- 全部133个在线样本×7张=931张。
- 合计1371张，比旧995张增加376张（阶段110张、在线266张）。
- metrics.csv共1128行（阶段330行、在线798行）；snapshot_summary.csv新增y_in汇总。这里是三类有参考指标的输出，x_sd不作为增强结果计PSNR。
- 完整模式的metrics.csv与补齐模式1340行指标覆盖范围不同：前者是阶段固定10图加在线当前样本；后者是全部134时刻固定10图。不能混作同一个统计总体。

完整模式使用manifest版本3，预检在任何写入前拒绝旧代码/旧版本目录，不自动升级visual_v2，也不修改旧图、指标或协议。旧G4/G5任务的代码哈希同样应保持原版本；本版不用于续跑那些已完成的历史任务。

## 验证与Git

9项本地专项测试通过，包括真实RDDM整图观测前后完整状态不变、实际适配损失及后续参数更新一致、y_in与学生原图前向一致、四类输出/前后共7张、旧目录零改写、统一补齐入口路由、补齐续跑与连续运行一致。

本地测试为CPU小图，程序机正式CUDA整图峰值与耗时仍待运行确认。容量参数max_gb是磁盘上限，不是显存限制。

本次统一入口与测试、说明单独提交；旧分析文件不混入代码提交。程序机用包内PACKAGE_MANIFEST.json核对文件身份。尚未推送或合并。结果回传后再单独提交分析报告与精选图表，不提交完整实验输出及权重。
