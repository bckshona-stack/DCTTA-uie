# G0 y_in 补导出：程序机运行说明

> 统一入口已更新：当前运行请看 [G0_UNIFIED_VISUAL_RUN.md](G0_UNIFIED_VISUAL_RUN.md)。本文保留013e188版独立补导出的方案/说明；不要混用版本续跑。

日期：2026-09-21。对应方案A，入口 export_g0_yin.py。现有训练/实验无需结束。

## 1. 同步文件

将交付包解压到 F:\qmh\DCTTA-uie，保持目录结构。包内包含新增入口、测试和本阶段文档，不替换旧算法实现。依赖程序机已有的 diagnose_visual_uie.py、diagnose_uie.py、uie_tta.py、uie_backbones 等原工程文件。

## 2. 在新的 PowerShell 窗口配置

```powershell
Set-Location F:\qmh\DCTTA-uie
conda activate resshift
nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.used --format=csv
```

根据之前核对的硬件映射，下列UUID是此前空闲的卡（nvidia-smi GPU0、任务管理器GPU1，PCI 17）。先看当前查询结果确认仍空闲，再使用。UUID指定物理卡后，Python内仍写cuda:0。

```powershell
$env:CUDA_VISIBLE_DEVICES = "GPU-53f5a782-8c8c-958b-006d-c8aad0f4d8a3"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$exp = ".\experiments\UIEBblue_weaktostrong_NU2Net"
$diag = "${exp}_diagnostic_v2"
$visual = "${exp}_visual_v2"
$out = "${exp}_visual_yin_v1"
```

环境变量只影响这个窗口后续启动的进程。不要在运行中的窗口中断旧程序，不要让两个进程同时写这个新$out目录。若该卡现在已被占用，先确定另一张空闲卡的UUID。

## 3. 预检后运行

```powershell
python export_g0_yin.py --phase preflight --experiment_dir $exp --diagnostic_dir $diag --visual_dir $visual --output_dir $out --device cuda:0
```

应显示134个checkpoint、110张图片、保守磁盘估计约0.316 GB。若提示环境、历史文件或身份不一致，回传报错，不修改原文件来绕过检查。

预检成功后执行：

```powershell
python export_g0_yin.py --phase run --experiment_dir $exp --diagnostic_dir $diag --visual_dir $visual --output_dir $out --device cuda:0
```

日志从 committed 0/133 到 committed 133/133；总计134时刻。结束应显示 COMPLETE: 110 images; 1340 metric rows。每时刻需10张图共90次整图推理，不是每打印一次就只推理一张图。

中断后保持同样的环境变量、文件和参数，重新运行同一条run指令即可续跑。已完成时只校验并跳过推理。不要为了续跑改文件或换输出参数；需要新配置时用新的输出目录。

## 4. 结果位置与回传

新目录为 experiments/UIEBblue_weaktostrong_NU2Net_visual_yin_v1/：

| 文件或目录 | 含义 |
|---|---|
| student_raw/000000/ 至 student_raw/000133/ 中的11阶段 | 原分辨率 y_in，共110张 |
| metrics.csv | 1340行；每步每图的学生、教师八视图、教师单视图PSNR |
| summary.csv | 134行；固定10张的逐步平均值 |
| selected_samples.csv | 原10张典型图名单 |
| manifest.json | 输入权重哈希、环境、选图、阶段、容量约定 |
| progress.json | 已提交进度及逐图指标 |
| status.json | 完成标记、输出数量与文件哈希 |

原教师伪标签图片仍在 ${visual}/teacher_pseudo/snapshots/<步数>/，按步数和文件名与新student_raw对齐；原y_sd仍在${visual}/y_sd/snapshots/<步数>/。不复制旧图片，避免冗余。新脚本不修改旧可视化图，回传后再结合新数据更新科研图和分析。

请回传整个新增$out目录，无需回传原134个权重。磁盘容量限制默认10 GB，本次估计约0.316 GB；这不是显存占用保证。

## 5. Git工程记录

本次新增代码、专项测试和本阶段文档应单独提交，不把experiments中的输出与权重加入Git。开发机沿用codex/redegradation-strength分支；程序机可通过包内PACKAGE_MANIFEST.json核对提交与文件SHA256。仅解压文件不等同于程序机已切到该Git提交。

结果回传后，将分析文档和精选小型图表另作结果提交；确认后再推送、审阅合并。本次不推送、不合并、不修改历史实验组命名。
