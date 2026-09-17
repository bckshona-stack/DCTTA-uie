# UIEB 第二版诊断：固定第一轮 Fisher 与数值模式对照

> 归档说明（2026-09-17）：本文件保留所属阶段的历史内容和当时状态；当前进展请看 [文档导航](../README.md)。文中“待运行”等描述不代表今天的状态。

日期：2026-09-11。用户已确认实施。本地代码已实现；正式 CUDA 验收及完整诊断须在程序机运行。

## 1. 此次调整及结论边界

程序机同配置两次重算 Fisher 有102个掩码位置不同，严格模式单图反向检查成功。继续要求重算结果按位等于第一轮，无法可靠进入诊断。现在将第一轮真实 Fisher 作为固定输入，建立严格数值模式下的新控制运行。

- 校验第一轮 Fisher、源最佳权重、原始 run.json 的精确 SHA-256；校验数据、裁剪、模型配置、候选参数和 Fisher 比例。历史源码身份通过文件内显式兼容记录核查，保留原始身份和掩码，不给任意旧缓存放行。
- 在新诊断进程中启用全局严格确定性，配置 CUBLAS 工作区；初始化后和逐次适配后检查数值模式没有改变。对不支持的操作直接报错，禁止自动改为 warn_only 或放宽轨迹比较。
- 不修改 NU2Net、原 DCTTA 数学目标、插值参数、学习率、Fisher 恢复、学生/教师更新及 RDDM 更新规则。严格模式可能改变 PyTorch 实际内核、浮点舍入和耗时，**不是第一轮的位级复现**。
- 增加相同初态、相同旧 Fisher 的默认数值模式前缀对照；统一离线评价输出和曲线。reference 不参与适配、选权重、选模式或停止条件。
- 先完成严格验收及前缀数值对照，再进入133张诊断。22张对照不能证明两种数值模式在133张全过程一致；若需将新诊断结论归因到第一轮，可能还需完整默认模式对照或局部干预实验。

## 2. 同步与准备

这次将本地 `movetoUIE/diagnose_uie.py`、`movetoUIE/options.py` 同步至程序机 `F:\qmh\DCTTA-uie`。上一版已同步的 `uie_tta.py` 和 `RDDM/src/residual_denoising_diffusion_pytorch.py` 仍需保留；无需再次修改网络或重训源模型。测试文件不用于正式实验。

原始第一轮目录和 `_diagnostic_v1` 保留，用新的 `_diagnostic_v2` 输出目录。不要编辑第一轮 run.json 或重新序列化原 Fisher；即使内容看起来相同，文件哈希变化也会被拒绝。

独立 PowerShell 窗口：

```powershell
conda activate resshift
cd F:\qmh\DCTTA-uie
$env:CUDA_VISIBLE_DEVICES="1"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
$exp = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net'
$diag = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_diagnostic_v2'
```

物理 GPU 1 映射为该进程的 cuda:0。当前诊断默认协议为 `fixed_fisher_strict`，以下命令已明确写出。环境、代码及配置在整个诊断期间保持一致；发生变化会使旧验收失效。不要切换为 legacy 绕过错误。

磁盘检查会预留20%余量。完整 RDDM 状态和浮点中间图较大，完整重跑的容量按验收实测估算。阶段输出不覆盖；失败时保留结果，先分析错误，修复后使用新的诊断目录。程序不会自动删除任何结果。

## 3. A：重新核验第一轮产物与评价

```powershell
python diagnose_uie.py --phase audit --protocol fixed_fisher_strict --experiment_dir $exp --output_dir $diag --device cuda:0
```

核验原图、清单、源模型及固定输入。默认数值模式下重载 Source-only、Source-only+x8、DCTTA 三组权重，复算逐图指标，与第一轮比较，沿用预先定义的评价误差上限：PSNR 1e-5 dB、SSIM 1e-6。该误差界限只用于独立指标重载，不用于轨迹验收。

另存三组严格模式评价及两种模式的最大指标差。`source_train.json`、`source_val.json` 是默认模式的完整图评价；`source_crop_vs_full.json` 是严格模式下同一模型的裁剪/完整图对应区域差异。不要把不同模式的记录混作同一评价协议。

检查 `static_audit/status.json` 中 `reload_passed=true` 后执行下一阶段。它不代表退化评分或 reference 的人工合理性审查已完成。

## 4. B：至少22张的完整组件验收

```powershell
python diagnose_uie.py --phase accept --protocol fixed_fisher_strict --experiment_dir $exp --output_dir $diag --device cuda:0
```

依次执行同初态的无观测控制、重复控制、开启观测、从第1步完整状态恢复重放。要求每步损失及完整关键状态哈希精确一致。前缀必须至少22张，包含320尺寸和较小裁剪，并跨过实际 RDDM EMA 预热边界。如果覆盖不足，只按提示增大 `--acceptance_steps`，不改原始图片顺序。

本阶段加载旧 Fisher，不重新计算；输出 `acceptance/fisher_compatibility.json`，其中记录原始身份、当前身份及审查合同；复制出的 `fisher.pt` 与原文件按字节相同。

`acceptance/acceptance.json` 必须为 `passed=true`。其中与第一轮前缀 loss 的差异是描述结果，不作为位级相同承诺。固定源验证4张、目标12张观察集按样本ID哈希和尺寸分层确定，保存在 `probe_plan.json`，不按历史失败幅度挑选。

若严格模式报错或一致性验收失败，停止后续步骤，将错误报告和阶段结果回传；单图 Fisher 检查成功不能保证此阶段必然通过。

## 5. C：默认数值模式前缀对照与离线比较

```powershell
python diagnose_uie.py --phase numeric_control --protocol fixed_fisher_strict --experiment_dir $exp --output_dir $diag --device cuda:0
python diagnose_uie.py --phase compare_numeric --protocol fixed_fisher_strict --experiment_dir $exp --output_dir $diag --device cuda:0
```

第一条从验收完全相同的初态恢复，然后仅关闭全局严格确定性，执行相同长度的前缀。保留 cuDNN、TF32 和共用 CUBLAS 工作区设置，因此称为“默认全局数值模式对照”，不能声称完整重建了第一轮未保存的运行环境。

第二条读取两组快照，将学生和教师统一在严格模式下离线评价：第0步至前缀末尾，每步固定观察集及当步适配图；记录 PSNR/SSIM、两组增强输出的平均和最大绝对差。

主要结果：

- `numeric_control/environment.json`、`trajectory.json`、`loss_difference.json`、`status.json`、每步 `snapshots/`。
- `numeric_comparison/curves.csv`、`psnr.png`、每步学生/教师逐图 JSON、`status.json`。

两阶段均须完成。数值比较没有事后设定的 PSNR 放行阈值，也不会自动选择表现较好的模式。若差异明显，后续诊断解释必须考虑数值敏感性。快照和初态哈希被记录并在完整重跑前复核，不能更换对照输入后沿用成功标记。

## 6. D：133张完整诊断及离线检查

```powershell
python diagnose_uie.py --phase run --protocol fixed_fisher_strict --experiment_dir $exp --output_dir $diag --device cuda:0
python diagnose_uie.py --phase evaluate --protocol fixed_fisher_strict --experiment_dir $exp --output_dir $diag --device cuda:0
python diagnose_uie.py --phase inspect --protocol fixed_fisher_strict --experiment_dir $exp --output_dir $diag --device cuda:0
```

run 必须通过前述阶段及初态、数据、环境和代码身份核验。记录内容延续原方案：

- 每张图的 raw、8视图、教师伪标签、再退化图、学生两路输出、NU2Net tail 前后浮点值。
- 665次 RDDM 更新的时间步、噪声、损失、梯度裁剪前后、优化器、EMA 和五步采样中间量。
- 学生 Adam 更新、Fisher 恢复、教师 EMA 的参数变化；非有限值和恢复不变量检查。
- 第0～133步学生/教师快照；0/1/5/10/20/40/80/133完整恢复状态，包括优化器、RDDM、buffer、梯度、RNG、数值模式、顺序和游标。
- 逐步耗时、CUDA 内存、完整状态哈希；最终权重与第一轮差异。

evaluate 生成每步固定观察集曲线，边界评价全部源验证27张、目标133张，并保留当前适配图结果。inspect 默认检查第1、20、133步，在独立模型副本中计算六项加权损失的梯度范数与夹角、同图更新前后和裁剪/完整图差异，输出浮点结果与预览。

按异常区间追加检查可以另用新输出安排；当前命令不自动修改教师、tail、退化模型或超参数。

## 7. 回传与验证范围

本地完整回归：36项全部通过，0失败、0跳过，耗时303.218秒；结果见项目根目录 `tmp/uie_unittest_results.txt`。覆盖真实RDDM观测与重放、固定Fisher不重算、历史产物和身份拒绝检查、数值对照、完整记录、离线评价及梯度检查。第一轮三个固定产物及算法文件的哈希另行核验通过。

优先回传：`static_audit`、`acceptance` 下 JSON/JSONL、`numeric_control` 下 JSON/JSONL、整个 `numeric_comparison`、`protocol.json`、`run` 下 JSON/JSONL、`evaluation`、`inspection`、`DIAGNOSTIC_REPORT.md`，以及任何 `failure_*.json`。

大型 tensors、snapshots、replay_states 在程序机保留，后续按定位需要回传。不能只留PNG、删除浮点证据。无需重复运行 v1 的 fisher_probe。

本地测试使用 CPU、合成图片、小型 NU2Net 和轻量感知特征；完整流程集成测试使用真实 RDDM，并且只在测试中将验收样本数缩为2。正式命令仍强制至少22张并检查尺寸和EMA边界。测试通过仅证明本地实现和阶段衔接，不代表 TITAN RTX 的正式严格验收已经通过，也不能预先保证定位唯一根因。
