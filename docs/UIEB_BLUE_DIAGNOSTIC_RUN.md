# UIEB 适配诊断运行说明

2026-09-10。诊断代码已实现；正式数据核验、22张前缀验收和133张诊断需在程序机运行。此流程不更改第一轮的源模型、数据划分、裁剪、学习率、教师更新或损失权重。

2026-09-11更新：程序机audit已通过，accept因Fisher不完全一致而停止。现阶段先按 [Fisher核查与探针说明](UIEB_BLUE_FISHER_MISMATCH_20260911.md) 执行新增 `fisher_probe`；以下常规accept/run流程暂不继续。新旧数据和源权重一致，但114个接近零Fisher的偏置掩码位置发生变化，尚需程序机同环境重复与算子检查。

最新更新（2026-09-11）：用户已确认，新协议已实现。**实际执行请使用 [第二版诊断运行说明](UIEB_BLUE_DIAGNOSTIC_V2_RUN.md)**，新建 `_diagnostic_v2`，包含固定旧Fisher、严格验收和默认数值模式对照。以下为第一版历史说明，不能与新版命令混用；不要重复运行已有fisher_probe。

## 1. 同步文件

把本地 movetoUIE 下这些文件同步到程序机 `F:\qmh\DCTTA-uie` 的相同相对位置：

- `diagnose_uie.py`（新增集中式诊断入口）
- `options.py`
- `uie_tta.py`
- `RDDM/src/residual_denoising_diffusion_pytorch.py`

NU2Net 原网络不改。无需重训源模型、重新准备清单或删除旧结果。保留程序机原环境；勿为诊断替换正在使用的 CUDA PyTorch。程序使用正式预训练感知网络；本地合成集成测试的轻量特征替代不用于正式运行。

## 2. 设置路径与显卡

在独立 PowerShell 窗口运行：

```powershell
conda activate resshift
cd F:\qmh\DCTTA-uie
$env:CUDA_VISIBLE_DEVICES="1"
$exp = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net'
$diag = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_diagnostic_v1'
```

物理 GPU 1 在该进程中映射为 cuda:0。后续五个命令分阶段执行；前一步报错时先核查，不跳过门槛。已有阶段输出不覆盖；失败结果应保留，处理原因后用新的诊断目录重跑，不批量删除。

## 3. 阶段 A：只读原始数据，独立核验评价

```powershell
python diagnose_uie.py --phase audit --experiment_dir $exp --output_dir $diag --device cuda:0
```

核验所有 raw/ref 内容哈希、清单与源训练关系；重载三组模型复算指标；计算目标 raw 基线、源训练/验证完整图指标和源模型裁剪/完整图差异。输出在 `static_audit/`。

独立重载对比第一轮逐图 PSNR 最大允许差1e-5 dB、SSIM 1e-6；超过则报错，不隐瞒环境/评价差异。`status.json` 的 `reload_passed=true` 是进入验收的必要条件。

若有原始 severity_scores.csv，可在本命令追加 `--severity_scores '实际文件路径'`。它只保留评分和身份供独立分析，不参与适配或重新划分。自动核验不能代替退化评分有效性和 reference 的人工核查；这两项始终单独列为待审查，不能把 reload_passed 当成“数据划分完全正确”。

## 4. 阶段 B：诊断开关与恢复验收

```powershell
python diagnose_uie.py --phase accept --experiment_dir $exp --output_dir $diag --device cuda:0
```

前缀默认22张，必须同时包含320和较小crop，并跨过实际RDDM EMA预热边界。若不足，按报错增大 `--acceptance_steps`，不改变原目标顺序。验收会依次执行无诊断控制、无诊断重复、开启诊断，以及从第一步完整状态恢复到前缀末尾。因而耗时明显超过单次22张适配。

对比全部关键状态及逐步loss的SHA-256，要求精确一致；不只比较最终PSNR。单独重新计算Fisher并与第一轮缓存数值/掩码精确比较；源码身份变化不通过绕过缓存校验解决。运行环境或实现变化后旧验收不再有效。

验收前也会检查磁盘空间，并为首次更新后才创建的优化器动量预留容量；完整重跑的估算使用已实测的最大单步张量量和成熟状态文件大小。空间不足时不会自动删除文件。

`acceptance/acceptance.json` 的 `passed=true` 才能继续。报告同时包含与第一轮已保存loss的前缀最大差异；若与第一轮不同，后续诊断解释的是当前可复现控制运行，不能宣称位级复现了第一轮。

## 5. 阶段 C：完整133步记录

```powershell
python diagnose_uie.py --phase run --experiment_dir $exp --output_dir $diag --device cuda:0
```

入口检查已通过的验收、环境/代码/配置/裁剪及初始化身份，并根据验收实际存储量估算容量，要求至少20%余量。建议提前确保磁盘空间充足；完整RDDM状态远大于NU2Net权重，不能仅按3.4GB的学生/教师快照估算总量。

实际输出结构：

```text
$diag/
  static_audit/
  acceptance/
  protocol.json
  run/
    fisher.pt
    fisher_statistics.json
    steps.jsonl                 # 133条，含逐步观测事件与张量引用
    rddm_steps.jsonl            # 665条RDDM子更新
    tensors/000001/...pt        # 原dtype浮点图、训练噪声与采样中间量
    snapshots/000000.pth       # 0～133每步学生和教师
    replay_states/000000.pth   # 0/1/5/10/20/40/80/133完整在线状态
    trajectory.json           # 每步loss和完整状态身份
    resources.jsonl            # 每步耗时和CUDA峰值分配/保留内存
    original_comparison.json  # 最终权重与第一轮的比较
    status.json
```

完整状态包括学生、教师、源参数、Fisher、Adam、RDDM在线/EMA/RAdam、buffer设备与内容、模块模式、梯度、scaler、RNG、顺序和游标。诊断执行前物化原DataLoader产生的顺序，后续按该列表读取；没有尚未消费的随机Sampler状态，重放按保存的顺序和游标执行。

关键图包括八视图、raw、伪标签、再退化、两路学生输出、tail前后；RDDM包含每次训练t和噪声、原始损失、梯度裁剪前后、EMA实际变化、扩散init前后buffer身份、采样各时间对和截断前预测。观测不额外调用采样或反向传播。

代码会检查133/665记录数、每次五步采样完整性、Fisher恢复不变量；异常标记 incomplete，不生成成功结论。

## 6. 离线曲线与分项检查

```powershell
python diagnose_uie.py --phase evaluate --experiment_dir $exp --output_dir $diag --device cuda:0
python diagnose_uie.py --phase inspect --experiment_dir $exp --output_dir $diag --device cuda:0
```

evaluate 使用每步快照评价固定源验证4张、目标12张观察集；指定边界评价全部目标133张和源验证27张，另外保留当前适配图评价。观察集在重跑前按尺寸分层和样本ID哈希确定，不按第一轮失败程度选择。

输出 `evaluation/probe_curves.csv`、PSNR/SSIM曲线、逐步逐图JSON和 `DIAGNOSTIC_REPORT.md`。reference只在此离线评价或前述静态核验中读取，不影响适配更新或停止时机。

inspect 默认检查第1、20、133步，可显式提供 `--inspect_steps 1 5 20 80 133`。在独立模型副本上，复用已保存raw/伪标签/再退化图，计算六项加权损失的梯度范数与夹角，以及同图更新前后、crop与完整图对应区域的差异。输出浮点结果和固定[0,1]显示的PNG预览，不自动调整任何损失。每次检查使用新inspection目录，已有输出不会覆盖。

首次异常区间的进一步重放/干预由诊断报告决定；当前命令不自动冻结教师、替换tail、扫描学习率或重分数据。

## 7. 回传内容与限制

优先回传 `static_audit`、`acceptance/acceptance.json`、`protocol.json`、`run`中的JSON/JSONL、`evaluation`、`inspection`和诊断Markdown。完整tensors、snapshots和replay_states在程序机保留，按后续定位需要回传；不要只留下PNG而丢掉浮点证据。

2026-09-10 用户清理临时测试目录后，最新代码完成包含真实RDDM的完整回归复验：33项全部通过，0失败、0跳过，耗时170.851秒。覆盖记录开关与完整状态恢复一致性、静态重载、离线曲线与梯度检查、非有限梯度和磁盘不足防护；测试记录见 `tmp/uie_unittest_results.txt`（本地项目根目录下）。本轮临时产物约4.1GB，未再次重复生成多轮快照。

本地短程合成测试不能代替程序机22张正式配置验收。先前空间不足造成的失败及处理已记录在实验日志中。程序机原数据缺失、Fisher不一致、重载误差超限或轨迹差异均会显式阻止依赖阶段，不会自动降低验证标准。

目标reference若用于指导后续算法/超参数选择，这批数据应作为开发数据，最终改良效果需另用未参与选择的数据验证。
