# 第一阶段：真实单步更新诊断

> 归档说明（2026-09-17）：本文件保留所属阶段的历史内容和当时状态；当前进展请看 [文档导航](../README.md)。文中“待运行”等描述不代表今天的状态。

2026-09-12。用户确认后实施。目的：区分适配目标、Adam实际更新、Fisher恢复及裁剪上下文分别如何影响真实图像恢复。不在本阶段改变教师、RDDM、tail、学习率或源模型。

## 1. 本次同步

仅将本地 `movetoUIE/diagnose_update_uie.py` 同步至程序机 `F:\qmh\DCTTA-uie\diagnose_update_uie.py`。

保留已有 `diagnose_uie.py`、`options.py`、`uie_tta.py`、NU2Net及RDDM文件。独立入口为了保持v2原始验收代码哈希不变；诊断仍集中在一个文件，不新增算法模块。不必重新做audit或22张验收。

必须保留程序机v2的原始完整 `run/replay_states/`、`acceptance/initial.pth`、轨迹和其他JSON，以及第一轮源最佳权重、Fisher与真实数据。新程序从原状态重放，不从零重新训练。

## 2. 运行命令

在原resshift环境、同一GPU映射下：

```powershell
conda activate resshift
cd F:\qmh\DCTTA-uie
$env:CUDA_VISIBLE_DEVICES="1"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
$exp = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net'
$diag = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_diagnostic_v2'
$upd = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_update_diagnostic_v1'
```

先执行重放：

```powershell
python diagnose_update_uie.py --phase replay --experiment_dir $exp --diagnostic_dir $diag --output_dir $upd --device cuda:0
```

成功后执行离线评价：

```powershell
python diagnose_update_uie.py --phase evaluate --experiment_dir $exp --diagnostic_dir $diag --output_dir $upd --device cuda:0
```

也可使用 `--phase all` 一次顺序执行两阶段，但不要在已完成replay的同一目录重新执行all。每个阶段拒绝覆盖已有子目录；失败后保留结果，先检查错误报告。

默认检查第1、20、133步。正常已有边界分别为0、10、80，因此实际重放1、10、53张，而不是只执行3次适配。这样才能保留真实Adam动量、RDDM在线状态和历史，不得用新建Adam替代。重放显式显示 `Replay to step ...` 进度；每个检查点完成后显示 `exact replay PASSED`。离线阶段显示五种模型状态评价进度，最终输出 `Update diagnosis COMPLETE`。

后续若需第80步，另用新的输出目录并指定 `--steps 80`，不要覆盖当前结果。

## 3. 验证门槛

- 原v2的代码、数值模式、环境、源权重、Fisher、目标内容、配置和顺序一致。
- 原完整边界快照内容哈希匹配v2轨迹；恢复后的完整状态再次匹配。
- 每一张重放后的loss和完整状态哈希与v2相同。
- 被检查步骤的raw、伪标签、confidence、再退化图及两路输出，与原记录张量哈希相同。
- 离线重算的再退化前向与真实输出完全相同；保留原运算顺序重算的总梯度，与原Adam实际收到的梯度完全相同。

任何检查失败即停止，不使用浮点容差放行原轨迹，不继续给出因果结论。运行环境变更或显存不足时保留错误，不自动改尺寸或切换计算模式。

## 4. 保存哪些证据

重放过程仅复制真实值，不读取reference作为更新输入。保存更新前、Adam后、Fisher恢复后学生权重、更新前教师、实际梯度、Fisher掩码、原图和再退化图等小型诊断包；不重复保存大型RDDM完整快照。

离线评价增加源模型对照，一共五种状态：before、adam、restored、teacher、source。每个状态记录：

- 当前raw裁剪、完整raw以及再退化图相对相应reference的PSNR、SSIM、L1、MSE和有符号RGB误差。
- 完整图对应区域与裁剪输出之间的MAE。
- 与v2一致的源验证4张、目标12张固定观察集的逐图指标。
- 每种输入的浮点增强输出。

另外计算六项加权适配梯度、两分支合成梯度和真实总梯度；与真实raw裁剪及完整图的L1/MSE监督诊断梯度比较。每项分别在全部候选参数和Fisher未保护子空间报告float64内积、余弦和范数。

实际更新方向包括 `delta_adam`、`delta_restored` 和 `delta_fisher_correction`，均来自真实权重差。对于真实监督梯度g与实际变化delta，`dot(g, delta)<0` 表示一阶近似有利；最终仍以实际误差变化核验。梯度与梯度的正余弦则表示下降方向较一致，两种符号含义不要混淆。

RDDM采样x_start按其[-1,1]内部范围报告低于-1、高于1的比例；raw/再退化图在[0,1]报告逐通道零值、满值比例及均值。修正此前把below_zero误用为潜在截断率的风险。

reference只在离线阶段用于评价和方向诊断，不执行监督optimizer.step，不参与重放、选权重或停止条件。再退化图与reference的比较是一项诊断指标，并不预先假定再退化严格保留全部内容。

## 5. 回传

```text
$upd/
  replay/
    status.json
    000001.pth       # 小型更新诊断包，留在程序机
    000020.pth
    000133.pth
    bootstrap/...   # 初始化及固定Fisher兼容记录，留在程序机
  evaluation/
    000001.json
    000001_outputs.pt
    000020.json
    000020_outputs.pt
    000133.json
    000133_outputs.pt
    status.json
  failure_*.json     # 仅在发生错误时存在
```

回传整个 `evaluation/`、`replay/status.json` 及存在的错误报告，到本地 `movetoUIE/experiments/UIEBblue_weaktostrong_NU2Net_update_diagnostic_v1/`。replay中的pth先留在程序机，不删除。JSON可判断更新影响，outputs.pt用于进一步浮点分析。

## 6. 实施范围与后续

本阶段得到结果后，再决定固定源教师和替换再退化输入的22步独立对照。当前入口没有实现或偷偷执行这两个算法干预，也不提前改tail。单步证据结合后续干预，才能判断主要原因与反馈放大因素。

本地验证使用此前保存的两步CPU合成实验、真实RDDM和轻量感知特征。它验证原状态恢复、逐步精确匹配、真实梯度一致以及离线评价流程；不能替代程序机正式CUDA/VGG链路的检查。

本次两项针对性测试全部通过，耗时24.738秒，包括真实两步重放/离线集成测试、float64方向比较及篡改诊断包拒绝检查。记录见项目根目录 `tmp/update_diagnosis_tests.txt`。没有重复运行无关的36项旧回归，也没有重新生成大型RDDM测试快照。
