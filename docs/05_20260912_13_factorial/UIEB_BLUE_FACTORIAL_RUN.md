# G1/G2/G3 三组对照运行说明

> 归档说明（2026-09-17）：本文件保留所属阶段的历史内容和当时状态；当前进展请看 [文档导航](../README.md)。文中“待运行”等描述不代表今天的状态。

新增入口 `run_factorial_uie.py`。G0使用原strict v2完整结果，零更新使用v2第0步评价，不重跑源训练、Fisher、G0或原验收。

## 同步与设置

只同步本地 `movetoUIE/run_factorial_uie.py` 到程序机 `F:\qmh\DCTTA-uie\run_factorial_uie.py`。保留原v2代码、结果、完整初态及数据。

```powershell
conda activate resshift
cd F:\qmh\DCTTA-uie
$env:CUDA_VISIBLE_DEVICES="1"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
$exp = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net'
$diag = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_diagnostic_v2'
$fac = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_factorial_v1'
```

只读预检并建立实验合同：

```powershell
python run_factorial_uie.py --phase preflight --experiment_dir $exp --diagnostic_dir $diag --output_dir $fac --device cuda:0
```

看到 `Preflight PASSED` 后执行：

```powershell
python run_factorial_uie.py --phase all --experiment_dir $exp --diagnostic_dir $diag --output_dir $fac --device cuda:0
```

顺序为G1运行→G2运行→G3运行→三组离线评价→与已有G0汇总。全部按预定133张完成，不按reference选择停止时机。

| 组别 | 教师 | 学生第二路输入 |
|---|---|---|
| G1 | 固定源教师 | RDDM再退化 |
| G2 | EMA教师 | raw |
| G3 | 固定源教师 | raw |

三组都保留RDDM训练和采样调用。每组从v2原第0步完整状态独立开始，不从其他组终态接着训练；原 `adapt`、损失、Adam、Fisher代码不改。

## 中断与重复运行

同一目录不要同时启动两个进程。代码和环境在整个实验期间保持不变。

正常中断后可重复上述all命令：完成组先核对产物哈希，然后跳过；未完成组恢复其最近原子提交的完整状态，已完成评价文件按索引哈希核对后复用。

完整状态在1/5/10/20/40/80/133等边界提交到一个 `resume.pth`，包含优化器、RDDM、教师、Fisher、RNG、游标和已提交日志/文件身份。每次原子替换同一明确文件，避免累计多份大状态；未到边界的步骤属于未提交区间，中断后会重算该区间并覆盖相应未提交快照。不是每张都能无损续跑，但不会重跑已提交边界之前的步骤。

遇到哈希不匹配、非有限值等错误，先回传failure报告，不要通过修改原产物或删除合同强行续跑。已有结果不会被自动删除。

也可以分阶段执行，例如：

```powershell
python run_factorial_uie.py --phase run --group G1 --experiment_dir $exp --diagnostic_dir $diag --output_dir $fac --device cuda:0
python run_factorial_uie.py --phase evaluate --group G1 --experiment_dir $exp --diagnostic_dir $diag --output_dir $fac --device cuda:0
```

将G1换成G2/G3可单独处理对应组。三组评价成功后：

```powershell
python run_factorial_uie.py --phase compare --experiment_dir $exp --diagnostic_dir $diag --output_dir $fac --device cuda:0
```

## 评价与结果

沿用v2的固定4张源观察图、12张目标观察图，每步另外保留当前图；边界评价全部目标133张和源验证27张。所有reference均只在离线评价时参与计算。最终结论使用完整集合，不能把4/12张曲线当成全量指标。

第0步直接复用源模型评价；G1/G3固定教师在快照身份核验后，也复用源模型对应图片指标。G0只读取已有JSON，不进行模型适配或推理。每个组保存两项loss、Adam和Fisher恢复后的更新范数、到源参数距离、raw及生成图RGB均值。raw组若接近源模型且更新极小，应解释为避免了退化，不应直接称为改良成功。

comparison/updates.json同时提供G0原日志和三组更新量。G0范数来自原float32逐张量统计，新组使用float64归约，不能根据极小精度差异作机制判断。

结果结构：

```text
$fac/
  plan.json
  G1/                       # G2、G3相同
    run_status.json
    trajectory.json
    progress.json
    resume.pth              # 程序机保留，不必回传
    snapshots/              # 程序机保留
    bootstrap/              # 固定Fisher兼容检查记录
    evaluation/             # 指标JSON、index.json、status.json
  comparison/
    summary.json            # 四组最终全量PSNR/SSIM及下降张数
    paired.json             # 逐图源模型及四组结果、主效应对比与交互差
    curves.json
    updates.json
    psnr.png
    status.json
```

回传整个 `comparison/`、三组各自的 `evaluation/`、`trajectory.json`、`run_status.json` 和根目录 `plan.json`。存在错误时同时回传failure JSON。pth大文件先留在程序机，不删除。

当前阶段不运行新的随机顺序、小步长组、逐图重置或第91步检查。根据三组结果再决定是否需要后续实验。

## 本地验证

使用此前保留的CPU两张合成数据、真实RDDM及轻量感知特征完成三类验证：三组运行及模拟提交后中断续跑；固定教师缓存评价、G0日志复用和完成任务跳过；关闭两项干预时，新包装及观测仍与原两步loss和完整状态哈希精确一致。正式CUDA133张尚未运行。记录位于项目根目录tmp/factorial_test_result.txt、factorial_evaluation_test_result.txt、factorial_control_test_result.txt。未重跑源模型训练或正式G0。
