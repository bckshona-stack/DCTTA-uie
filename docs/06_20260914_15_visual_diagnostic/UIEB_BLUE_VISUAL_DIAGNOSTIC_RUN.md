# G0 整图诊断运行说明

> 归档说明（2026-09-17）：本文件保留所属阶段的历史内容和当时状态；当前进展请看 [文档导航](../README.md)。文中“待运行”等描述不代表今天的状态。

## 整图 OOM 修复：已有 visual_v1 用户从这里开始

旧版在 RDDM 全局注意力中一次性构建全部空间位置的相似度矩阵。2180×1447 整图补齐后，瓶颈层约有 49,413 个位置，4 头 float32 的单个矩阵约 36.38 GiB，超过 24 GiB 显卡总容量。输出约 2.21 GB 是磁盘估计，与此显存申请无关。

修复版仅在诊断生成采样期间临时替换 RDDM EMA 的全局注意力前向：超过 1024 个空间位置时，每批计算 128 个 query，每批仍使用全体 key/value 并沿全部 key 做 softmax。保留全图尺寸、全局关系和 float32，不使用图像分块、缩放或混合精度。对应相似度和 softmax 临时张量合计约 0.19 GiB；其他卷积、中间特征和模型状态仍需要显存，不能将此数字理解为整次运行的峰值。计算总量仍为全局注意力，大图可能较慢。

分批矩阵计算可能有浮点舍入差异，因此新旧诊断图片不混合。本地对原密集实现的输出比较通过 rtol=1e-5、atol=1e-6 检查；真实 RDDM 强制使用分批路径时，诊断前后状态和实际 G0 更新结果完全一致。没有修改原 RDDM 源码和真实适配路径。正式 CUDA 大图峰值尚待程序机验证。

**修复操作：**

1. 将最新 diagnose_visual_uie.py 同步到 F:\qmh\DCTTA-uie，替换旧入口。无需替换 RDDM、uie_tta.py、diagnose_uie.py。
2. 保留旧 visual_v1 目录，不删除或修改其运行合同；新版本入口哈希已变化，不能继续往旧目录写。
3. 在原 PowerShell 环境中运行下列命令，沿用原 $exp、$diag 和 GPU 设置。每一步成功后再执行下一步。

~~~powershell
$oldout = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_visual_v1'
$out = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_visual_v2'
python diagnose_visual_uie.py --phase select --experiment_dir $exp --diagnostic_dir $diag --output_dir $out --selection_csv "$oldout\selected_samples.csv"
python diagnose_visual_uie.py --phase preflight --experiment_dir $exp --diagnostic_dir $diag --output_dir $out --device cuda:0
python diagnose_visual_uie.py --phase run --experiment_dir $exp --diagnostic_dir $diag --output_dir $out --device cuda:0
~~~

新目录复用原名单、源权重、Fisher 和原 G0 初态；重新生成本次诊断图片，不重训源模型。若仍报错，请保留从 Traceback 开始的完整堆栈，以区分其他整图网络层的显存峰值。无需先修改 PYTORCH_CUDA_ALLOC_CONF。

## 本次交付与运行状态

新增独立入口 diagnose_visual_uie.py。原 uie_tta.py、diagnose_uie.py、RDDM 和已验收 G0 代码保持不变。

已完成本地选图和容量估算，正式输出预计 995 张 PNG，保守约 2.21 GB（含 20 MB 元数据预留），上限 10 GB。已提供选图文件 UIEB_BLUE_VISUAL_SELECTED.csv。

本地已通过六项测试，包括真实 RDDM 非方形整图采样、PNG/PSNR 口径、诊断状态恢复、实际更新精确一致，以及采样异常后的恢复。另用原小型 G0 测试资产验证了两张图片完整流程：28 张 PNG、20 行指标、中断续跑、已完成任务跳过、文件篡改拒绝。

**正式 133 张 CUDA 运行尚未执行。**当前开发机是 CPU PyTorch 环境，回传目录没有完整 replay_states；正式运行需要在保留原 strict v2 环境、数据及完整状态的程序机完成。不要为此重训源模型、重算 Fisher 或重跑旧验收。

## 1. 同步文件

从开发目录 movetoUIE 同步以下文件到程序机项目根目录 F:\qmh\DCTTA-uie，保持相对位置：

- diagnose_visual_uie.py
- docs/06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_SELECTED.csv
- docs/06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_DIAGNOSTIC_PLAN.md
- docs/06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_DIAGNOSTIC_RUN.md
- tests/test_visual_diagnostics.py（可选，用于测试）

本次无需替换原诊断或算法文件。不要升级原环境依赖。

## 2. 设置原运行环境

在 PowerShell 执行：

~~~powershell
conda activate resshift
cd F:\qmh\DCTTA-uie
$env:CUDA_VISIBLE_DEVICES="1"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
$exp = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net'
$diag = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_diagnostic_v2'
$out = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_visual_v2'
~~~

GPU 映射沿用原 G0；若当时使用了其他映射，应保持原映射。严格预检核验原环境、代码、数据和权重，不会静默放宽要求。

必需旧产物包括源 best.pth、固定 Fisher、原始 raw/reference、source_only/run.json，以及 strict v2 的 protocol.json、acceptance/initial.pth、run/trajectory.json、run/replay_states/000000.pth 和其他保留的重放边界、evaluation 的第 0 步与最终评价。

## 3. 生成或导入 10 张选图 CSV

使用本次已选好的名单：

~~~powershell
python diagnose_visual_uie.py --phase select --experiment_dir $exp --diagnostic_dir $diag --output_dir $out --selection_csv docs/06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_SELECTED.csv
~~~

看到 Selection ready 后，打开 $out\selected_samples.csv 检查。也可以省略 --selection_csv，让程序根据已有结果自动选图。此阶段不加载模型、不进行适配，在只有回传结果的开发机也能运行。

当前名单：

| 类别 | 文件名 |
|---|---|
| 最大退化 | 507_img_.png、424_img_.png、425_img_.png |
| 变化最小 | 78_img_.png、391_img_.png、10_img_.png |
| 最大改善 | 518_img_.png、869_img_.png |
| 最大分辨率 | 841_img_.png，2180×1447 |
| 最小分辨率 | 284_img_.png，276×182 |

先保留两张最大增益样本，再选退化和变化最小样本，避免少量改善样本被重复占用。最后从剩余样本补分辨率两端。CSV 必须恰好包含 10 个不重复的目标 sample_id。可在首次 preflight 前更改 sample_id/reason；尺寸和分数会从原结果重新读取，不能靠改 CSV 分数影响排序或指标。preflight 后选择被冻结，要改变选择需使用新的输出目录。

## 4. 正式预检

~~~powershell
python diagnose_visual_uie.py --phase preflight --experiment_dir $exp --diagnostic_dir $diag --output_dir $out --device cuda:0
~~~

看到 Preflight PASSED 后再进行下一步。预检核验原环境/代码/初态、数据内容、source_only 权重身份、整图尺寸、名单及容量。

约 2.21 GB 是保守输出估计，不是显存估计。最大整图的计算比 320 图块重；若出现 CUDA out of memory，程序不会自动缩图或改为拼块，保留报错以便诊断。

## 5. 运行

~~~powershell
python diagnose_visual_uie.py --phase run --experiment_dir $exp --diagnostic_dir $diag --output_dir $out --device cuda:0
~~~

同一输出目录一次只运行一个进程。运行过程显示 G0 总进度和每阶段 10 张图的快照进度，并输出 Committed k/133 及已用磁盘空间。

实际阶段为：0、14、27、40、54、67、80、94、107、120、133 次更新。第 0 阶段先输出 10 张图的三类快照，再开始适配。随后每个目标图依次执行：

1. 保存更新前教师八视图伪标签、当前生成器的整图 x_sd、学生 y_sd。
2. 恢复诊断前完整状态，执行原 G0 的图块适配。
3. 核验 loss 和完整状态哈希与旧 G0 对应步骤完全一致。
4. 保存更新后教师伪标签，以及同一张未量化 x_sd 上的 y_sd。
5. 到达固定阶段时，为 10 张图重新生成快照；最后提交本步骤进度。

每个 sample_id 使用由原 seed 和文件名派生的固定生成种子，阶段之间不变。当前生成器会随着真实适配更新，所以相同种子不保证跨阶段 x_sd 相同。

**诊断不额外训练 RDDM。**现有算法用教师伪标签训练 RDDM，而采样条件是 raw。整图导出调用当时 RDDM EMA 的采样路径；阶段 0 的生成器还没有进行目标域在线训练。实际适配前后的整图记录不是当次损失实际使用的裁剪图块，也不是把旧图块放大成整图。

整图为保持 RDDM 尺寸要求，仅在右/下补齐到 8 的倍数且最小 32，采样后裁回原尺寸。不缩放、不拼接。教师使用原八视图逆变换均值，学生使用当前模型单次前向。

## 6. 输出结构与指标

~~~text
$out/
  teacher_pseudo/
    online/000001/before/原文件名.png
    online/000001/after/原文件名.png
    snapshots/000000/原文件名.png
    snapshots/000014/原文件名.png
    ...
  x_sd/
    online/000001/before/原文件名.png   # after 共用此图，不重复保存
    snapshots/000000/原文件名.png
    ...
  y_sd/
    online/000001/before/原文件名.png
    online/000001/after/原文件名.png
    snapshots/000000/原文件名.png
    ...
  selected_samples.csv
  metrics.csv
  snapshot_summary.csv
  storage_estimate.json
  run_manifest.json
  progress.json
  trajectory.json
  status.json
  bootstrap/                              # 只含构建核验记录/固定 Fisher 副本
~~~

PNG 输入保留原文件名；其他格式追加 .png。三个目录预计分别为教师 376 张、x_sd 243 张、y_sd 376 张，共 995 张。其他目录不保存额外诊断图片，不保存累计的大模型状态。

metrics.csv 预计 752 行：全目标更新前后 133×4=532 行，阶段快照 10×11×2=220 行。字段：

- sample_id、kind、step、timing、role：定位图片、更新阶段、before/after。
- psnr：输出浮点张量先截断到 [0,1]，与原整图 reference 比较，RGB、data_range=1；与既有评价口径一致。
- png_psnr：同一输出量化为 PNG 后的 PSNR，方便与保存图片复算核对。
- source_only_psnr：直接复用原 source_only/run.json 对应记录，不重跑基线。
- delta_source_psnr：相对 source_only 的变化。
- delta_before_psnr：同图同角色更新后减去更新前；只在 online/after 填写。
- width、height、sampling_seed、x_sd_tensor_sha256：记录尺寸、种子及学生实际诊断输入身份，前后须完全相同。

不计算 x_sd 的 PSNR。教师八视图伪标签在第 0 步也可能优于/差于 source_only 单视图，这不是教师已经漂移的证据；观察漂移时还需比较同图教师伪标签自身的第 0 步值。

snapshot_summary.csv 仅汇总选中 10 张图的阶段均值，不能当成全部 133 张的平均表现。全目标 online 指标来自不同时间的模型，也不能当成最终模型的统一测试结果。reference 只用于评价，不进入参数更新或决定何时停止。

## 7. 完成、中断与异常

成功标志：控制台 Visual diagnosis COMPLETE，且 status.json 的 status 为 complete、images 为 995、metric_rows 为 752、exact_g0_replay 为 true。

中断后重复相同 run 命令即可。已提交图片先核验哈希并保留；程序从旧 G0 最近可用完整状态恢复，重放到已提交游标，随后继续导出。不新增大状态文件，因此重启时可能重算少量已完成的适配步骤，但不会重新生成已提交图片。一个步骤及其阶段快照未完成提交时，下次会重新生成该步骤的未提交文件。

已全部完成后重复运行只核验并跳过。不要手动修改 run_manifest.json、progress.json 或已提交图片。出现 Exact G0 replay failed、环境不一致、文件哈希错误或非有限值时，立即停止，不能以改哈希/关闭严格模式继续。

容量限制包含新增输出目录的内容和原子写入临时文件，超过上限会停止。旧实验产物和原重放状态不计入新增 10 GB，但必须继续保留。程序不自动删除旧数据。异常时请保留控制台完整报错及 progress.json，未出现 COMPLETE 不代表已完成。

## 8. 回传与查看顺序

回传整个新增 $out 目录即可，预计明显小于 10 GB；不必再次回传旧权重或旧实验目录。

先打开 selected_samples.csv，按同一文件名纵向查看 teacher_pseudo/snapshots；用 metrics.csv 核对同图 PSNR 是否下降、从哪一阶段开始。随后对照同阶段 x_sd、y_sd，并查看 online 对应图的更新前后差值。

这些结果用于检查学生、教师及再退化路径的变化顺序，不将同时变差直接等同于因果证明。真实 CUDA 运行还会逐步核验是否精确复现原 G0，完成后才能据正式输出分析。

## 9. 本地测试入口

在具备项目依赖的环境、项目根目录运行：

~~~powershell
python -m unittest discover -s tests -p test_visual_diagnostics.py -v
$env:DCTTA_TEST_RDDM='1'
python -m unittest discover -s tests -p test_visual_diagnostics.py -v
~~~

第一条默认跳过真实 RDDM；第二条开启。测试目录保留在上一级 tmp 中，不批量删除。小型 CPU 测试验证功能和状态隔离，不能替代程序机上 133 张原轨迹的严格核验。
