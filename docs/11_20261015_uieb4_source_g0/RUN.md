# 程序机：新源训练 → source_only → G0

日期：2026-10-15。程序机工程根目录为 `F:\qmh\DCTTA-uie`，使用原有 `resshift` 环境。开发机路径不能直接用于程序机清单。

## 同步哪些文件

开发机根目录：`E:\Codex\DCTTA-main\movetoUIE`。以下项目复制到程序机 `F:\qmh\DCTTA-uie` 下的同名相对位置，不打包。

新增或本轮调整的内容：

- `prepare_uieb4_experiment.py`
- `run_uieb4_g0.py`
- `.gitattributes`（通过Git同步时使用）
- `docs\11_20261015_uieb4_source_g0\` 整个目录，包含运行脚本和运行文件哈希清单。
- `docs\10_20261015_dataset_reclassification\` 整个目录，包含批准标签与近重复分组。
- `docs\README.md`
- `tests\test_uieb4.py`（测试文件，不是正式运行依赖）。
- `dataset\UIEB_reclassified_v2\` 整个目录：879对原字节图片及labels.csv等元数据，约1.458GB图像。若已完整同步，不必重复复制；prepare会逐一校验SHA256。

已有运行依赖建议覆盖同步一次，防止程序机停留在旧代码版本：

```text
train_source_uie.py
options.py
uie_tta.py
diagnose_uie.py
diagnose_visual_uie.py
run_strength_uie.py
run_factorial_uie.py
RDDM\net.py
RDDM\src\residual_denoising_diffusion_pytorch.py
uie_backbones\nu2net.py
uie_backbones\__init__.py
utils\loss_utils.py
utils\val_utils.py
```

这些历史算法文件本轮没有改动。`RUNTIME_FILES.json`检查上述依赖及新入口，文本哈希只容许LF/CRLF换行差异，其他改动明确拒绝运行。完整项目中其余已有导入模块仍应保留。继续使用已运行G0–G7的环境，不要重装根目录requirements.txt。

**不要复制开发机的 `experiments\...\manifests` 到程序机**，其中是开发机绝对路径。程序机执行prepare，生成自己的F盘清单。不要覆盖任何旧蓝色域实验目录。

如果程序机使用Git，可在保存其本地修改后fetch并切到 `codex/uieb4-source-g0`，再单独同步被Git忽略的dataset目录；不要对有本地改动的程序机执行reset --hard。

## 推荐的一条运行命令

在新PowerShell窗口执行，先确认GPU UUID仍为空闲卡：

```powershell
Set-Location F:\qmh\DCTTA-uie
conda activate resshift
nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.used --format=csv
& .\docs\11_20261015_uieb4_source_g0\RUN_PROGRAM_MACHINE.ps1 -GpuUuid 'GPU-53f5a782-8c8c-958b-006d-c8aad0f4d8a3'
```

这个UUID是之前空闲的nvidia-smi GPU0。脚本只对新窗口设置CUDA_VISIBLE_DEVICES，程序内部仍使用cuda:0；它代表被选中的那张物理卡。运行中的其他程序不受此窗口环境变量影响。若该卡现在不空闲，把参数换成确认空闲卡的UUID。

脚本依次执行：校验文件与全部原图 → 生成244/62/172清单 → 源模型250轮 → G0验收 → 正式172步 → 最终比较。任一步非零退出即停止后续阶段。可重跑同一脚本：源训练从last.pth轮次边界续训；G0从已提交的完整状态续跑；完成的G0只核验结果，不重复适配。

如果PowerShell因ExecutionPolicy阻止执行脚本，可使用下方独立Python命令，不需要修改系统策略。

## 分步命令

```powershell
Set-Location F:\qmh\DCTTA-uie
conda activate resshift
$env:CUDA_VISIBLE_DEVICES = 'GPU-53f5a782-8c8c-958b-006d-c8aad0f4d8a3'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$exp = 'F:\qmh\DCTTA-uie\experiments\UIEB4_light_to_severe_NU2Net_v1'
$data = 'F:\qmh\DCTTA-uie\dataset\UIEB_reclassified_v2'
python prepare_uieb4_experiment.py --phase prepare --dataset_dir $data --experiment_dir $exp
```

看到 `Prepared: source train 244 / val 62; target 172; moderate 401 sealed` 后，首次源训练：

```powershell
python prepare_uieb4_experiment.py --phase train --experiment_dir $exp --device cuda:0 --epochs 250 --batch_size 16 --num_workers 2 --patch_size 256 --basic_channel 64 --lr 0.001 --seed 2022
```

如果源训练中断，保持配置不变，使用：

```powershell
python prepare_uieb4_experiment.py --phase train --experiment_dir $exp --device cuda:0 --epochs 250 --batch_size 16 --num_workers 2 --patch_size 256 --basic_channel 64 --lr 0.001 --seed 2022 --resume "$exp\source\checkpoints\last.pth"
```

**源训练完成250轮后**执行：

```powershell
python run_uieb4_g0.py --phase preflight --experiment_dir $exp --device cuda:0 --max_gb 10
python run_uieb4_g0.py --phase run --experiment_dir $exp --device cuda:0 --max_gb 10
```

`run`自动完成accept验收，无需另跑accept。也可先用 `--phase accept` 单独验收，再用run继续。必须前一条成功才运行下一条；推荐用脚本自动检查退出码。

## 输出和回传

全部新结果位于：`F:\qmh\DCTTA-uie\experiments\UIEB4_light_to_severe_NU2Net_v1`。

| 相对位置 | 用途 |
|---|---|
| manifests\audit.json / 三个CSV | 程序机绝对路径、类别计数、哈希与划分 |
| source\checkpoints\best.pth / last.pth | 源验证选出的best、最后完整轮次及续训状态 |
| source\logs\epochs.csv / source\curves\ | 源训练曲线、逐轮L1、PSNR、SSIM |
| G0\contract.json / acceptance.json | 代码、数据、数值设置与实际CUDA验收证据 |
| G0\source_only\ | 新源模型对全部172张目标图的最终整图输出 |
| G0\teacher_pseudo\snapshots\步数\ | 固定10张的教师八视图整图伪标签 |
| G0\x_sd\snapshots\步数\ | 同10张的RDDM整图再退化 |
| G0\y_sd\snapshots\步数\ | 学生对整图再退化输入的输出 |
| G0\y_in\snapshots\步数\ | 学生对原图的整图输出 |
| G0\evaluation\步数.json | 全目标学生/教师、源验证学生指标；10张快照的浮点和PNG量化PSNR |
| G0\snapshots\步数.pth | 11个阶段的学生与教师小权重 |
| G0\final_y_in\ | G0最终模型对全部172张原图的增强结果 |
| G0\adapted_model.pth | 仅用于部署/评价的最终学生权重，不用于续训 |
| G0\comparison.csv / comparison.json | 逐图source_only与G0差值，总体、四类均值、类别等权均值 |
| G0\trajectory.json | 逐图适配损失与耗时；耗时不作为确定性一致性依据 |
| G0\resume_0.pth / resume_1.pth / progress.json | 完整状态双槽与提交指针，禁止单独拿adapted_model.pth续跑 |
| G0\complete.json | 完成状态及输出哈希索引 |

阶段步数：0、18、35、52、69、86、104、121、138、155、172。快照10张按类别和分辨率预选，不以目标GT分数选择。共440张阶段图片，加source_only172张、final_y_in172张，共784张输出图片。每阶段对全目标域计算指标，但只为固定10张导出四类阶段图像。

本版本对实际172张尺寸计算的G0保守峰值为7.025GB，低于10GB预算。固定样本为362_img_.png、841_img_.png、322_img_.png、734_img_.png、544_img_.png、112_img_.png、454_img_.png、455_img_.png、716_img_.png、15544.png；真实执行顺序见contract.json。完整在线状态每图原子保存，磁盘写入量远大于最终目录大小，耗时应包含这部分开销。

快照x_sd是固定随机种子的整图只读采样，区别于真实逐图适配中随机裁剪产生的训练张量。第0步RDDM尚未适配，不代表有效再退化。teacher_pseudo为八视图，y_in为单视图；可跟踪各自变化，不能把推理视图的差异都解释为EMA损坏。

10GB预算约束G0目录的磁盘峰值，包括原子保存临时副本；不代表显存上限。数据集、源训练输出独立存放，磁盘还要为它们留余量。禁止为通过预算而缩图。若CUDA OOM或严格算子不支持，入口报错停止，保留现场后回传日志，不会静默降级数值模式。

建议整个新实验目录回传，保留best、last、G0初态及双槽，以便后续复盘和恢复。回传前确认显示 `source_only / G0 COMPLETE`。不需要上传到Git，也不需要本地打包。
