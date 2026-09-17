# DCTTA 水下图像增强迁移项目

本项目基于 DCTTA（Degradation-Consistent Test-Time Adaptation），将测试时适配框架迁移到水下图像增强（UIE），恢复骨干采用 NU²Net。

本仓库根目录为 `movetoUIE/`，程序机部署目录为 `F:\qmh\DCTTA-uie`。源域监督训练、目标域适配和独立推理已分开；目录内保留部分 PromptIR 对照代码。

## 文档导航

文档已按实验阶段归档，先看 [文档导航](docs/README.md)。当前工作为 [两组再退化使用强度对照方案](docs/07_20260917_strength_control/UIEB_BLUE_STRENGTH_CONTROL_PLAN.md)，尚未实现或运行；Git操作建议见 [版本管理说明](docs/00_project/GIT_WORKFLOW.md)。

## 当前功能

- **源域监督训练**：使用 raw/reference 成对图像训练 NU²Net；通过 CSV 清单划分数据，原图不移动；保存最佳权重、最后一轮权重、训练日志及曲线，支持续训。
- **目标域测试时适配**：使用目标 raw，执行教师八视图伪标签、RDDM 在线再退化、学生一致性与教师约束、Fisher 参数恢复及教师 EMA。目标 reference 仅用于评价。
- **独立推理与评价**：支持源模型、八视图集成和已适配模型；只有 raw 时保存增强图，有 reference 时计算 PSNR/SSIM。NU²Net 完整图推理内部补齐到8的倍数，输出裁回原尺寸。
- **诊断与对照**：静态核验、精确重放、数值模式对照、单步 Adam/Fisher 检查，以及固定教师/替换再退化输入的三组对照。

固定教师、将学生第二路替换为 raw 等属于诊断变体，并非原论文完整算法。已有实验发现原配置存在明显负迁移，不能将当前适配默认视为一定优于源模型。

## 项目结构

```text
movetoUIE/
├── README.md                       # 项目说明
├── .gitignore                      # 数据、权重、结果与缓存排除规则
├── train_source_uie.py              # 源域监督训练及清单准备
├── adapt_target_uie.py              # 目标适配与基础方法比较
├── test_uie.py                      # 独立推理、八视图集成及评价
├── uie_tta.py                       # 集中的数据、适配与推理实现
├── options.py                      # 各入口参数
├── uie_backbones/nu2net.py          # NU²Net 网络
├── RDDM/                           # 在线退化生成模块
├── utils/                          # 损失与评价工具
├── diagnose_uie.py                 # 完整适配诊断
├── diagnose_update_uie.py          # 单步更新重放与离线分析
├── run_factorial_uie.py            # G1/G2/G3 路径对照
├── tests/                          # 源训练、迁移和诊断测试
├── docs/                           # 已有方案、运行说明与实验记录
├── images/                         # 说明图片
├── third_party/                    # 第三方来源与许可记录
├── requirements_uie.txt            # UIE 环境依赖
├── train_test_promptir.py / tta.py  # 保留的 PromptIR 适配对照
├── test_promptir.py / net/         # 保留的 PromptIR 测试与网络
├── requirements.txt                # 原项目依赖参考
├── dataset/ / testdata/            # 本地数据，不纳入 Git
├── pretrain/                       # 本地预训练权重，不纳入 Git
└── experiments/                    # 清单、权重和实验产物，不纳入 Git
```

UIE 工作使用下面的明确入口。上一级原 DCTTA 文件仅作为本地对照，不属于本仓库。

## 环境与运行位置

以下 UIE 命令均在 `movetoUIE/` 中执行；程序机若把该目录内容直接部署为 `F:\qmh\DCTTA-uie`，则在该目录执行。

```powershell
cd F:\qmh\DCTTA-uie
python -m pip install -r requirements_uie.txt
```

请先选择与显卡兼容的 PyTorch/torchvision。已有程序机环境为 `resshift`，正式诊断使用 PyTorch 2.1.2+cu118；复现旧诊断必须保持其记录的依赖、代码和数值环境一致。感知损失使用的预训练 VGG 权重需事先可用或允许首次下载。

使用物理 GPU 1 时，在当前 PowerShell 进程设置：

```powershell
$env:CUDA_VISIBLE_DEVICES="1"
```

此时脚本内指定 `--device cuda:0`，对应被映射后的物理 GPU 1。

## 源域训练入口

当前实验使用 UIEB 弱蓝色色偏作为源域、强蓝色色偏作为目标域。数据目录为 `raw-890_4classes_severity/blue_tint` 和 `reference-890_4classes_severity/blue_tint`，下分 `weak`、`strong`。

新实验先设置独立目录，再准备清单：

```powershell
$exp = 'F:\qmh\DCTTA-uie\experiments\my_uie_experiment'

python train_source_uie.py --prepare_only --experiment_dir $exp
python train_source_uie.py --experiment_dir $exp --device cuda:0
```

默认路径和已确认的样本排除规则针对当前 UIEB 实验；其他数据需要显式指定 `--source_raw`、`--source_ref`、`--target_raw`、`--target_ref`，并调整预期数量和排除规则。用 `--help` 查看全部参数。不要为已有实验重复生成清单或覆盖结果。

源训练默认250轮，训练配置参考NU²Net，使用源验证集选择最佳权重；主要输出：

```text
$exp/manifests/                         # 训练、验证、目标 CSV 与数据核验
$exp/source/checkpoints/best.pth        # 源验证最佳权重
$exp/source/checkpoints/last.pth        # 最后一轮及续训状态
$exp/source/                           # 日志、训练与验证曲线等
```

续训时保持原实验及训练合同一致：

```powershell
python train_source_uie.py --experiment_dir $exp --device cuda:0 --resume "$exp\source\checkpoints\last.pth"
```

## 目标域适配入口

源训练及清单完成后：

```powershell
python adapt_target_uie.py --experiment_dir $exp --device cuda:0
```

若新实验需要同时运行源模型、八视图集成和DCTTA，再统一比较：

```powershell
python adapt_target_uie.py --experiment_dir $exp --device cuda:0 --run_all
```

已完成三组结果只需汇总时使用 `--summarize_only`，不要重复适配。输出目录已有结果时，入口会拒绝覆盖。

默认每张目标图适配一次，RDDM每张训练5步、采样5步，状态跨图累积。NU²Net适配裁剪以上限320为准，小图使用可用的8倍数尺寸；最终评价采用完整图。适配权重保存在 `$exp/dctta/adapted_model.pth`。

## 独立推理入口

源模型对文件夹内raw推理，不需要reference：

```powershell
python test_uie.py --mode baseline --model_path "$exp\source\checkpoints\best.pth" --lq_dir 'F:\path\to\raw' --result_dir 'F:\path\to\source_predictions' --device cuda:0
```

已适配模型推理：

```powershell
python test_uie.py --mode evaluate --model_path "$exp\dctta\adapted_model.pth" --lq_dir 'F:\path\to\raw' --result_dir 'F:\path\to\adapted_predictions' --device cuda:0
```

有reference时追加 `--gt_dir 'F:\path\to\reference'`。也可用 `--manifest` 指定已有清单；清单方式与 `--lq_dir/--gt_dir` 方式互斥。源模型八视图推理将 `--mode baseline` 改成 `--mode x8`。

## 诊断、对照与测试

| 入口 | 用途 | 已有运行说明 |
|---|---|---|
| `diagnose_uie.py` | 完整诊断、严格验收、数值对照 | [v2诊断说明](docs/03_20260910_11_strict_diagnostic/UIEB_BLUE_DIAGNOSTIC_V2_RUN.md) |
| `diagnose_update_uie.py` | 重放真实单步，分析Adam及Fisher作用 | [单步诊断说明](docs/04_20260912_update_diagnostic/UIEB_BLUE_UPDATE_DIAGNOSTIC_RUN.md) |
| `diagnose_visual_uie.py` | G0 整图伪标签、再退化输入和学生输出诊断 | [整图诊断说明](docs/06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_DIAGNOSTIC_RUN.md) |
| `run_factorial_uie.py` | 三组路径对照；复用已有G0 | [三组对照说明](docs/05_20260912_13_factorial/UIEB_BLUE_FACTORIAL_RUN.md) |

这些诊断入口绑定既有实验资产及哈希，不是任意新数据直接可用的通用训练入口。固定第一轮Fisher的兼容记录具有明确实验身份，不能通过修改旧结果绕过检查。

在 `movetoUIE/` 下运行常规测试：

```powershell
python -m unittest discover -s tests
```

部分真实RDDM诊断测试需设置 `DCTTA_TEST_RDDM=1`，会生成较大的临时产物，常规测试默认跳过它们。

实验过程和当前结论见 [实验记录](docs/00_project/UIEB_BLUE_EXPERIMENT_LOG.md) 与 [路径对照分析](docs/05_20260912_13_factorial/UIEB_BLUE_FACTORIAL_ANALYSIS.md)。

## Git 管理范围

保留源码、依赖说明、测试、现有文档和说明图片。`.gitignore` 排除数据集、预训练与训练权重、实验结果、临时测试目录、缓存、日志、下载归档及本机编辑器配置。

忽略文件只影响版本管理，不删除本地数据。特别保留 `RDDM/datasets/`、`RDDM/experiments/` 中的源码，避免误用全局同名目录规则。克隆仓库后需另行准备数据和权重；诊断复现还需对应实验清单与历史产物。

## 原项目来源

本项目在 [DCTTA 原项目](https://github.com/tonia86/DCTTA) 基础上开展UIE迁移，原恢复骨干相关实现参考 [PromptIR](https://github.com/va1shn9v/PromptIR)。迁移实验、诊断干预与原论文方法的差异应在使用和报告时明确区分。
