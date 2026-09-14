# UIEB 弱蓝色 → 强蓝色 NU2Net 两阶段实验方案

确认日期：2026-09-07。状态：已授权实施；正式数据划分、250 轮训练及 GPU 评价需在程序机运行。

## 目标与数据

弱蓝色色偏目录原有 134 对。2026-09-08 用户确认排除 `582_img_.png`、`648_img_.png` 两对同 raw 不同 ref 的样本，仅从清单排除，原文件保留；有效源域预计 132 对，固定种子 2022，预计划分为源训练 105 对、源验证 27 对。重复及同场景分组可能使数量略有变化。原图不移动，通过 CSV 管理。先检查配对、尺寸、重复、近重复场景以及源目标重叠，排除 `score_sorted` 副本。近重复检测只产生候选，不能代替人工场景核对。核查证据和处理经过见 [实验过程记录](UIEB_BLUE_EXPERIMENT_LOG.md)。

强蓝色色偏 133 对，全部 raw 用于 DCTTA；全部 ref 仅用于最终评价，不参与源训练、模型选择、适配损失、早停或调参。源域与目标域不能包含同一图片或确认的同场景近重复内容。

程序机代码目录：`F:\qmh\DCTTA-uie`。

数据路径（2026-09-08 修正：下划线是文件夹名称的一部分）：

```text
F:\qmh\DCTTA-uie\dataset\UIEB\raw-890_4classes_severity\blue_tint\weak
F:\qmh\DCTTA-uie\dataset\UIEB\reference-890_4classes_severity\blue_tint\weak
F:\qmh\DCTTA-uie\dataset\UIEB\raw-890_4classes_severity\blue_tint\strong
F:\qmh\DCTTA-uie\dataset\UIEB\reference-890_4classes_severity\blue_tint\strong
```

实验目录：`F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net`。

## 第一阶段：源域监督训练

- 单张 TITAN RTX，NU2Net 随机初始化，basic_channel=64、is_residual=True、tail=norm；不修改原网络。
- 仅 L1，无 URanker、额外 VGG 源训练损失或颜色退化增强。
- 采用 NU2Net 原数据处理：配对 RandomResizedCrop，面积 0.08～1.0、宽高比 3/4～4/3，双三次插值至 256×256；同步水平和垂直翻转，各 50%。无随机旋转。
- Adam，lr=1e-3，batch_size=16，num_workers=2，固定 250 轮，无早停。batch size 仅在显存不足时显式调整并记录。
- 余弦学习率按 optimizer step 更新，T_max=250×每轮实际批次数，覆盖整次训练。原配置 T_max=1000 配合逐 batch 更新，与本实验轮数不匹配，因此明确进行周期换算。
- 每轮在源验证集完整图片上验证。按平均 PSNR 保存 best.pth，同时记录验证 L1 和 SSIM。每轮保存 last.pth，第 250 轮结束后 last 即最终权重。
- 断点保存模型、配置、数据身份、优化器、调度器、轮次、最佳指标、历史日志和随机状态。恢复后继续完成总计 250 轮，不重启调度。
- 保存 CSV 日志和 PNG 曲线：训练/验证 L1、验证 PSNR、验证 SSIM、学习率；记录耗时与最佳轮次。

## 第二阶段：目标适配及三组评价

从同一个源 best.pth 独立开始：Source-only、Source-only+x8、DCTTA。三组使用同一目标清单，完整图片评价，输出逐图及平均 PSNR/SSIM、适配收益和增强图像。

DCTTA 保留教师八视图、在线 RDDM（每图五步及跨图状态）、原图学生停止梯度、再退化学生反传、L1/感知/小波一致性与教师约束，以及 Adam→Fisher源参数恢复→EMA 的顺序。目标阶段沿用 lr=2e-4、batch_size=1、teacher_weight=5、ema_decay=0.45、fisher_ratio=0.6，重新计算 NU2Net 的 Fisher 掩码。

目标 patch 上限默认 320。用户确认采用逐图裁剪：`8 * floor(min(320, H, W) / 8)`；大图裁 320，小图使用自身支持的最大 8 倍数，不缩放、不补大块边界、不丢弃样本。若显式提供 --patch_size，则替换公式中的上限；裁剪至少 32。固定种子和样本 ID 决定裁剪位置，逐图尺寸及坐标记录到 dctta/run.json 的 target_identity，亦绑定 Fisher 缓存。RDDM 在变尺寸图片间持续保留模型、优化器和 EMA。最终完整图像评价仍按 pad8 补齐并裁回原尺寸。

适配完成后评价最终学生，这属于在同一目标集上适配后评价，不声称是对未见目标样本的泛化测试。源训练断点恢复不等同于 DCTTA 在线状态恢复。

## 共用评价协议

源验证及三组目标评价均使用完整 RGB 图片、网络输入 [0,1]，右下补齐至 8 的倍数，输出裁回原尺寸。无随机评价增强，x8 组除外。有效原图区域计算指标，采用当前 utils/val_utils.py 的 RGB PSNR/SSIM 协议，不先量化保存再计算。

## 文件与输出组织

`train_source_uie.py` 集中实现数据准备、源训练、验证、曲线和恢复；`adapt_target_uie.py` 明确表示目标适配入口并支持三组运行及汇总；`test_uie.py` 保留独立评价；参数统一在 `options.py`；DCTTA 核心仍在 `uie_tta.py`。不重新拆出 dctta 包。

```text
experiments/UIEBblue_weaktostrong_NU2Net/
  manifests/       # source_train.csv、source_val.csv、target_eval.csv、检查报告
  source/
    checkpoints/  # best.pth、last.pth
    logs/         # 配置、CSV
    curves/       # PNG
  source_only/
  source_only_x8/
  dctta/
  comparison/
```

## 实施与验收

先完成准备清单和人工候选核对，再做小批训练、严格重载、断点续训一致性检查；正式训练满 250 轮后固定 best，运行三组目标实验。代码验证须覆盖同步增强、源目标隔离、同场景分组、源训练真实梯度与收敛冒烟、调度恢复、日志曲线和对照身份检查。原有 DCTTA 回归测试继续通过。

本地 CPU 测试只能证明软件流程；正式 GPU 显存、数据划分及增强效果以程序机结果为准。

## 实施结果（2026-09-07）

已新增源训练和目标适配入口，移除单个旧文件 `train_test_uie.py`，并提供 [程序机运行步骤](UIEB_BLUE_RUN.md)。NU2Net 原网络和 uie_tta.py 的 DCTTA 核心在本次实现中未修改。

26 项测试通过，覆盖原算法回归、配对增强、源目标隔离、近重复候选核对、固定清单校验、真实源训练、轮次级恢复、保存中断恢复、三组目标运行及汇总身份检查。另行完成 Windows num_workers=2 的恢复测试，通过连续训练/恢复训练逐轮指标及最终权重一致性检查。三个入口 --help 和 Python 静态语法检查通过，曲线 PNG 已生成并检查。

完整流程测试使用小型合成图像和窄通道 NU2Net；目标适配使用真实 RDDM 与小波模块，但预训练 VGG 由测试特征损失替代。正式 NU2Net 64 通道前后向另有既有回归测试覆盖。上述测试不表示已完成 UIEB 数据划分、正式源训练或 GPU 效果评价。
