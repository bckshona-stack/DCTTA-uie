# 程序机运行步骤

将本地 `movetoUIE` 的内容同步到程序机 `F:\qmh\DCTTA-uie`。正式数据、源权重和 GPU 实验不在本地自动创建。入口明确分为源训练和目标适配。

以下命令在该目录下运行；示例使用 PowerShell。确认当前 Python 环境已有兼容的 CUDA PyTorch/torchvision，再补齐 `requirements_uie.txt` 中的依赖（源训练曲线需要 matplotlib）。不要用本地 CPU 测试环境替换程序机 CUDA 环境。

```powershell
cd F:\qmh\DCTTA-uie
$exp = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net'
```

## 1. 只准备数据

```powershell
python train_source_uie.py --prepare_only --experiment_dir $exp
```

四个路径默认按用户提供的原文设置；如果实际目录不一致，程序明确报错，请显式提供 `--source_raw`、`--source_ref`、`--target_raw`、`--target_ref`，不会自动猜测目录层级。

默认预期 source=134 对、target=133 对、val_count=27。输出 `manifests/image_inventory.csv`（每图宽高和文件哈希）与 `preparation_report.json`（短边分布、重复情况）。原图不移动、不改名、不删除。

2026-09-08 已确认排除源域 `582_img_.png`、`648_img_.png`，默认配置自动按已核实的像素哈希从清单排除，记录在 `source_exclusions.json`。目录仍应有 134 对，所以不要修改 expected_source；有效源域预计 132 对（训练105/验证27），目标133对不变。同步 `options.py` 和 `train_source_uie.py` 后重跑准备命令即可；未知冲突仍会报错。详细经过见 [实验过程记录](UIEB_BLUE_EXPERIMENT_LOG.md)。

如果提示精确 raw 重复或 reference 冲突，查看 `manifests/exact_duplicate_report.json`。终端也会列出两张图片的 ID 与 reference 路径；报告包含全部检测到的重复对、reference 平均像素差及最大差值。此时尚未生成最终清单，不能直接开始训练。相同 raw 配不同 ref 不一定是匹配错误，须对照原始 UIEB 文件核对，不自动删除或任选一个 reference。

如果提示近重复候选待核对：打开 `manifests/near_duplicate_review.csv`，参照两个 sample_id 在原目录核对，每行 decision 填 `same` 或 `different`，然后重跑相同准备命令。`same` 会让同一源场景处于同一集合；源目标同场景会报错，必须先纠正数据划分。dHash 只能检出部分相似图，不替代人工场景检查。

已知同场景可额外提供 `--scene_groups scenes.csv`，CSV 列为 `domain,sample_id,scene_id`，domain 为 source/target。一个场景的所有成员填写同一 scene_id；不同场景必须用不同 ID。先确认场景再固定划分。最终产生三份 manifest 和 `audit.json` 后不允许覆盖重分；新划分使用新实验目录。

## 2. 源训练

```powershell
python train_source_uie.py --experiment_dir $exp --device cuda:0
```

默认 250 轮、batch=16、workers=2、lr=1e-3、256 输入。每轮覆盖所有训练样本（最后不足一个 batch 也保留），随机裁剪和翻转与 NU2Net 原实现一致。完整图片源验证，按平均 PSNR 保存 best。源码不会读取目标图像进行训练或选模型。

建议正式训练前在单独的小实验目录用少量源图检查运行与显存；或者在正式配置下用 `--stop_after_epoch 1` 完成第一轮后检查日志，再续训。这个选项不缩短 250 轮的学习率周期。

```powershell
python train_source_uie.py --experiment_dir $exp --device cuda:0 --stop_after_epoch 1
python train_source_uie.py --experiment_dir $exp --device cuda:0 --resume "$exp\source\checkpoints\last.pth"
```

第一条仅在尚未训练时使用。已有权重的目录必须显式 resume；代码、数据、总轮数、batch 和学习率等恢复契约不一致时拒绝恢复。若发生 OOM，显式降低 batch，并用新的实验训练输出重新开始；不要在续训时悄悄改 batch。保留 best.pth 与 last.pth，支持最近完整轮次边界续训，不支持半轮精确恢复。

输出：`source/logs/epochs.csv`、`source/logs/config.json`、四张 PNG 曲线、`source/checkpoints/best.pth` 和 `last.pth`。

## 3. 三组目标实验

```powershell
python adapt_target_uie.py --experiment_dir $exp --device cuda:0 --run_all
```

依次运行 Source-only、Source-only+x8、DCTTA，均从 `source/checkpoints/best.pth` 初始化。目标 ref 只在数据核验和最终评价时读取，不进入适配损失。目标 patch 上限默认 320，逐图使用 `8 * floor(min(320, H, W) / 8)`；小图仅降低裁剪尺寸，不缩放、不大幅填充、不丢弃。全部 133 张保留，最小 patch 为 152。`--patch_size` 可显式修改上限，本次实验保持默认 320。逐图裁剪尺寸和坐标写入 `dctta/run.json` 的 `target_identity.samples`。最终评价输入完整原图，输出裁回原尺寸。

本次源训练完成后，同步更新的 `uie_tta.py`、`options.py`、`adapt_target_uie.py` 即可开展目标实验；不需要重新准备清单或重训源模型。若物理 GPU 1 已通过 `$env:CUDA_VISIBLE_DEVICES="1"` 映射，命令仍使用 `--device cuda:0`。

分别输出 `source_only/`、`source_only_x8/`、`dctta/`，并自动生成 `comparison/per_image.csv`、`summary.json` 和 `summary.md`。它们记录同一源 checkpoint 与目标清单的身份，不能混用不同实验结果。

只运行目标适配可以省略 `--run_all`。已有输出目录不覆盖；失败或重跑应由用户保留/移动原结果后重试，程序不自动清理。DCTTA 的 adapted_model.pth 是评价权重，不是在线状态续训断点。

```powershell
python adapt_target_uie.py --experiment_dir $exp --summarize_only
```

## 4. 独立评价

```powershell
python test_uie.py --mode evaluate --model_path "$exp\dctta\adapted_model.pth" --manifest "$exp\manifests\target_eval.csv" --device cuda:0 --result_dir "$exp\reloaded_evaluation"
```

独立临时评价支持原有 test_uie.py 参数；正式三组实验使用 adapt_target_uie.py 的审计清单，不设置 limit 或替换目标集合。
