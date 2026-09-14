# NU²Net / UIE 迁移使用说明

更新：2026-09-06。

## 定位与范围

本版本以根目录 DCTTA 原实现为算法参照，修改位于 `movetoUIE`。保留原始目标函数、梯度路径、在线 RDDM、Fisher/TIPS 和 EMA 更新顺序。NU²Net 原文件未改，许可证与来源摘要位于 `third_party/UnderwaterRanker/`。

这是保留核心机制的 UIE 迁移实现，不宣称逐项复现论文数值。源码与论文的 EMA、teacher 权重、Fisher 比例和感知损失定义等差异没有在此次适配中混改。独立源训练仍在源训练工程完成，DCTTA 不加载源训练数据、不继承源训练优化器。

## 当前入口与模块

- `uie_tta.py`：集中实现设备管理、骨干接口、数据加载、八视图、损失、学生更新、Fisher、EMA 与评价主流程。按八个中文分区排列，关键代码有中文注释。
- `options.py`：命令行参数及参数合法性校验，与原项目命名一致。
- `test_uie.py`：对应原 `test_promptir.py`，默认 Source-only，也支持 `--mode x8` 和 `--mode evaluate`。
- `train_source_uie.py`：源域配对监督训练、CSV 清单准备与断点续训。
- `adapt_target_uie.py`：目标域 DCTTA 适配，支持三组实验及汇总。原 `train_test_uie.py` 已移除。
- `test_promptir.py` / `train_test_promptir.py`：保留 PromptIR 的原命名入口，共用 `uie_tta.py`。
- `tta.py`：兼容之前的类与函数导入名称；旧 `run_uie.py` 已删除，改用上述两个 UIE 入口。
- `uie_backbones/nu2net.py`：用户提供的原始 NU²Net，未改动。
- `RDDM/net.py`：持续在线训练的包装器；底层扩散数学计算保持原实现。
- `requirements_uie.txt`：NU²Net 迁移路径的直接依赖；旧 `requirements.txt` 保留作历史环境参考。

阅读算法时，先看 `uie_tta.py` 的 `main()`，再看 `SRTTA.adapt()`、`compute_fisher()` 和 `fisher_restoration()`。尺寸适配器也已移入同一文件。`dctta/`、旧 `nu2net_adapter.py` 和 `run_uie.py` 已删除；主流程与测试直接使用集中实现。

整理只改变文件组织和注释，不改变参数、损失、训练/采样或状态更新逻辑。Fisher 身份校验改为跟踪集中后的 `uie_tta.py` 与 `options.py`，因此旧自动缓存会重新计算；显式指定旧缓存时，需要 `--recompute_fisher` 或换新路径。

NU²Net 路径不导入 PromptIR 的混合模型文件，也不依赖 URanker、clip、mmcv、Lightning、wandb、Augmentor 或 OpenCV。PromptIR 旧模型文件本身的额外依赖仍由原环境提供。

## 运行前准备

用程序机已有的 Python 环境运行，不要整包覆盖其 CUDA/PyTorch 安装。检查 PyTorch/torchvision 匹配，按需补充 `requirements_uie.txt` 中的缺项。测试过的依赖组合与范围见本文件后面的验证说明。

需要一份已充分训练的 NU²Net-L1 源 checkpoint，格式如下：

```python
{
    "net": bare_nu2net.state_dict(),
    "model_config": {
        "basic_channel": 64,
        "is_residual": True,
        "tail": "norm",
    },
    # 可以附带 epoch、源训练优化器等信息，但 DCTTA 不继承这些状态。
}
```

不能仅凭 state_dict 猜测 tail 或残差配置。权重缺键、多键或尺寸不匹配会明确失败，不使用 `strict=False` 掩盖。`model_config` 的三个字段应准确反映源模型。正式第一版使用 `basic_channel=64`，测试可用较窄模型检查接口。源权重的选择只能依据源验证集。

默认感知损失使用 ImageNet VGG19 权重，首次运行可能需要下载到 torch hub 缓存；离线程序机应提前准备。原 `pretrain/wavelet.mat` 仍然必需，没有删除低频一致性项。

## 数据清单

推荐 CSV，每行是一个目标样本：

```csv
sample_id,raw_path,reference_path
scene_001,raw/a.png,reference/a.png
scene_002,raw/b.png,reference/b.png
```

路径相对于 CSV 所在目录；也可通过 `--data_root` 指定根目录。允许 raw 与 reference 文件名不同；`sample_id` 必须唯一且不能包含目录分隔符。也支持 `--lq_dir` 与 `--gt_dir`，此时按完全相同的文件名配对，不分别排序后盲配。

没有 reference 时省略该列或全部留空。此时可以正常适应、保存图片，但报告明确显示不计算 PSNR/SSIM；本轮没有新增未验证的无参考指标。

raw/reference 在源代码接口上分离：Fisher 与适应只接收 `(sample_id, raw)`。评价可预检参考文件存在与尺寸，参考像素不会进入适应流程。目标 reference 不能用于选择最佳适应步数或超参数。

`--limit N` 先在清单顺序（目录输入则按文件名排序）选定同一组 N 张，再建立 Fisher、适应和评价数据；限制的是图像数量。Fisher 不再越过这一子集。适应默认在固定随机种子下打乱，评价保持清单顺序。

## 输入与随机性

- 输入为 RGB float `[0,1]`。NU²Net 原 normalization tail 保留，不额外做 min-max 归一化。
- NU²Net 评价补齐至8倍数，模型输出裁回原尺寸。PromptIR 评价保留原先对称裁至16倍数。
- Source-only、x8 和 DCTTA 最终评价始终使用同一种骨干预处理。NU²Net 新协议与其原仓库缩放评价不同，不直接声称复现原仓库指标。
- 默认适应 patch 为320；NU²Net 要求至少32且为8倍数，PromptIR 为16倍数。图像小于 patch 时明确报错，请显式选择较小 patch，不静默缩放或改变退化。
- 固定 `batch_size=1`，NU²Net 八视图必须逐次推理，避免 tail 的 batch 级判断改变结果。保持8种可逆几何变换，不对非方形图做裁切式旋转。
- 使用按样本 ID 和 seed 确定的随机裁剪，单独的 DataLoader 随机生成器。VGG 构建与 Fisher 阶段隔离随机状态，RDDM 单独用 `seed+1` 初始化。源模型和完整实验都记录种子。

这使新流程可重现，但不保证与旧版依赖隐式随机调用顺序的整次运行逐位相同。原脚本的 GT 随机裁剪、重复 teacher 随机变换和反复 VGG 构建不再扰动 RDDM。

## 可执行命令

以下命令在 `E:\Codex\DCTTA-main\movetoUIE` 下运行。将 `python` 换成程序机真实环境的 Python，将示例路径替换为实际文件。

```powershell
# 1. Source-only
python test_uie.py --model_path pretrain/nu2net_source_l1_best.pth --manifest manifests/target.csv --device cuda:0 --result_dir results/nu2net_source

# 2. 同一源模型，仅八视图集成
python test_uie.py --mode x8 --model_path pretrain/nu2net_source_l1_best.pth --manifest manifests/target.csv --device cuda:0 --result_dir results/nu2net_x8

# 3. 完整 DCTTA；默认启用 Fisher
python test_uie.py --mode adapt --model_path pretrain/nu2net_source_l1_best.pth --manifest manifests/target.csv --device cuda:0 --result_dir results/nu2net_dctta

# 4. 少量样本功能验证；Fisher、适应、评价都只用这2张
python test_uie.py --mode adapt --model_path pretrain/nu2net_source_l1_best.pth --manifest manifests/target.csv --device cuda:0 --limit 2 --result_dir results/nu2net_smoke

# 5. 已适应模型独立评价，不重新创建 RDDM/VGG/Fisher
python test_uie.py --mode evaluate --model_path results/nu2net_dctta/adapted_model.pth --manifest manifests/target.csv --device cuda:0 --result_dir results/nu2net_reloaded

# 6. 原 PromptIR 入口
python test_promptir.py --device cuda:0 --limit 1
python train_test_promptir.py --device cuda:0 --limit 1
```

运行后输出：增强图、`metrics.txt`、`run.json`。完整适应另外保存逐图损失 `adaptation.jsonl` 和 `adapted_model.pth`。指标按原 DCTTA 的 RGB 浮点 clip[0,1] 计算，未先量化为 PNG 再计算；输出图片则按原保存方式量化为8位。

`adapted_model.pth` 是用于部署/评价的权重，**不是完整在线恢复 checkpoint**。不要用它冒充新的独立源模型；恢复中断适应需要教师、RDDM、全部优化器及随机状态，本轮未实现该功能。独立目标域实验须从源权重重新启动。

`--compute_fisher 0` 是显式消融，不属于完整 DCTTA；报告会标注。`--verbose` 查看内部过程；`--save_rddm_results` 开启调试图片和 RDDM 权重保存，默认关闭。

## 核心参数和更新顺序

| 项目 | 首版设置 |
| --- | --- |
| 学生 Adam | lr=2e-4，betas=(0.9,0.999) |
| 每图学生更新 | iterations=1 |
| 两类损失 | 学生原图/再退化一致性 + 5 × teacher 约束 |
| 每类损失内部 | 像素 L1=1，VGG19 感知 L1=0.01，小波低频 L1=0.1 |
| confidence | 原确定性重复均值的等效值1，不使用新视图方差公式 |
| teacher EMA | 0.45 |
| Fisher | 单次目标 raw 重要性估计，逐张量 top 0.6，Adam 后恢复源权重 |
| NU²Net 候选参数 | 全部卷积权重/偏置，原 InstanceNorm 无仿射参数 |
| PromptIR 候选参数 | 原实际优化器的329个张量，不包含新增 prompt_param/temperature |
| RDDM | 原 dim64、残差预测单分支，每图5步、采样5步 |

每张图执行：逐视图 teacher 伪标签 → 持续使用的 RDDM 训练5步 → 以原 `x_in` 为采样条件得到 `x_sd` → 学生两次前向，`y_in` 停止梯度 → 损失反向/Adam → Fisher 恢复 → EMA teacher。

`iterations>1` 会在同一张图重复整套流程，每次也追加5步 RDDM；一个目标域内只创建一次 RDDM。原 RDDM 的 RAdam、内部 EMA、损失缩放、扩散系数重建与采样数学没有因迁移而替换。去掉的是无用导入、硬编码目录、设备隐式选择和无用中间拼接。

## Fisher 缓存

自动缓存名包含完整身份摘要。身份覆盖：源权重文件与实际状态、骨干配置、候选参数名称/形状/类型、Fisher 比例、目标 raw 文件内容、固定子集/裁剪/种子、空间协议、实现文件摘要和运行环境。

换目标数据、权重、参数策略或比例会生成不同缓存。若显式指定 `--fisher_cache_path`，缓存不匹配、为空、不完整或掩码非法时明确失败。确认要覆盖重算时使用 `--recompute_fisher`，或指定新路径。旧版只有 `fishers` 字段的缓存不会再被接受。

## 本次验证

- 20项检查全部通过（19项单元/对照检查，加1项真实 RDDM 连续两图检查）：NU²Net 未改动、原 tail 的 batch 耦合、真实默认宽度256输入反向、尺寸接口、逐视图变换、严格权重加载、参数集合、缓存拒绝、raw/reference 隔离、随机性、无参考评价、阻止误用已适应权重及 CLI 无导入副作用等。
- 通过根目录原 `tta.py` 提取的参考计算做 CPU 对照，仅将参考的 `.cuda()` 替换为 CPU 设备调用：比较 Fisher 与连续两步损失、梯度、Adam 后权重、恢复掩码和 EMA。使用窄 NU²Net 与固定测试特征损失/退化输出，以隔离核心更新逻辑。
- 当前生效 PromptIR 及依赖类 AST 一致；原优化器329个张量；仓库已有 `model.ckpt` 可严格加载。
- 真实 VGG19 特征结构使用固定随机权重，验证合批和分别前向的损失与预测梯度一致，目标分支无梯度。没有下载/运行 ImageNet VGG19 预训练权重。
- 真实 RDDM 在 CPU 连续处理两张32×32测试图：每张5步、优化器对象不变、参数连续变化、内部 EMA 步数5→10、输出有限且尺寸正确。
- 完整入口集成检查：baseline、x8、Fisher计算/加载、真实 RDDM 适应、保存、严格重载与评价全部通过；使用随机测试用窄 NU²Net、真实 wavelet 和替代的小型特征损失。重新计算/加载 Fisher 的适应后权重逐位一致。

验证环境为本机 Python 3.9 / PyTorch 2.8.0+cpu，不是程序机的 resshift/CUDA 环境。临时安装的测试依赖位于项目 `tmp/uie_test_deps`，没有替换现有 Python 环境。尚无训练好的 NU²Net-L1 源权重和正式目标清单，因此没有验证 UIE 指标提升、真实 VGG19 预训练权重下完整适应或 GPU 显存/速度。

在完整运行环境中可执行：

```powershell
python -m unittest discover -s tests -v
# 可选：真实 RDDM 连续两图检查
$env:DCTTA_TEST_RDDM="1"
python -m unittest discover -s tests -v
```

测试文件保留临时结果供检查，不递归删除目录。上述测试通过不代表源模型已训练或 UIE 效果已验证。
