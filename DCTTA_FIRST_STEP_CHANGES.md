# DCTTA 第一阶段运行适配记录

> 本文件保留历史记录。2026-09-06 已执行 NU²Net/UIE 迁移：当前入口、算法边界、缓存和测试协议请以 [UIE_MIGRATION.md](docs/UIE_MIGRATION.md) 为准。第二轮的视图方差 confidence 和 PromptIR 扩大参数集合不再作为默认行为；旧 Fisher 缓存不能直接复用。

更新时间：2026-08-20

## 目标

在当前 Windows 程序机上使用已有的 `resshift` Conda 环境，先打通 DCTTA 的基础推理入口和单图冒烟测试。第一阶段不改变 PromptIR 或 DCTTA 的网络结构，也不开始强退化实验。

## 环境检查

使用的 Python：

```text
D:\Anaconda\Installation\envs\resshift\python.exe
```

检查结果：

```text
Python 3.10.20
PyTorch 2.1.2+cu118
torch.version.cuda = 11.8
torch.cuda.is_available() = True
device_count = 2
GPU 0 = NVIDIA TITAN RTX, idle
GPU 1 = NVIDIA TITAN RTX, about 1 GB occupied
```

实际运行固定使用 `cuda:0`。GPU 1 虽然可以被 PyTorch 枚举，但当前有其他占用，不作为本项目设备。

## 安装的依赖

已安装到 `resshift` 环境：

```text
accelerate==0.14.0
ema-pytorch==0.8.3
lightning==2.1.3
pytorch-lightning==2.6.5
wandb==0.28.2
Augmentor==0.2.12
scikit-video==1.1.11
PyWavelets==1.8.0
```

原始 `requirements.txt` 是一个包含大量 Linux/CUDA 固定包的环境快照，不适合在 Windows 上直接整包安装。本阶段只安装了源码实际导入所需的缺失依赖。

`mmcv` 没有强行安装。Windows 下安装其编译版算子可能触发 C++/CUDA 编译；当前实际使用的后置 `PromptIR` 定义不调用 DCN 分支，因此采用了导入兼容处理，见下文。

## 代码修改

### `options.py`

修改内容：

- `--cuda` 默认值明确为 `0`；
- 新增 `--device`，支持显式传入 `cuda:0` 或 `cpu`；
- `--num_workers` 默认改为 `0`，作为 Windows 安全默认值；
- 新增 `--lq_dir`、`--gt_dir`、`--model_path`、`--result_dir`；
- 新增 `--limit`，用于单图或少量图片冒烟测试。

原因：原脚本虽然定义了若干路径参数，但实际主流程全部忽略并使用硬编码路径；同时默认 `num_workers=16` 不适合第一轮 Windows 调试。

### `test_promptir.py`

修改位置和内容：

- 删除 baseline 不需要的 TTA、RDDM、Lightning、wandb 初始化和导入；
- 删除无用的随机训练集/`train_loader` 创建；
- 删除 `CUDA_VISIBLE_DEVICES=4`；
- 使用仓库根目录推导默认数据、权重和结果路径；
- 使用 `opt.device` / `opt.cuda` 选择设备，并调用 `torch.cuda.set_device`；
- 所有张量使用 `.to(device)`；
- 使用 `Path.mkdir(..., parents=True, exist_ok=True)` 创建输出目录；
- checkpoint 同时兼容 Lightning 的 `checkpoint["state_dict"]` 和普通 state dict；
- 支持 `--limit`，结果文件名使用 `Path(...).stem`。

原因：baseline 原本会在正式前向之前初始化 TTA 和 wavelet，导致即使不做 TTA 也依赖 Linux 路径；同时硬编码 `/data2/...`、`cuda:4` 和 `mkdir -p` 在本机不可用。

### `train_test_promptir.py`

修改位置和内容：

- 删除 `CUDA_VISIBLE_DEVICES=4`；
- 数据、checkpoint、输出目录改为仓库相对默认值，并支持命令行覆盖；
- 使用显式 `cuda:0` 设备；
- 将 `.cuda()` 改为 `.to(device)`；
- 使用 Python 原生目录创建；
- 支持 `--limit`，便于先跑少量 TTA 图片。

原因：保留原始 TTA 流程，但消除 Windows 路径、设备和目录创建阻塞。

### `tta.py`

修改位置和内容：

- `SRTTA` 的设备从 `opt.device` / `opt.cuda` 推导；
- TTA 内部的输入张量从 `.cuda()` 改为 `.to(self.device)`。

原因：TTA 原来固定使用默认 CUDA 设备，无法和主脚本的设备选择保持一致。

### `utils/loss_utils.py`

修改位置和内容：

- `WaveletTransform` 不再使用作者机器的绝对路径；
- 默认从仓库的 `pretrain/wavelet.mat` 加载；
- 文件不存在时给出明确的 `FileNotFoundError`。

原因：原默认路径为 `/data2/tn/code/work_tta/DCTTA/pretrain/wavelet.mat`，在本机必然失效。

### `net/model.py`

修改位置和内容：

- 对 `mmcv.ops.modulated_deform_conv2d` 增加导入兼容分支；
- 当 `mmcv` 编译算子不可用时，提供普通卷积回退，使当前实际使用的后置 PromptIR 定义可以导入。

原因：`mmcv` 的 DCN 算子在本项目当前推理模型中未被调用，避免为了一个未使用的 Windows 编译扩展阻塞整个模型导入。若以后启用 `DCN_layer` 分支，需要单独安装并验证 mmcv 算子。

## 验证结果

已完成：

1. 所有修改后的 Python 文件通过 `py_compile`；
2. `test_promptir`、`train_test_promptir`、`tta`、`RDDM.net` 可以导入；
3. wavelet 文件可从仓库相对路径加载；
4. baseline 使用 GPU 0 完成 1 张 Rain100H 图片推理；
5. 输出已生成：

```text
results/first_step_baseline/norain-1.png
```

单图结果：

```text
PSNR = 22.792
SSIM = 0.7935
```

这只是单图冒烟结果，不能替代 100 张 Rain100H 的平均结果。

随后使用 `--limit 1 --compute_fisher 0` 完成了单图 DCTTA 测试：

```text
TTA 单图耗时约 15.8 秒
PSNR = 22.898
SSIM = 0.7953
```

TTA 结果已写入：

```text
results/first_step_tta/norain-1.png
```

## 当前可用命令

baseline 单图：

```powershell
D:\Anaconda\Installation\envs\resshift\python.exe test_promptir.py --device cuda:0 --cuda 0 --num_workers 0 --limit 1 --result_dir results\first_step_baseline
```

baseline 全量：

```powershell
D:\Anaconda\Installation\envs\resshift\python.exe test_promptir.py --device cuda:0 --cuda 0 --num_workers 0
```

TTA 第一轮建议仍然只跑少量图片，并关闭 Fisher：

```powershell
D:\Anaconda\Installation\envs\resshift\python.exe train_test_promptir.py --device cuda:0 --cuda 0 --num_workers 0 --limit 1 --compute_fisher 0
```

TTA 仍然包含每张图片的 RDDM 在线训练。当前实测约 15.8 秒/张，100 张预计约 26 分钟，实际全量运行前应先确认日志、显存和结果目录。

## 第二轮 TTA 优化记录

更新时间：2026-08-20

本轮保持 `online RDDM` 不变。也就是说，RDDM 仍然在每张图片上执行原来的 5 步在线训练和采样，没有把 `train()` 静默替换成 `sample()`，也没有删除退化一致性分支。

### `tta.py`

#### 单次八视图 teacher

原实现把同一组确定性的八路变换重复计算 4 次，并额外执行一个未使用的 `tmp` forward。现在新增 `_teacher_eight_view()`：

- 仍然使用原来的 8 个几何变换；
- 对非正方形图像按空间尺寸分成两组，通常为 4+4 两次 batch forward；
- 对正方形图像可以一次处理 8 路；
- 将每路输出逆变换后再求平均；
- 使用 8 路对齐结果的视图方差计算 confidence；
- 删除重复 4 次循环和无效 `tmp` forward；
- teacher 显式设置为 `eval()`。

这样保持八视图 teacher ensemble 和 confidence 机制，同时消除了确定性重复计算。由于 Rain100H 图片不是正方形，不能把旋转前后的 8 路强行拼成一个 batch，否则会发生尺寸不一致。

#### VGG 感知损失缓存

原 `compute_loss2()` 每次调用都重新构建 `PerceptualLoss/VGG19`，一次适配会调用两次。现在：

- `SRTTA.__init__()` 只创建一个 `self.perceptual_loss`；
- VGG 设置为 `eval()` 并冻结参数；
- `PerceptualLoss.forward()` 将预测和目标拼成一个 batch，只进行一次 VGG 特征提取，再拆分两部分计算损失。

梯度检查已通过，预测分支仍然可以正常反向传播。

#### 优化器参数收集

`collect_params()` 改为直接遍历 `model.named_parameters()`，收集所有 `requires_grad=True` 的参数。

修改前实测：

```text
requires_grad 参数：364
实际进入优化器：329
重复参数：0
```

修改后：

```text
requires_grad 参数：364
实际进入优化器：364
唯一参数：364
```

这修复的是参数遗漏问题，而不是重复参数问题；`prompt_param` 和 attention temperature 等非 `weight/bias` 参数现在也会被正确更新。

#### Fisher 缓存

新增：

- `SRTTA.save_fisher(path)`；
- `SRTTA.load_fisher(path)`；
- Fisher 参数名和形状校验；
- 修正 `compute_fisher()` 返回值；
- `fisher_restoration()` 改为按完整参数名恢复，不再只处理 `weight/bias`。

`train_test_promptir.py` 现在会：

1. 如果 Fisher 缓存存在，直接加载；
2. 否则计算 Fisher 并保存；
3. 默认缓存路径为 `pretrain/fisher_<checkpoint_name>.pt`；
4. 也可以通过 `--fisher_cache_path` 指定路径。

Fisher 缓存只适用于同一模型权重和相同的参数配置；更换 checkpoint 后应重新生成。

### `utils/loss_utils.py`

`PerceptualLoss.forward()` 改为单次 batch VGG 前向，保持目标分支 detach，预测分支梯度不变。

### `options.py`

新增：

```text
--fisher_cache_path
```

### 第二轮验证

优化前单图 TTA（`compute_fisher=0`）：

```text
耗时约 15.8 秒
PSNR = 22.898
SSIM = 0.7953
```

优化后单图 TTA：

```text
耗时约 6.2 秒
PSNR = 22.898
SSIM = 0.7937
```

结果文件：

```text
results/second_step_tta/norain-1.png
```

PSNR 保持一致，SSIM 的轻微变化来自 confidence 改为八视图实际方差以及新增可训练参数进入优化器，属于本轮修复的预期行为。RDDM 仍然执行在线训练，未被跳过。

第二轮推荐命令：

```powershell
D:\Anaconda\Installation\envs\resshift\python.exe train_test_promptir.py --device cuda:0 --cuda 0 --num_workers 0 --limit 1 --compute_fisher 0 --result_dir results\second_step_tta
```

## 第三轮指标报告记录

更新时间：2026-08-20

baseline 和 TTA 推理入口现在都会在结果图片目录下保存逐图指标报告。

### `utils/val_utils.py`

新增 `write_metrics_report()`，报告文件开头包含：

- 数据集名称；
- 实际评测图片数量；
- 平均 PSNR；
- 平均 SSIM；
- PSNR 最高和最低的图；
- SSIM 最高和最低的图。

报告后半部分按照推理顺序记录每一张图片的文件名、PSNR 和 SSIM。

### `test_promptir.py`

baseline 推理过程中收集每张图片的指标，并在推理结束后写入报告。

### `train_test_promptir.py`

TTA 后的评测过程使用相同格式写入报告。

### `options.py`

新增：

```text
--metrics_filename
```

默认文件名为：

```text
metrics.txt
```

可以通过命令行指定其他名称，例如：

```powershell
--metrics_filename Rain100H_PromptIR_metrics.txt
```

指标报告功能已使用单张 Rain100H 图片实际验证，输出格式和数值正确。

## 第四轮终端输出优化记录

更新时间：2026-08-20

### 默认输出调整

`train_test_promptir.py` 默认只保留以下信息：

- TTA 初始化摘要；
- 可训练/冻结参数数量；
- Fisher 是否启用、是否加载或保存缓存；
- 一个 TTA 总进度条；
- 一个 Eval 总进度条；
- 最终指标文件、平均 PSNR/SSIM 和完成状态。

以下内容默认不再输出：

- 每一个参数的 `train params` / `freezing params`；
- 每张图片单独打印的编号；
- 每张图片的 `loss_s` / `loss_t`；
- RDDM 内部 5 步训练进度；
- RDDM 采样进度；
- RDDM intermediate steps 数量；
- RDDM 中间图和 checkpoint 的保存输出。

### 新增详细模式

如果需要调试细节，可以在命令中增加：

```powershell
--verbose
```

该模式会重新显示 Fisher 进度、RDDM 内层进度和详细 loss。

如果确实需要保存 RDDM 中间图片和 checkpoint，可以额外使用：

```powershell
--save_rddm_results
```

默认不保存这些调试结果，避免产生不必要的文件和目录错误。

### 实测输出

单图 TTA 的默认终端输出已压缩为阶段摘要和两个主进度条，流程及指标保持正常：

```text
TTA setup: device=cuda:0, images=100, checkpoint=model.ckpt
TTA parameter configuration: trainable_tensors=364, frozen_tensors=184
Fisher: disabled
TTA: ...
TTA adaptation complete
Evaluation: starting
Eval: ...
Metrics: ...
Final: PSNR=..., SSIM=...
Done
```

## 尚未处理的事项

- TTA 的 RDDM 仍然是每张图片在线训练的原始实现，尚未做速度或显存优化；
- 还没有运行 100 张 Rain100H 的 baseline/TTA 全量复现；
- 还没有准备 denoise、dehaze 或更强 rain 的成对数据；
- `mmcv` 回退只保证当前 PromptIR 推理路径可导入，不等价于真正的 DCN 算子。
