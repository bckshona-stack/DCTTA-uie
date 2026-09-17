# NU²Net-L1 源模型训练与 DCTTA 对接说明

> 归档说明（2026-09-17）：本文件保留所属阶段的历史内容和当时状态；当前进展请看 [文档导航](../README.md)。文中“待运行”等描述不代表今天的状态。

日期：2026-09-06

## 1. 目标与范围

先使用配对源域数据从头训练 NU²Net，仅使用 L1 损失，得到可靠的源模型；再将同一源权重用于 Source-only 评价，以及 DCTTA 学生、教师的初始化。

本说明依据当前本地 UnderwaterRanker 源码编写。当前项目路径为 `E:/Codex/UnderwaterRanker-master`。尚未检查用户实际 DCTTA 工作目录，因此下文 DCTTA 内部路径均为**建议新增路径**，不是已确认存在的文件；对接执行者应先检查目标项目目录与导入方式，再落位，不能直接覆盖同名文件。

本文交付的是迁移规范，不代表已经完成复制、源模型训练或 DCTTA 运行验证。

## 2. 到底需要复制哪些文件

### 2.1 必须复制的内容

| 当前项目中的文件 | DCTTA 侧建议位置 | 处理方式 |
|---|---|---|
| `all_model/NU2Net/model.py` | `uie_backbones/nu2net.py` | 完整复制，第一版保留原网络、残差连接和 normalization tail |
| `LICENSE` | `third_party/UnderwaterRanker/LICENSE` | 保留来源许可证，不覆盖 DCTTA 自己的许可证 |
| 本文 `NU2Net_DCTTA_HANDOFF.md` | `docs/01_20260906_migration/NU2Net_DCTTA_HANDOFF.md` | 作为对接任务说明 |

**就运行 NU²Net 架构而言，只需要一个 Python 文件：`all_model/NU2Net/model.py`。** 它仅导入 `torch` 与 `torch.nn`，不依赖本项目的 `utils.py`、`loss.py`、数据集或 URanker。

新建 `uie_backbones/__init__.py`，内容可以为空，或只写：

```python
from .nu2net import NU2Net
```

不要复制原 `all_model/__init__.py`：它会同时导入 URanker，重新引入无关依赖。原模型文件里的 `if __name__ == "__main__":` 示例在正常导入时不会执行，可保留。`ChannelAttention` 虽有定义，但当前 NU²Net 没有调用它。

建议在第三方说明中记录原项目地址、论文题目、复制日期、源文件 SHA-256；如可获得 Git 提交号也一并记录。原 LICENSE 标注 `Attribution-NonCommercial 4.0 International`，应随第三方代码保留。

### 2.2 仅供参考，不作为运行依赖复制

| 文件 | 可借鉴的部分 | 为什么不直接接入 |
|---|---|---|
| `dataset/dataset.py` | raw/reference 同步裁剪与翻转 | 写死 UIEB 目录，同时包含排序数据集 |
| `options/NU2Net.yaml` | 通道数、Adam、学习率、patch 大小 | 默认开启 URanker，存在历史字段和路径问题 |
| `runner.py` | 基本训练、验证思路 | 混合两类 Runner，有验证与调度问题 |
| `utils.py` | 指标、构建函数的意图 | 含动态导入、直方图和历史兼容代码 |
| `uie_main_train.py`、`uie_main_test.py`、`uie_demo.py` | 入口用途 | 与原 Runner、配置、路径高度耦合 |
| `dataset/uie_train_list.txt`、`dataset/uie_test_list.txt` | 原 UIEB 800/90 划分 | 仅在明确采用该 UIEB 源域协议时复用 |
| `README.md` | 来源及引用信息 | 可以摘录引用，不必作为代码依赖 |

若要保留这些文件供比对，可放到独立参考目录，不能将其覆盖到 DCTTA 的同名 `utils.py`、`loss.py` 或 `dataset` 包中。

### 2.3 本任务不需要复制的内容

- `all_model/URanker/`、`options/URanker.yaml`、所有 `ranker_*.py`。
- `dataset/ranker_train_list.txt`、`dataset/ranker_test_list.txt`、URankerSet。
- `loss.py`：本任务的源训练直接使用 `torch.nn.L1Loss()`。
- 原 `environment.yaml`：不要覆盖 DCTTA 环境；模型本身只新增 PyTorch 接口需求，数据加载等依赖按目标环境检查。
- 官方 NU²Net、URanker 权重：当前方案是从头训练 NU²Net-L1，不能将官方权重误标为新训练的 L1 源模型。

注意：关闭的是**源训练阶段**的 URanker/VGG 附加损失，不是要求删除 DCTTA 原有的感知一致性损失。

## 3. 建议目标目录与新增工作

```text
<DCTTA_ROOT>/
  uie_backbones/
    __init__.py                    # 新建
    nu2net.py                      # 从当前项目复制
    nu2net_adapter.py              # 如需任意尺寸支持，则新增
  uie_data/
    paired_dataset.py              # 新增通用源域配对加载器
    target_dataset.py              # 新增或复用目标 raw 加载器
  configs/
    nu2net_source_l1.yaml           # 新增源训练配置
    nu2net_dctta.yaml               # 新增适应配置
  scripts/
    train_nu2net_source.py          # 新增源训练入口
    evaluate_nu2net.py              # 新增或改造统一评价入口
  manifests/
    source_train.csv
    source_val.csv
    target_eval.csv
  pretrain/
    nu2net_source_l1_best.pth       # 训练完成后生成
  docs/
    NU2Net_DCTTA_HANDOFF.md
  third_party/UnderwaterRanker/
    LICENSE
```

这些新增文件可以合并到 DCTTA 已有的模块体系中，不要求逐个照此命名。核心要求是：模型导入独立、源训练独立、数据边界明确、评价前处理一致。

如果选择在原项目或独立训练项目中完成源训练，DCTTA 侧就不需要迁移源训练入口和源训练数据；最终交付模型文件、训练好的权重、配置、数据划分与预处理说明即可。

## 4. 模型接口约定

```python
from uie_backbones.nu2net import NU2Net

model = NU2Net(basic_channel=64, is_residual=True, tail="norm")
y = model(x)
```

| 内容 | 约定 |
|---|---|
| 输入 | RGB 浮点张量 `[B, 3, H, W]`，正常范围 `[0,1]` |
| 输出 | 增强 RGB 张量，默认 tail 下范围 `[0,1]` |
| 输入尺寸 | 直接前向时 H、W 为8的倍数；常规 patch 使用256或320 |
| 网络输入 | 仅 raw，不需要 GT、直方图、任务标签或深度 |
| 输出类型 | Tensor，不是 PromptIR 包装器的字典或元组 |
| 源训练监督 | 输出与对应 reference 的 L1 |
| 适应输入 | 学生分别接收原图与再退化图，执行两次前向 |

三次下采样后的特征空间需要足够尺寸；不要用极小图块作为功能验证输入。第一版可用256×256进行前向与反向检查。

### 4.1 任意尺寸与前处理

推荐源验证、Source-only、DCTTA 教师/学生和最终评价统一使用右侧、下侧补齐至8倍数，输出后裁回原高宽。补齐模式可固定为 reflect；小尺寸不满足 reflect 条件时须定义统一回退规则。训练 patch 为256时通常无需补齐。

这属于相对原仓库“缩放至16倍数，再放大回原尺寸”的协议调整，必须记录。不要用两套尺寸处理方式比较适应前后收益。如果目标是严格复现原论文指标，则所有评价阶段统一保留原缩放协议。

模型适配器必须保留学生计算图，不能在统一 forward 上使用 `no_grad()`、`detach()` 或转 NumPy。只有教师伪标签与纯评价阶段关闭梯度。

### 4.2 normalization tail 与参数选择

- 第一版保留 `tail="norm"`，源训练与适应阶段保持一致。
- 不在模型外额外做无条件 min-max 归一化。
- 原 tail 的越界判断覆盖整个 batch，触发后才逐样本、逐通道归一化。比较 x8 逐张和打包前向时要检查这一影响；首版可固定逐张生成教师集成。
- 当前 InstanceNorm 使用默认 `affine=False`，没有可训练仿射参数。不能照搬“只更新归一化参数”的参数选择逻辑。
- TIPS 必须基于 NU²Net 的真实参数名、形状重新计算，不复用 PromptIR 的掩码。

## 5. 源模型训练规范

### 5.1 数据

使用显式配对清单，例如 CSV 字段：

```text
sample_id,raw_path,reference_path,scene_id
```

路径统一相对于一个明确的数据根目录解析。加载前检查文件存在、配对唯一、尺寸对应；用 `.convert("RGB")` 统一通道。按场景或清晰底图划分 source_train/source_val，不能只随机切分同底图生成的 patch 或不同退化版本。

源训练输出仅包含 raw/reference 及必要元信息。raw 与 reference 同步裁剪和翻转。第一版使用256×256同步裁剪；若沿用原 RandomResizedCrop 的 `(0.08,1.0)` 范围，必须在配置中记录，不能与固定随机裁剪混写。

目标 raw 不参与源训练；目标 reference 不参与任何训练、模型选择或适应早停。目标为 UIEB 时，不用其训练清单或官方 UIEB 源权重。具体源域/目标域数据集与显卡仍待用户确定。

### 5.2 训练设置

```yaml
# 新训练脚本的配置契约示例，不是原仓库入口可直接运行的配置
model:
  name: NU2Net
  basic_channel: 64
  is_residual: true
  tail: norm
train:
  loss: l1
  optimizer: Adam
  lr: 0.001
  batch_size: 16
  patch_size: 256
  epochs: 250
  scheduler: cosine
  scheduler_unit: optimizer_step
  seed: 2022
```

以上为首轮基线起点，不代表最优参数。batch 按显存确定，epochs 按源验证收敛情况决定。余弦调度总长度应等于实际计划的 optimizer 更新次数；若无梯度累积，可用 `epochs * len(train_loader)`。不要沿用原 `T_max=1000` 与每 batch 调度的混合设置。若使用梯度累积，按实际 optimizer.step 次数计算。

训练核心为：

```python
model.train()
pred = model(raw)
loss = torch.nn.functional.l1_loss(pred, reference)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
# 若按 optimizer step 调度，在这里调用 scheduler.step()
```

验证使用 `model.eval()` 与 `torch.no_grad()`，下一轮训练恢复 `model.train()`。日志累计用 `loss.item()`。保存 last 和源验证主指标最优的 best，不能根据目标指标选择源模型。

Windows 入口使用 `if __name__ == "__main__":`；先以 `num_workers=0` 完成冒烟检查。可复现实验还需固定 Python、NumPy、PyTorch 和 DataLoader worker 的随机性设置。

不要原样迁入原 Runner：其验证拒绝空 checkpoint、配置 `save_img/save_image` 不一致、恢复调度写死每轮50步等问题会一起迁入。

## 6. 权重交付与加载

### 6.1 建议保存格式

```python
checkpoint = {
    "net": model.state_dict(),  # 保存裸 NU2Net；不要保存适配器外壳的状态字典
    "model_config": {
        "basic_channel": 64,
        "is_residual": True,
        "tail": "norm",
    },
    "epoch": epoch,
    "global_step": global_step,
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "best_source_psnr": best_source_psnr,
}
```

另行保存完整配置、数据清单及哈希、随机种子、依赖版本、评价和前处理协议。`tail`、`is_residual` 等行为配置不能只靠 state_dict 恢复。

DCTTA 初始化只加载网络参数，不继承源训练的 Adam 状态；DCTTA 适应优化器应重新创建。恢复源训练时，才恢复其 optimizer/scheduler/global_step。

### 6.2 加载示例

```python
import torch
from uie_backbones.nu2net import NU2Net

# 读取本次训练产生的可信 checkpoint；加载参数按目标 PyTorch 版本核对。
ckpt = torch.load(checkpoint_path, map_location="cpu")
source = NU2Net(**ckpt["model_config"])
source.load_state_dict(ckpt["net"], strict=True)
source = source.to(device).eval()
```

不使用 `strict=False` 掩盖键名或形状不匹配。若使用 DataParallel，保存 `model.module.state_dict()`；若已有权重含 `module.` 前缀，仅在确认来源后统一去除并严格加载。若适配器包含 `backbone` 字段，先向裸模型加载，再包装，避免键名前缀差异。

必须交付的训练结果：`nu2net_source_l1_best.pth`、完整配置、源训练/验证清单、源验证结果、统一前处理说明。仅有模型源码还不能开始有效的 DCTTA 适应实验。

## 7. DCTTA 侧接入工作

对接执行者先搜索实际项目中的恢复网络构建、checkpoint 加载、教师复制、x8、参数重要性、optimizer、EMA 和评价调用位置；不要仅替换某个 PromptIR 类名就认为完成适配。

1. 在模型工厂增加 NU²Net 选项，加载第6节源权重。
2. 保留独立 source 基线，深复制 student/teacher，验证三者初始输出一致。
3. 学生设置允许梯度；教师伪标签阶段关闭梯度。
4. 将所有恢复调用统一为 Tensor 输入、Tensor 输出，消除 PromptIR 包装层特有返回值处理。
5. x8 对每个输出做逆几何变换后平均；伪标签停止梯度，固定尺寸处理与 batch 策略。
6. 保留 DCTTA 的目标退化学习、再退化生成及一致性机制；本次不额外引入 URanker。
7. 检查 RDDM 与增强骨干间的数值范围。若扩散模块内部使用另一范围，只在明确接口转换，不把 NU²Net 输入直接改成该内部范围。
8. 使用 NU²Net 参数重新计算 TIPS。若重要性计算需要教师参数梯度，应在独立重要性估计阶段临时启用，不能因教师常规冻结而得到全零梯度；计算后恢复教师状态。
9. 若按元素冻结，验证被冻结元素在 optimizer 更新后仍未变化；考虑 Adam 动量、权重衰减和优化器历史状态。每次独立适应实验从新源模型和新优化器开始。
10. EMA 按参数对应关系更新，核对 buffers 的处理，不引用 PromptIR 固定参数名。
11. 每个目标域实验从同一源 checkpoint 重新开始；不无意沿用上一目标域已经适应的模型。

学生的两次前向：

```python
y_in = student(x_in)
y_sd = student(x_sd)
# 与教师伪标签计算目标项目定义的一致性与锚定损失。
```

RDDM 的训练顺序、采样输入、TIPS 阈值、损失权重和 EMA 系数，应以实际 DCTTA 版本及已确认实验协议为准。本文不把历史讨论里的简化表达当作已核对的目标代码实现。

## 8. 评价与验收

先完成下列检查，再开展长时间训练和适应：

- [ ] 在 DCTTA 环境中能独立导入 NU²Net，不导入 URanker/timm 相关排序网络。
- [ ] `[1,3,256,256]` 输入得到同形状有限值输出，L1 反向能产生有限且非全零的梯度。
- [ ] 任意尺寸适配器输出裁回原尺寸；源评价、教师、学生前处理一致。
- [ ] 小批配对数据过拟合检查显示训练损失下降，验证配对和优化路径工作正常。
- [ ] 训练保存后严格重载，在相同输入与模式下输出一致；恢复训练能正确衔接调度步数。
- [ ] source/student/teacher 初始化输出一致，学生更新不会直接修改 source 或 teacher 的共享存储。
- [ ] TIPS 的实际冻结/更新元素与预期一致；EMA 按预期改变教师。
- [ ] 目标参考图没有进入适应损失、checkpoint 选择或超参数搜索。
- [ ] PSNR/SSIM 统一 RGB、data_range=1.0、尺寸及边界策略；SSIM API 按环境版本适配。
- [ ] 报告 Source-only、Source-only+x8、DCTTA 的同协议结果；区分集成收益与适应收益。

源模型是否可靠，需要完整源验证结果和训练曲线支持；仅通过随机张量前向不代表训练已验证。

## 9. 可交给 DCTTA 项目执行者的任务

请先阅读本文件和目标项目自己的 AGENTS.md。将 UnderwaterRanker 的 `all_model/NU2Net/model.py` 独立引入为 NU²Net 骨干，保留许可证与来源记录，不迁入 URanker 或原 Runner。建立仅用 L1 的独立源训练流程，明确源训练/验证清单，统一预处理、指标和 checkpoint 格式。先完成源模型及 Source-only 验证，再接入 DCTTA 的学生/教师、x8、RDDM、TIPS、EMA 与评价。所有结构性改动和实验超参数分别记录，不能利用目标参考图选择训练或适应设置。实施前检查 DCTTA 实际目录，本文建议路径不视为已有文件。数据集、数据路径、显卡和训练预算尚未确定，先完成不依赖这些信息的代码对接及功能验证。

## 10. 来源

- 原项目：[RQ-Wu/UnderwaterRanker](https://github.com/RQ-Wu/UnderwaterRanker)。
- 本地核查：`all_model/NU2Net/model.py`、`all_model/__init__.py`、`dataset/dataset.py`、`runner.py`、`utils.py`、`options/NU2Net.yaml`、`LICENSE`。
- 研究目标上下文：[DCTTA 迁移任务](codex://threads/019ff507-432c-7171-8eeb-afcd6118207a)。
