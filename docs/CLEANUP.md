# UIE 文件清理清单

本清单的路径均相对于 `movetoUIE/`。已检查两个 UIE 入口的 Python 导入链（包括可选 PromptIR 分支），并区分运行依赖、独立工具和历史附件。未执行任何文件或目录删除：用户规则要求批量删除由用户手动完成。

2026-09-06 复查：用户已删除 `dctta/`、`uie_backbones/nu2net_adapter.py` 和 `run_uie.py`；`tta.py` 以及第二组工具仍在。删后两个 UIE 入口与 20 项回归测试通过。下文保留原清理范围，供核对。

2026-09-07：源训练入口为 `train_source_uie.py`，目标适配入口为 `adapt_target_uie.py`；已移除单个旧入口 `train_test_uie.py`，下文涉及该旧名称的条目仅为历史记录。

## 第一组：可以手动删除的兼容壳

- `dctta/`：九个 Python 文件仅转发到 `uie_tta.py` 或 `options.py`，当前入口和正式测试没有导入此包。目录内缓存也不需要保留。
- `uie_backbones/nu2net_adapter.py`：尺寸接口已在 `uie_tta.py`。
- `tta.py`：转发旧接口，当前主流程没有引用。
- `run_uie.py`：通用旧入口，已由 `train_test_uie.py` 和 `test_uie.py` 覆盖。删除后旧命令需改用新入口。

## 第二组：UIE 主流程不使用的原项目工具

如果只维护当前 UIE 运行方式，可手动删除以下明确路径；它们的独立工具功能随之移除，原项目上一级目录仍保留原始参考实现。

- `RDDM/datasets/`
- `RDDM/eval/`
- `RDDM/experiments/`
- `RDDM/train.py`
- `RDDM/sample.py`
- `RDDM/src/denoising_diffusion_pytorch.py`（注意不是带 `residual_` 前缀的文件）
- `net/network_soda.py`
- `net/wat.py`
- `utils/dataset_utils.py`
- `utils/dataset_utils_tn.py`
- `utils/degradation_utils.py`
- `utils/image_io.py`
- `utils/image_utils.py`
- `utils/imresize.py`
- `utils/pytorch_ssim/`
- `utils/schedulers.py`
- `utils/utils_image.py`
- `utils/utils_tta.py`
- `utils/__init__.py.###`（非 Python 包入口）

## 必须保留

- `uie_tta.py`、`options.py`、`test_uie.py`、`train_test_uie.py`。
- `uie_backbones/nu2net.py` 及包入口；用户提供的原网络实现不改动。
- `RDDM/net.py`、`RDDM/src/residual_denoising_diffusion_pytorch.py`：在线退化生成器及其训练、采样实现。
- `utils/loss_utils.py`、`utils/val_utils.py`、`pretrain/wavelet.mat`：损失、评价和小波资源。
- `tests/`：保护八视图、损失梯度、Fisher、EMA 和 RDDM 跨图状态的回归测试。
- `third_party/UnderwaterRanker/`：来源及许可证信息。
- `net/model.py`、`net/arch_util.py`、`net/local_arch.py`、两个 PromptIR 入口：现有可选 PromptIR 对照仍使用；没有把支持分支误判成无用代码。
- 文档、权重、数据和实验结果：本轮不将其按“未被 import”判定为无用。`test_metrics.py` 是独立评价工具，同样保留。

各处 `__pycache__/` 和 `.idea/` 均非算法运行依赖，但只能由用户手动清理。不要整体删除 `pretrain/`、`RDDM/`、`utils/` 或 `net/`。

## 验证范围

清单依据静态导入检查；用户已完成的删除已通过上述复查，第二组文件删除后的状态尚未测试。删除兼容壳后，仅依赖旧导入路径的外部脚本需要同步更新。项目正式入口及 `tests/` 已使用集中实现。Fisher 缓存绑定实现文件内容，整理代码后显式指定的旧缓存需用 `--recompute_fisher` 重算。
