# 物理再退化步数预实验：程序机运行说明

日期：2026-10-13。状态：本预实验已完成并回传，200步已获用户确认。以下为预实验历史运行说明；下一步使用 [G6/G7正式运行说明](UIEB_BLUE_PHYSICAL_TTA_RUN.md)，无需重跑预实验。

## 1. 这次运行做什么

入口为pilot_physical_uie.py。复用visual_v2已选定10张图，每张图分别在真正的适应图块（patch）和完整原图（full）上独立拟合物理参数。每个范围均从相同初始a/b和新Adam状态开始，源教师八视图候选固定。

观察点为0、5、20、50、100、200。每张图/范围执行一条到200步的轨迹，在中间保存状态；不是分别重新训练5、20等步数。两种gamma共用相同T/B，只有施加额外退化的强度不同。

此阶段不更新学生、不运行RDDM、不读取目标reference、不计算参考PSNR。输出用于检查拟合稳定性与生成输入有效性；不是G6/G7正式适应结果。

## 2. 同步代码

将交付包解压到F:\qmh\DCTTA-uie，保持目录结构。新增两个入口/模块、专项测试和本阶段文档；不替换已有uie_tta.py、NU2Net、RDDM或旧实验目录。运行依赖现有工程中的diagnose_visual_uie.py及源模型、诊断protocol、visual_v2选图文件。

包内PACKAGE_MANIFEST.json记录开发提交和新文件SHA256。解压不等同于程序机已切换Git分支；已有工程依赖身份由预检另行校验。

## 3. 新PowerShell窗口配置

```powershell
Set-Location F:\qmh\DCTTA-uie
conda activate resshift
nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.used --format=csv
```

确认当前空闲卡后选择UUID。下面UUID是此前核对的PCI 17卡，曾对应nvidia-smi GPU0、任务管理器GPU1；以这次查询的空闲情况为准。

```powershell
$env:CUDA_VISIBLE_DEVICES = "GPU-53f5a782-8c8c-958b-006d-c8aad0f4d8a3"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$exp = ".\experiments\UIEBblue_weaktostrong_NU2Net"
$diag = "${exp}_diagnostic_v2"
$visual = "${exp}_visual_v2"
$out = "${exp}_physical_pilot_v1"
```

UUID暴露一张卡后，Python内使用cuda:0。环境变量只影响这个窗口后续启动的程序。同一$out只运行一个进程；正在运行的其他实验不需要结束。

## 4. 预检、设备验收与运行

先运行预检：

```powershell
python pilot_physical_uie.py --phase preflight --experiment_dir $exp --diagnostic_dir $diag --visual_dir $visual --output_dir $out --device cuda:0
```

默认应显示760张诊断图片、约1.214 GB保守磁盘估计、strict_cuda。它核对源权重、固定选图、raw文件哈希、原裁剪、相关模型实现和Torch/CUDA数值环境。预检只创建新增目录中的manifest及选图副本。

可先单独执行实际设备验收：

```powershell
python pilot_physical_uie.py --phase accept --experiment_dir $exp --diagnostic_dir $diag --visual_dir $visual --output_dir $out --device cuda:0
```

验收检查已知合成关系的拟合误差下降、gamma恒等和混合关系、双线性插值等价性；并在选图中最大整图上执行源教师八视图及一次物理参数反向。acceptance.json记录结果和PyTorch显存分配峰值。

通过后运行：

```powershell
python pilot_physical_uie.py --phase run --experiment_dir $exp --diagnostic_dir $diag --visual_dir $visual --output_dir $out --device cuda:0
```

如跳过单独accept，run会自动先做设备验收。验收失败时停止，不静默关闭严格确定性、不自动缩图。

默认20个提交单元：10张图×patch/full。结束显示COMPLETE: 760 diagnostic images, 120 snapshot rows; no TTA run。760张图之外另有3张设备验收图和1张曲线图，总计764张PNG；parameter.pt共120份。

## 5. 输出布局

```text
physical_pilot_v1/
  manifest.json
  selected_samples.csv
  acceptance.json
  acceptance/                       # 最大图设备验收，3张RGB图
  progress.json
  status.json
  metrics.csv                       # 120行观察点指标
  fit_curves.png                    # patch/full重建L1曲线
  images/
    patch/<sample_id>/
      raw.png
      source_pseudo.png
      trace.csv                     # 0至200步共201行
      timing.json                   # 该单元推理/拟合/导出全过程耗时
      000000/ ... 000200/            # 六个观察点
        reconstruction.png          # J0*T+(1-T)*B，与raw比较
        x_sd_gamma05.png            # 在raw上增加0.5强度退化
        x_sd_gamma10.png            # 在raw上增加完整强度退化
        T_r.png T_g.png T_b.png      # 固定[0,1]灰度显示，不是RGB场景图
        parameters.pt               # a/b及配置、尺寸、初态与参数哈希
    full/<sample_id>/                # 同布局，独立整图拟合
```

trace共20×201=4020行。metrics记录重建L1/TV、T分位数和接近下界比例、B值、两强度生成图的颜色/边界/标准差比例与扰动指标。精确0/1与接近0/1分别记录；不能把PNG量化边界当成真实浮点截断。

patch保持原确定性裁剪，最大320×320；full保留原分辨率。两个范围分别拟合，不能把full结果当成真实适应图块参数。PNG固定量化、不自动调色；参数和指标按浮点计算。

## 6. 续跑和配置调整

中断后在同样环境、相同代码/参数下重新执行run。每个图/范围结束后原子提交；最多重新计算一个未提交的范围，从共同初态开始。完成后再次运行只校验并跳过。

代码、输入、配置、设备约定改变时拒绝续跑；需要调整时使用新的输出目录，例如physical_pilot_v2，不删除或覆盖旧目录。step列表和其它参数预检后即冻结。

可配置项：--steps、--grid_size、--t_min、--fit_lr、--tv_weight、--fit_seed、--scopes。默认32网格、T_min=.1、学习率.05、TV权重.01、种子23；这些是预实验起点，不是已验证最优参数。--raw_dir仅用于raw目录迁移，仍严格校验原文件哈希。

--max_gb默认10，表示新增目录磁盘预算，不表示显存/RAM上限。CPU模式标为cpu_debug，仅用于实现检查，不当作严格CUDA历史对照。

## 7. 回传与下一步

请回传整个新增$out目录，无需回传原权重或数据。先结合10图训练稳定性、T/B触界和生成信息保持判断统一K及拟合配置；不按目标参考PSNR挑K。

预实验结果分析另作Git提交。确认配置后，再实现并验收G6/G7正式学生TTA入口和运行说明。当前交付包不包含正式G6/G7运行入口，也不会自动开始133图适应。
