# G4/G5 程序机运行说明

日期：2026-09-17。入口：run_strength_uie.py。G4=EMA教师+α0.5；G5=固定源教师+α0.5。先G5、后G4，两组均从原第0步完整状态启动。正式实验在程序机运行，本地测试不代替程序机严格验收。

## 1. 同步文件与旧资产

程序机根目录仍为 F:\qmh\DCTTA-uie，环境仍为 resshift。新代码在本地分支 codex/redegradation-strength，正式运行前使用本次开发完成的提交。

也提供按相对路径打包的 G4_G5_program_machine_<commit>.zip，内含 SHA-256 文件清单；可解压后按下列清单同步到现有程序机根目录，不要新建一个缺少旧资产的空工程来运行。

需要同步/核对的文件：

- run_strength_uie.py：新增G4/G5入口。
- diagnose_visual_uie.py：上一轮已验证的整图诊断/OOM修复辅助代码，保持本次Git版本。
- run_factorial_uie.py：旧G1/G2/G3辅助函数，本轮未改动。
- docs/06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_SELECTED.csv：原10张选图CSV的新归档位置。
- docs/07_20260917_strength_control/：方案和本说明。
- tests/test_strength.py：本地验收测试，正式运行不依赖执行它。

若程序机使用Git，先检查其未提交更改，再同步并切到相同提交；不要强制覆盖程序机修改。本轮没有自动推送，远端没有更新前不能直接靠pull取得新代码。也可复制上述文件，但要记录本地实现提交，并让入口核对核心文件哈希。不要用本地CPU环境覆盖程序机CUDA环境。

原算法相关文件 diagnose_uie.py、uie_tta.py、options.py、RDDM、NU2Net、损失实现保持原strict v2版本。入口会比较环境/实现身份，不允许通过修改旧protocol或关闭检查绕过差异。

必须保留：

- 原实验：源best、dctta/fisher.pt、dctta/run.json、source_only/run.json、manifests及真实raw/reference。
- diagnostic_v2：protocol.json、run/status.json、run/trajectory.json、run/replay_states/000000.pth、acceptance/initial.pth、原evaluation完整结果。
- factorial_v1：plan.json；G1/G2/G3的run_status.json、trajectory.json、snapshots/000001.pth，以及各自evaluation目录。
- 若G1/resume.pth仍存在，会额外核查G5最终生成器完整状态与G1一致；不存在时明确只具备旧记录中的逐步通道均值核对，不声称逐像素生成图完全相同。新实验身份会记录该文件是否存在，预检后不要中途增删它。

不需要原visual_v2整套输出参与新运行，不重训源模型、不重算Fisher、不重跑完整G0–G3。

## 2. 设置路径

以下命令在同一个PowerShell窗口执行。GPU映射与原strict v2一致；示例使用此前的物理GPU 1。

~~~powershell
conda activate resshift
Set-Location F:\qmh\DCTTA-uie
$env:CUDA_VISIBLE_DEVICES="1"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
$exp = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net'
$diag = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_diagnostic_v2'
$fac = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_factorial_v1'
$out = 'F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_strength_v1'
$pick = 'docs/06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_SELECTED.csv'
$common = @('--experiment_dir', $exp, '--diagnostic_dir', $diag, '--factorial_dir', $fac, '--output_dir', $out, '--selection_csv', $pick, '--device', 'cuda:0', '--max_gb', '10')
~~~

$out必须是独立新目录，不在任何旧实验目录内部。已有本入口同身份结果可以续跑，不相关非空目录会被拒绝。α固定0.5，没有用于扫描的命令行参数。

## 3. 先预检，再验收

~~~powershell
python run_strength_uie.py @common --phase preflight
~~~

应显示G4/G5 preflight PASSED及整个新增目录的保守峰值估计。两组共有1366张正式PNG；验收另生成10张最大选图的诊断PNG。10 GB限制包含两组恢复状态、临时原子副本、图片、日志及元数据，不只是图片。小模型本地测试的体积不能代替正式预检。

~~~powershell
python run_strength_uie.py @common --phase accept
~~~

验收内容：G0–G3四个端点各核对原第1步；G4/G5分别进行22张前缀的无观测/有观测轨迹对照，包含大小crop与RDDM EMA预热边界；完整序列化恢复后继续更新；最大选图整图推理前后状态完全一致。验收不会覆盖旧结果，不是重新运行完整旧实验。

只有出现G4/G5 acceptance PASSED，才能进行正式运行。失败时保留输出和报错，不能跳过验收。验收中断后重复同一命令，会从头重做短程验收；已成功验收会核对文件后跳过。

## 4. 正式运行两组

~~~powershell
python run_strength_uie.py @common --phase run --group G5
~~~

上一条成功结束后再运行：

~~~powershell
python run_strength_uie.py @common --phase run --group G4
~~~

两组都完成后汇总：

~~~powershell
python run_strength_uie.py @common --phase compare
~~~

也可以使用下面命令按顺序完成验收、G5、G4和汇总；遇到异常会停止，不会继续下一组：

~~~powershell
python run_strength_uie.py @common --phase all
~~~

每步显示committed进度；完整组别状态见各自status.json。运行阶段已经完成评价和PNG保存，不需要额外evaluate阶段。整图采样在诊断中使用query分批全局注意力以避免旧36 GiB矩阵申请，不改变真实训练前向；其他激活仍占显存，正式显存适配由程序机验收确认。

## 5. 中断与恢复

重复对应group的run命令即可从最近完整resume.pth继续。每步适配和观测成功后原子提交；progress.json只是提示，恢复以resume.pth为准。中途未提交的同一步输出允许确定性重算覆盖，已提交文件会核对哈希。已完成组别不会重复适配。

不要更换α、CSV、环境或代码后继续用同一$out。入口身份变化会拒绝续跑；修订代码后应先核查原因，再使用新的版本目录，而不是改旧plan.json。

预算不足时停止，不自动缩图或批量删除。每组只保留最近一份完整状态；写入临时副本时也计入预算。本轮不保留每个历史阶段的大型权重，因此不能事后任意重放到旧阶段，只能用已有阶段图像/指标分析。

## 6. 输出和回传

~~~text
$out/
  plan.json                       # 输入/代码/环境身份、G4/G5定义、创建时Git状态
  selected_samples.csv
  storage_estimate.json
  acceptance.json                 # 程序机验收证据
  acceptance/                     # 验收图片、轻量完成标记及bootstrap
  G5/、G4/
    resume.pth                    # 最近完整续跑状态
    progress.json、status.json
    trajectory.json               # 每步损失、更新幅度、raw/x_sd/x_mix浮点统计
    metrics.csv                   # 图片浮点/PNG PSNR
    evaluation/                   # 每步固定观察集；11阶段全目标/源验证与教师伪标签
    images/
      teacher_pseudo/、x_sd/、x_mix/、y_mix/、student_raw/
        000000/ ... 000133/        # 原CSV的10张完整图
    final/student_raw/            # 最终133张真实raw增强结果
  comparison/
    summary.json、paired.csv
    probe_curves.csv、G0_G5_probe_psnr.png
    status.json
~~~

x_sd是未混合的RDDM输出，x_mix是α0.5混合结果，y_mix是学生在混合图上的输出；student_raw才是学生对真实raw的恢复。阶段图使用固定诊断种子重新采样，不是训练crop的放大。实际训练输入统计在trajectory.json，不能混用两者。

建议完整回传$out，保留其相对目录。包括resume.pth有利于后续核查；不要把$out提交到Git。以G4、G5及comparison三个status.json均complete为完成标志。

## 7. 本地验证范围

本地使用CPU、小型NU2Net及真实RDDM，感知/小波特征使用轻量测试替代。测试覆盖两项损失的输入/梯度路径、边界与预算保护、四个旧端点、完整状态恢复、观测隔离、完成跳过和文件篡改拒绝。六项专项测试覆盖上述内容和G0–G5汇总曲线；本地合成前缀缩至2张并人为跨RDDM EMA预热边界；正式程序机仍强制22张验收，不能将本地测试解释为正式效果或CUDA验收通过。
