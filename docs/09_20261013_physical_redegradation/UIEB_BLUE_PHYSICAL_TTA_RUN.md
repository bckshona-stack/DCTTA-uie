# G6/G7物理再退化：正式程序机运行说明

日期：2026-10-13。用户已确认A方案：统一K=200；G6为γ=0.5，G7为γ=1。入口已实现；11项正式TTA本地CPU专项测试通过，程序机CUDA验收和正式133图实验待运行。

## 1. 冻结配置

每张实际训练图块重新初始化a/b与Adam，独立拟合200步；学生与学生Adam跨图累计。固定源教师提供八视图伪标签和loss_t监督，不做教师EMA。沿用原133张图顺序、确定性裁剪、原损失、学习率2e-4、teacher_weight=5和固定首轮Fisher。

拟合配置与已完成预实验相同：32×32三通道衰减网格，三个全局B，T_min=0.1，初始T=0.9，初态噪声0.05，拟合Adam学习率0.05、betas=(0.9,0.999)，TV权重0.01，局部种子23。正式入口不提供改变K或上述配置的CLI开关。

拟合I_hat=J0*T+(1-T)*B与raw的关系后，在raw上生成x_sd=raw*(1-γ*(1-T))+γ*(1-T)*B。学生监督对象与原DCTTA一致；不把I_hat当成x_sd。只使用当前图raw与固定源伪标签拟合，不使用reference训练或提前停止。

G6/G7独立从同一源学生初态开始；两组对应图片的T/B参数哈希必须完全一致，只改变γ。执行结束自动检查这一点。

## 2. 同步与Git

直接同步文件至程序机F:\qmh\DCTTA-uie，保持相对目录，同名文件覆盖。运行所需文件：

~~~text
run_physical_uie.py
physical_degradation.py
pilot_physical_uie.py
diagnose_visual_uie.py
run_strength_uie.py
run_factorial_uie.py
~~~

本方案与本运行说明保存在docs/09_20261013_physical_redegradation/，同步至程序机相同子目录。若要执行专项测试，再同步tests/test_physical_tta.py和tests/test_migration.py。

请在仓库根目录记录：

~~~powershell
Set-Location F:\qmh\DCTTA-uie
git rev-parse --show-toplevel
git rev-parse HEAD
git status --short
~~~

直接复制文件不会切换程序机Git分支。正式plan.json记录程序机提交、工作区状态以及实际源码SHA256；复制后有未提交改动时，HEAD不能单独代表正在运行的源码，以plan.json中的实际文件哈希为准。不要在运行中换代码或切换分支。

## 3. 新窗口选择显卡与目录

~~~powershell
Set-Location F:\qmh\DCTTA-uie
conda activate resshift
nvidia-smi --query-gpu=index,pci.bus_id,uuid,memory.used --format=csv
~~~

先确认空闲卡。以下UUID此前对应PCI 17、nvidia-smi GPU0、任务管理器GPU1；以本次空闲情况为准。限定可见UUID后，程序内部仍使用cuda:0。环境变量仅影响新窗口后续启动的程序，不影响其他已运行程序。

~~~powershell
$env:CUDA_VISIBLE_DEVICES = "GPU-53f5a782-8c8c-958b-006d-c8aad0f4d8a3"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$exp = "F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net"
$diag = "F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_diagnostic_v2"
$fac = "F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_factorial_v1"
$strength = "F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_strength_v1"
$visual = "F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_visual_v2"
$pilot = "F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_physical_pilot_v1"
$out = "F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_physical_v1"
$physicalArgs = @(
  "--experiment_dir", $exp,
  "--diagnostic_dir", $diag,
  "--factorial_dir", $fac,
  "--strength_dir", $strength,
  "--visual_dir", $visual,
  "--pilot_dir", $pilot,
  "--output_dir", $out,
  "--device", "cuda:0"
)
~~~

$out为新增独立目录，不是预实验目录。每个$out只允许一个运行进程；G6/G7依次运行，共享整图拟合缓存，不在同一目录同时启动两组。

## 4. 预检、验收、正式运行

~~~powershell
python run_physical_uie.py --phase preflight @physicalArgs
python run_physical_uie.py --phase accept @physicalArgs
python run_physical_uie.py --phase all @physicalArgs
~~~

preflight核对源权重、旧protocol/Fisher、原图片和裁剪、预实验完成状态及其文件哈希、冻结的拟合配置、G0—G5最终评价身份、当前代码和数值环境。首次预检在plan.json提交后、选图CSV提交前中断时，重试可按同一冻结内容补写缺失CSV；已有但内容不同的CSV会拒绝覆盖。输出总目录磁盘估算，不超过10 GB才继续。10 GB是磁盘预算，不是显存/RAM上限。

accept先用γ=0核对原G3第一步损失和学生权重，再对两组分别验证3张图的连续/恢复结果、观测隔离，并用10张典型图中最大的完整原图测试诊断路径。它执行真实学生更新验收，但不写入正式G6/G7学生初态。

all自动复核验收，随后依次运行G6、G7及G0—G7比较。可直接执行all，入口会先预检和验收。出现身份或数值错误时明确停止；不自动缩图、关闭严格模式或重新计算Fisher。

需要分阶段运行时：

~~~powershell
python run_physical_uie.py --phase run --group G6 @physicalArgs
python run_physical_uie.py --phase run --group G7 @physicalArgs
python run_physical_uie.py --phase compare @physicalArgs
~~~

每步显示G6/G7: committed N/133及累计磁盘大小。完成后可重复all：已经提交完成的组只校验并跳过，不重训。

## 5. 输出目录与图片数量

~~~text
UIEBblue_weaktostrong_NU2Net_physical_v1/
  plan.json                         # 冻结协议、代码/输入哈希、Git身份
  selected_samples.csv              # 原10张图
  acceptance.json
  acceptance/G6|G7/images/...        # 最大整图，合计8张PNG
  full_cache/<sample_id>.pth         # 10张完整图的浮点源候选及独立T/B，跨组共用
  G6/                               # G7布局相同
    progress.json
    resume.pth                      # 学生/Adam/Fisher/RNG完整续跑状态，无RDDM
    trajectory.json                 # 133步真实训练图块统计、损失、更新、耗时
    metrics.csv                     # 阶段快照与最终PNG/浮点PSNR
    evaluation/000000_student.json ... 000133_student.json
    images/
      teacher_pseudo/<stage>/<sample_id>.png
      x_sd/<stage>/<sample_id>.png
      y_sd/<stage>/<sample_id>.png
      y_in/<stage>/<sample_id>.png
    training/<step>/
      raw.png teacher_pseudo.png x_sd.png   # 真正参与训练的确定性图块
      parameters.pth                # 200步后的a/b、初态和最终参数SHA256
      fit_trace.csv                 # 0—200步，201行L1/TV/总损失/梯度范数
    final/y_in/<sample_id>.png       # 最终全133张整图学生增强结果
    status.json
  comparison/
    summary.json                    # G0—G7目标/源验证PSNR、SSIM及损害统计
    paired.csv                      # 逐图PSNR；G6/G7相对G5、彼此的差异
    status.json                     # 组间公共拟合一致性检查
~~~

阶段为0、14、27、40、54、67、80、94、107、120、133。每组固定10张图×11阶段×4种输出=440张；最终133张；实际训练图块133×3=399张。合计每组972张，两组合计1944张；加验收8张，总计1952张PNG。缓存没有额外PNG。

整图诊断独立拟合并缓存，保留原分辨率；真实适应仍是最大320×320确定性图块。两者T/B不同，不能用整图参数代替训练图块参数。固定教师、固定整图参数使同一组阶段teacher_pseudo和x_sd保持不变，y_sd/y_in随学生更新变化。

11阶段全量评价目标133张及源验证27张，其他步骤评价原固定探针。评价JSON有浮点PSNR/SSIM；metrics.csv同时记录浮点PSNR和保存后PNG的PSNR，x_sd的PSNR仅是退化输入诊断，不能解释为增强质量。

最终结果目录为$out\G6\final\y_in和$out\G7\final\y_in。

## 6. 耗时与续跑

trajectory.json的timing分开记录teacher_seconds、fit_seconds、fit_statistics_seconds、student_with_observer_seconds、training_export_seconds、observation_seconds。fit_seconds包含200步优化、逐步有限性/梯度检查、轨迹记录与最终生成，不包含教师推理、输入统计或PNG导出；student_with_observer_seconds包含原学生更新与只读更新摘要开销。不能与旧28.27秒完整诊断计时直接当作纯算法加速比。原子权重提交耗时不包括在这些分项中。

每张图更新与观测完成后原子提交resume.pth。resume中的cursor为已提交步骤的依据；若其提交后、progress.json写入前中断，再启动会按cursor修复进度文件，最后一步已经提交时不会重复训练。异常后保持相同代码、参数、设备环境，在同一新增$out重执行all或对应run。最多重算一个未提交的适应步骤；不中途继承单图退化Adam。已提交文件哈希不一致会拒绝续跑。

本机仅完成CPU实现检查；程序机上的正式源模型、VGG和CUDA组合以accept实际结果为准。最大图诊断按完整图处理，需要足够GPU空闲显存。

## 7. 回传与下一步Git节点

运行完成回传整个新增$out；旧输出、源权重和数据不必重复回传。先确认G6/G7/status.json与comparison/status.json为complete。

开发机已在codex/physical-redegradation分支记录实现提交。回传分析另作独立提交，完整输出/数据/权重留在Git之外。正式验收和结果分析通过后再考虑推送与审阅合并；当前没有推送或合并。

## 8. 代码复查后同步

本次复查修复入口的初始化与断点恢复边界。若已同步上一版，只需重新同步run_physical_uie.py与本运行说明；专项测试文件同步tests/test_physical_tta.py。无须重新同步其余5个运行辅助文件。

已有plan.json会绑定旧入口SHA256；更新代码后不得在旧协议下混合续跑。若已做过旧版预检，使用新的输出目录（例如将$out设置为F:\qmh\DCTTA-uie\experiments\UIEBblue_weaktostrong_NU2Net_physical_v2）重新预检和验收，保留旧目录。如果尚未创建正式plan，仍使用physical_v1。正在运行的进程不要中途替换其源码。

详细复查范围与证据见 [代码复查记录](UIEB_BLUE_PHYSICAL_TTA_CODE_REVIEW.md)。
