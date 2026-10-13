# Git工作建议

## 当前推进记录（2026-10-13）

- 原开发分支codex/redegradation-strength已提交d487609：G4/G5分析及已完成G0补导出状态。
- 已提交786cb66：逐图独立物理再退化方案。
- 已从该状态建立codex/physical-redegradation。预实验实现、测试与运行说明单独提交，以最新git log为准。
- 预实验实现提交7eeb47c；回传分析提交8ff0b5c。用户已确认K=200，G6/G7正式入口、测试和运行说明作为下一独立实现提交。程序机CUDA验收和正式实验尚待执行。
- 正式运行前核对提交号、工作区与包内文件SHA256；experiments/、权重不加入Git。尚未推送或合并。

以下09-17内容为历史操作记录，不重复创建旧分支或重跑旧提交步骤。


日期：2026-09-17。仓库根目录：E:\Codex\DCTTA-main\movetoUIE。上级DCTTA-main也有.git，所有下面命令都必须先进入movetoUIE，避免在错误仓库提交。

## 已执行进展（2026-09-17）

用户授权后，已创建 codex/redegradation-strength 分支，已提交：

- 824c704：上一轮整图诊断入口和测试。
- 2713958：文档阶段归档和G4/G5方案。

下面“建议现在执行”的命令保留为操作记录，不要重复创建同名分支或重复提交。本轮G4/G5实现、专项测试与程序机运行说明另作开发提交，其身份以 git log -1 为准。尚未推送到远端或合并main。

## 开发开始前状态

检查时分支为main，最新提交为2bfbb6f（Initial-DCTTA-UIE-NU2Net-project）。上一轮diagnose_visual_uie.py、tests/test_visual_diagnostics.py及可视化诊断文档尚未提交，README也有未提交改动。本轮仅整理docs并新增实验方案；没有自动创建分支、暂存、提交或推送。

## 建议现在执行

先创建开发分支，已有工作区改动会保留在该分支工作区。下面是建议命令，不表示已经执行：

~~~powershell
Set-Location E:\Codex\DCTTA-main\movetoUIE
git rev-parse --show-toplevel
git status --short
git switch -c codex/redegradation-strength
~~~

将已有整图诊断实现单独提交，避免混入下一轮算法开发。先审阅文件，再暂存明确路径：

~~~powershell
git add -- diagnose_visual_uie.py tests/test_visual_diagnostics.py
git diff --cached --stat
git diff --cached
git commit -m "feat: add full-image G0 visual diagnostics"
~~~

然后提交本轮文档整理、已有分析报告和两组新方案。docs移动会在未暂存时显示旧路径D、新目录??；不是丢失文件，暂存后Git通常按内容识别重命名。

~~~powershell
git add -A -- docs
git add -- README.md
git diff --cached --stat
git diff --cached --summary
git diff --cached --check
git commit -m "docs: organize experiment stages and plan strength controls"
~~~

提交前检查暂存内容；若原先已有其他暂存更改，不要把它们顺带提交。这里不用git add .，避免把未来不相关文件带入提交。文档附件包括精选分析图/CSV/JSON，可以跟随分析报告提交；完整experiments、数据、权重继续由.gitignore排除。

## 后续开发节点

1. 在当前开发分支新增两组入口和必要测试，通过验收后单独提交，例如feat: add controlled redegradation strength experiments。
2. 正式程序机运行前记录git rev-parse HEAD与git status --short；相关源码应已提交且工作区干净，程序机使用同一提交和规定环境。身份哈希仍按实验协议核对，不能只凭commit认定数据正确。
3. 如要同步到远端，先核对git remote -v，再使用git push -u origin codex/redegradation-strength；本轮没有执行推送。
4. 回传后只提交分析报告和精选小型证据，另作结果分析提交；不把实验输出整体加入Git。
5. 验收完成后通过PR或审阅合并回main。不要在未保存工作时执行reset --hard或clean来处理文档移动。

本文件是操作建议，不把分支切换、提交或发布视为已完成。后续每个开发节点继续提示对应的Git操作。
