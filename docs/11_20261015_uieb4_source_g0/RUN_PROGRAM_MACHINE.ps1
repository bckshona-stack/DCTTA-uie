param(
    [string]$ProjectDir = 'F:\qmh\DCTTA-uie',
    [string]$GpuUuid = 'GPU-53f5a782-8c8c-958b-006d-c8aad0f4d8a3'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $ProjectDir
$env:CUDA_VISIBLE_DEVICES = $GpuUuid
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$experiment = Join-Path $ProjectDir 'experiments\UIEB4_light_to_severe_NU2Net_v1'
$dataset = Join-Path $ProjectDir 'dataset\UIEB_reclassified_v2'
function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed ($LASTEXITCODE); later phases stopped." }
}
git rev-parse HEAD
git status --short
Invoke-CheckedPython -Arguments @('prepare_uieb4_experiment.py','--phase','prepare',
    '--dataset_dir',$dataset,'--experiment_dir',$experiment)
$sourceArguments = @('prepare_uieb4_experiment.py','--phase','train','--experiment_dir',$experiment,
    '--device','cuda:0','--epochs','250','--batch_size','16','--num_workers','2',
    '--patch_size','256','--basic_channel','64','--lr','0.001','--seed','2022')
$last = Join-Path $experiment 'source\checkpoints\last.pth'
if (Test-Path -LiteralPath $last) { $sourceArguments += @('--resume',$last) }
Invoke-CheckedPython -Arguments $sourceArguments
Invoke-CheckedPython -Arguments @('run_uieb4_g0.py','--phase','run','--experiment_dir',$experiment,
    '--device','cuda:0','--max_gb','10')
