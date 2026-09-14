"""目标域 DCTTA 入口；--run_all 从同一源权重运行三组，--summarize_only 核验后汇总。"""
import json
from pathlib import Path
import sys
import numpy as np

from options import parse_target_args, parse_args
from uie_tta import main as run, sha256_file, load_checkpoint, TargetRawDataset
from train_source_uie import checked_manifest, read_csv, write_csv, write_json


def summarize(folder):
    folder = Path(folder)
    labels = ('source_only', 'source_only_x8', 'dctta')
    reports = {label: json.loads((folder / label / 'run.json').read_text(encoding='utf-8')) for label in labels}
    identities = {label: json.loads((folder / label / 'experiment_identity.json').read_text(encoding='utf-8')) for label in labels}
    expected_modes = dict(zip(labels, ('baseline', 'x8', 'adapt')))
    reference = reports['source_only']
    for label, report in reports.items():
        if report['status'] != 'complete' or report['configuration']['mode'] != expected_modes[label]:
            raise ValueError(f'Incomplete/wrong experiment mode: {label}')
        if identities[label] != identities['source_only']:
            raise ValueError('Experiment checkpoint/target identity mismatch')
        for key in ('source_checkpoint_sha256', 'selected_samples', 'model_config', 'spatial_policy', 'metrics'):
            if report[key] != reference[key]:
                raise ValueError(f'Comparison mismatch: {key}')
        if report['source_checkpoint_sha256'] != identities[label]['source_checkpoint_sha256']:
            raise ValueError('Report checkpoint does not match experiment identity')
        for item in report['evaluation_records']:
            if not {'psnr', 'ssim'} <= item.keys():
                raise ValueError('Comparison requires complete reference metrics')
    indexed = {label: {r['image']: r for r in reports[label]['evaluation_records']} for label in labels}
    ids = list(indexed['source_only'])
    if not ids or any(set(indexed[label]) != set(ids) or len(indexed[label]) != len(reports[label]['evaluation_records']) for label in labels):
        raise ValueError('Missing or duplicate evaluation samples')
    rows = []
    for sample_id in ids:
        row = dict(sample_id=sample_id)
        for label in labels:
            row.update({label + '_' + metric: indexed[label][sample_id][metric] for metric in ('psnr', 'ssim')})
        for metric in ('psnr', 'ssim'):
            row['dctta_minus_source_' + metric] = row['dctta_' + metric] - row['source_only_' + metric]
            row['dctta_minus_x8_' + metric] = row['dctta_' + metric] - row['source_only_x8_' + metric]
        rows.append(row)
    output = folder / 'comparison'
    write_csv(output / 'per_image.csv', rows, list(rows[0]))
    means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'sample_id'}
    write_json(output / 'summary.json', dict(count=len(rows), means=means, identity=identities['source_only']))
    lines = ['# UIEB blue weak-to-strong comparison', '', f'Images: {len(rows)}', '',
             '| Method | PSNR | SSIM |', '|---|---:|---:|']
    for label in labels:
        lines.append(f'| {label} | {means[label + "_psnr"]:.4f} | {means[label + "_ssim"]:.6f} |')
    lines += ['', 'DCTTA is evaluated after adaptation on these same target raw images.',
              'Source checkpoint and target content identities were checked before comparison.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return means


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    wrapper, remaining = parse_target_args(argv)
    folder = Path(wrapper.experiment_dir).resolve()
    if '--help' in remaining or '-h' in remaining:
        print('Experiment options: --experiment_dir PATH --run_all --summarize_only\n'
              'Default: adapt from source/checkpoints/best.pth and manifests/target_eval.csv')
    opt = parse_args(remaining, default_backbone='nu2net', default_mode='adapt')
    if wrapper.summarize_only:
        return summarize(folder)
    if opt.backbone != 'nu2net' or opt.mode != 'adapt' or opt.lq_dir or opt.gt_dir or opt.manifest or opt.limit:
        raise ValueError('This experiment entry uses the audited full target manifest and NU2Net adapt mode; use test_uie.py for standalone evaluation')
    if opt.result_dir:
        raise ValueError('Use --experiment_dir to keep experiment outputs together')
    manifests = folder / 'manifests'
    audit = json.loads((manifests / 'audit.json').read_text(encoding='utf-8'))
    samples = checked_manifest(manifests, 'target_eval.csv', audit)
    TargetRawDataset(samples, opt.patch_size, opt.seed)  # 核验逐图裁剪，NU2Net 默认以 320 为上限。
    checkpoint = Path(opt.model_path) if opt.model_path else folder / 'source/checkpoints/best.pth'
    payload = load_checkpoint(checkpoint)
    if payload.get('artifact_type') != 'source_training':
        raise ValueError('Use this experiment source training checkpoint, not adapted or unknown weights')
    data = payload['training_contract']['data']
    if any(data[name] != audit['files'][name] for name in ('source_train.csv', 'source_val.csv')):
        raise ValueError('Source checkpoint was trained on different source manifests')
    target_ids = {Path(s.sample_id).stem.casefold() for s in samples}
    source_ids = {Path(s).stem.casefold() for s in payload['source_train_ids'] + payload['source_val_ids']}
    if target_ids & source_ids:
        raise ValueError('Source/target IDs overlap')
    identity = dict(source_checkpoint_sha256=sha256_file(checkpoint), target_manifest_sha256=audit['files']['target_eval.csv'])
    # 清单内包含 raw/ref 文件哈希，checked_manifest 已核验当前文件内容。
    common = remaining + ['--model_path', str(checkpoint), '--manifest', str(manifests / 'target_eval.csv'),
                          '--dataset_name', 'UIEB_blue_strong']
    modes = [('source_only', 'baseline'), ('source_only_x8', 'x8'), ('dctta', 'adapt')] if wrapper.run_all else [('dctta', 'adapt')]
    for label, mode in modes:
        output = folder / label
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Output exists: {output}; preserve it and choose a new experiment directory for reruns')
    for label, mode in modes:
        output = folder / label
        args = common + ['--mode', mode, '--result_dir', str(output)]
        if mode == 'adapt' and not opt.fisher_cache_path:
            args += ['--fisher_cache_path', str(output / 'fisher.pt')]
        run(args)
        write_json(output / 'experiment_identity.json', identity)
    return summarize(folder) if wrapper.run_all else json.loads((folder / 'dctta/run.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
