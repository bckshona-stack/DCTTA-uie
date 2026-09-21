"""G0 历史权重补推理：11阶段 y_in，全部步骤的固定10图学生/教师 PSNR。"""
import argparse
import csv
import os
from pathlib import Path

import diagnose_visual_uie as v


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def runtime():
    global torch, u, d
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    v.runtime()
    torch, u, d = v.torch, v.u, v.d
    torch.use_deterministic_algorithms(True, warn_only=False)
    u.seed_everything(23)


def prepare(args):
    roots = [args.experiment_dir, args.diagnostic_dir, args.visual_dir]
    v.independent_output(args.output_dir, roots)
    exp, diag, visual = map(Path, roots)
    protocol, rows = v.inventory(diag, exp)
    if protocol.get('diagnostic_protocol') != 'fixed_fisher_strict':
        raise ValueError('G0 strict v2 required')
    chosen = v.selection(visual/'selected_samples.csv', rows)
    old = v.read(visual/'run_manifest.json')
    stages = v.schedule(len(rows))
    if old['selected'] != chosen or old['snapshot_steps'] != stages:
        raise ValueError('Original visual selection/stages differ')
    for root, name in [(diag, 'run/status.json'), (diag, 'evaluation/status.json'),
                       (visual, 'status.json')]:
        if v.read(root/name)['status'] != 'complete':
            raise ValueError('Original experiment incomplete: '+str(root/name))
    if v.sha(visual/'selected_samples.csv') != old['selection_csv_sha256']:
        raise ValueError('Original selection changed')
    for name, digest in old['inputs'].items():
        # 只需要小权重；不读取完整 replay_states。
        if 'replay_states' not in name and v.sha(diag/name) != digest:
            raise ValueError('Original diagnostic input changed: '+name)
    status = v.read(visual/'status.json')
    if v.sha(visual/'metrics.csv') != status['files']['metrics.csv']:
        raise ValueError('Original visual metrics changed')
    d.verify_fixed_files(exp)
    groups, _, checkpoint = d.inputs(exp)
    samples = {s.sample_id: s for s in groups['target_eval']}
    if set(samples) != set(protocol['order']):
        raise ValueError('Target identities differ')
    env = d.environment()
    # GPU可见数量允许改变，便于用UUID选择另一张同型号卡。
    env_keys = ('packages', 'cuda', 'cudnn', 'numerical_settings', 'implementation')
    if any(env[k] != protocol['environment'][k] for k in env_keys):
        raise ValueError('Use original environment, numerical settings and model code')
    weights = {str(i): v.sha(diag/'run/snapshots'/('%06d.pth' % i))
               for i in range(len(rows)+1)}
    size = len(stages)*sum((3*r['width']+1)*r['height']*1.01+65536 for r in chosen)+20_000_000
    if size > args.max_gb*1e9:
        raise ValueError('Conservative disk estimate exceeds budget')
    out = v.Output(args.output_dir, int(args.max_gb*1e9))
    contract = dict(version=1, entry_sha256=v.sha(__file__),
        helper_sha256=v.sha(v.__file__), selected=chosen, stages=stages,
        steps=len(rows)+1, snapshots=weights, environment=env,
        device=args.device, max_bytes=out.limit,
        source_sha256=v.sha(checkpoint), protocol_sha256=v.sha(diag/'protocol.json'),
        visual_manifest_sha256=v.sha(visual/'run_manifest.json'),
        visual_metrics_sha256=v.sha(visual/'metrics.csv'),
        roots=[str(p.resolve()) for p in (exp, diag, visual)],
        expected_images=len(chosen)*len(stages), estimated_bytes=int(size),
        definition='Post-update checkpoint; whole-frame float32 eval; no adaptation/RDDM; teacher x8 and identity view')
    manifest = out.root/'manifest.json'
    if manifest.exists():
        if v.read(manifest) != contract:
            raise ValueError('Contract changed; use a new output directory')
    else:
        if list(out.root.iterdir()):
            raise ValueError('Output must be empty on first use')
        out.json('manifest.json', contract)
        out.put('selected_samples.csv', v.csv_bytes(chosen))
    return out, contract, samples, checkpoint


def measure(model, sample, source_psnr, out, step, save_image):
    raw = u.read_rgb(sample.raw_path).unsqueeze(0).to(next(model.parameters()).device)
    ref = u.read_rgb(sample.reference_path).unsqueeze(0)
    pred = v.student_raw_prediction(model, raw)
    result = dict(step=step, sample_id=sample.sample_id, source_only_psnr=source_psnr,
                  student_raw_psnr=v.psnr(pred, ref), student_raw_png_psnr='', path='')
    files = {}
    if save_image:
        name = sample.sample_id if sample.sample_id.lower().endswith('.png') else sample.sample_id+'.png'
        path = 'student_raw/%06d/%s' % (step, name)
        digest, quantized = v.export_image(out, path, pred)
        files[path] = digest
        result.update(path=path, student_raw_png_psnr=v.psnr(quantized, ref))
    return result, files


def verify_reference(actual, expected, label, tolerance=1e-5):
    if abs(actual-expected) > tolerance or not v.math.isfinite(actual):
        raise ValueError('%s differs: %.9f vs %.9f' % (label, actual, expected))


def execute(args):
    out, contract, samples, checkpoint = prepare(args)
    digest = v.sha(out.root/'manifest.json')
    progress_path = out.root/'progress.json'
    progress = v.read(progress_path) if progress_path.exists() else dict(
        manifest_sha256=digest, next_step=0, rows=[], files={})
    if progress['manifest_sha256'] != digest:
        raise ValueError('Resume identity differs')
    if (not 0 <= progress['next_step'] <= contract['steps'] or
            len(progress['rows']) != progress['next_step']*len(contract['selected']) or
            [(r['step'], r['sample_id']) for r in progress['rows']] !=
            [(s, item['sample_id']) for s in range(progress['next_step']) for item in contract['selected']]):
        raise ValueError('Resume coverage differs')
    v.verify_files(out, progress['files'])
    status_path = out.root/'status.json'
    if status_path.exists():
        status = v.read(status_path)
        if status['status'] != 'complete' or status['manifest_sha256'] != digest:
            raise ValueError('Completion identity differs')
        v.verify_files(out, status['files'])
        print('Already COMPLETE; no inference', flush=True)
        return
    model = u.build_backbone('nu2net', checkpoint, u.resolve_device(args.device, 0)).eval()
    model.requires_grad_(False)
    old_teacher = {(int(r['step']), r['sample_id']): float(r['psnr'])
                   for r in csv_read(Path(args.visual_dir)/'metrics.csv')
                   if r['kind']=='snapshot' and r['role']=='teacher_pseudo'}
    diag = Path(args.diagnostic_dir)
    for step in range(progress['next_step'], contract['steps']):
        path = diag/'run/snapshots'/('%06d.pth' % step)
        if v.sha(path) != contract['snapshots'][str(step)]:
            raise ValueError('Checkpoint changed during run')
        state = u.load_checkpoint(path)
        measured, files = [], {}
        old_student_path = diag/'evaluation'/('%06d_student.json' % step)
        old_student = {r['sample_id']: r['psnr'] for r in v.read(old_student_path)} if old_student_path.exists() else {}
        with torch.no_grad():
            model.load_state_dict(state['student'], strict=True)
            for item in contract['selected']:
                name = item['sample_id']
                row, images = measure(model, samples[name], item['source_psnr'], out, step, step in contract['stages'])
                if name in old_student:
                    verify_reference(row['student_raw_psnr'], old_student[name], 'Historical student '+name)
                row['historical_student_checked'] = name in old_student
                measured.append(row); files.update(images)
            model.load_state_dict(state['teacher'], strict=True)
            for row in measured:
                sample = samples[row['sample_id']]
                raw = u.read_rgb(sample.raw_path).unsqueeze(0).to(next(model.parameters()).device)
                ref = u.read_rgb(sample.reference_path).unsqueeze(0)
                identity_score = []
                def observer(event, views):
                    identity_score.append(v.psnr(views[''], ref))
                pseudo = u.eight_view_mean(model, raw, observer=observer)
                row['teacher_pseudo_psnr'] = v.psnr(pseudo, ref)
                row['teacher_raw_psnr'] = identity_score[0]
                key = (step, sample.sample_id)
                if step in contract['stages']:
                    verify_reference(row['teacher_pseudo_psnr'], old_teacher[key], 'Historical teacher '+sample.sample_id)
                row['historical_teacher_checked'] = step in contract['stages']
                row['student_minus_teacher_raw'] = row['student_raw_psnr']-row['teacher_raw_psnr']
                row['student_delta_source'] = row['student_raw_psnr']-row['source_only_psnr']
                del raw, ref, pseudo
        del state
        progress['rows'].extend(measured); progress['files'].update(files)
        progress['next_step'] = step+1
        out.json('progress.json', progress)
        print('G0 y_in: committed %d/%d; %.3f GB' % (step, contract['steps']-1, out.used()/1e9), flush=True)
    if len(progress['rows']) != contract['steps']*len(contract['selected']) or len(progress['files']) != contract['expected_images']:
        raise ValueError('Incomplete coverage')
    files = dict(progress['files'])
    files['metrics.csv'] = out.put('metrics.csv', v.csv_bytes(progress['rows']))
    summary = []
    for step in range(contract['steps']):
        rows = [r for r in progress['rows'] if r['step']==step]
        summary.append(dict(step=step, count=len(rows), **{
            key: sum(r[key] for r in rows)/len(rows) for key in
            ('student_raw_psnr', 'teacher_pseudo_psnr', 'teacher_raw_psnr', 'student_delta_source')}))
    files['summary.csv'] = out.put('summary.csv', v.csv_bytes(summary))
    v.verify_files(out, files)
    out.json('status.json', dict(status='complete', manifest_sha256=digest,
        images=contract['expected_images'], metric_rows=len(progress['rows']), files=files))
    print('COMPLETE: %d images; %d metric rows' % (contract['expected_images'], len(progress['rows'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment_dir', 'diagnostic_dir', 'visual_dir', 'output_dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--phase', choices=['preflight', 'run'], default='preflight')
    parser.add_argument('--max_gb', type=float, default=10.)
    args = parser.parse_args()
    if not 0 < args.max_gb <= 10:
        parser.error('--max_gb must be in (0,10]')
    runtime()
    if args.phase == 'run':
        execute(args)
    else:
        _, contract, _, _ = prepare(args)
        print('Preflight PASSED: %d checkpoints, %d images, <= %.3f GB disk' %
              (contract['steps'], contract['expected_images'], contract['estimated_bytes']/1e9), flush=True)


if __name__ == '__main__':
    main()
