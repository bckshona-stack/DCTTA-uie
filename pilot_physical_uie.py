"""只拟合物理退化参数的步数预实验；不更新学生，不读取目标reference。"""
import argparse
import csv
import importlib.metadata
import io
import math
import os
from pathlib import Path
import time

import diagnose_visual_uie as v


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def configure_runtime(device):
    global torch, u, p, Image, np
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
        raise ValueError('Unsupported CUBLAS workspace config')
    import torch
    import numpy as np
    from PIL import Image
    import uie_tta as u
    import physical_degradation as p
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    return u.resolve_device(device, 0)


def prepare(args):
    device = configure_runtime(args.device)
    config = p.FitConfig(grid_size=args.grid_size, t_min=args.t_min, lr=args.fit_lr,
                         tv_weight=args.tv_weight, seed=args.fit_seed)
    points = sorted(set(int(x) for x in args.steps.split(',')))
    if not points or points[0] != 0 or points[-1] > 10000 or any(x < 0 for x in points):
        raise ValueError('steps must include 0 and be in [0,10000]')
    scopes = args.scopes.split(',')
    if not scopes or len(set(scopes)) != len(scopes) or not set(scopes) <= {'patch', 'full'}:
        raise ValueError('scopes must be patch,full or one of them')
    roots = list(map(Path, (args.experiment_dir, args.diagnostic_dir, args.visual_dir)))
    v.independent_output(args.output_dir, roots)
    exp, diag, visual = roots
    protocol = v.read(diag/'protocol.json')
    if protocol['diagnostic_protocol'] != 'fixed_fisher_strict':
        raise ValueError('Original strict v2 required')
    if v.read(visual/'status.json')['status'] != 'complete':
        raise ValueError('Completed visual_v2 required')
    selected = csv_rows(visual/'selected_samples.csv')
    visual_plan = v.read(visual/'run_manifest.json')
    ids = [r['sample_id'] for r in selected]
    if (len(ids) != 10 or len(set(ids)) != 10 or ids != [r['sample_id'] for r in visual_plan['selected']] or
            v.sha(visual/'selected_samples.csv') != visual_plan['selection_csv_sha256']):
        raise ValueError('Use the original fixed 10 images/CSV')
    checkpoint = exp/'source/checkpoints/best.pth'
    if v.sha(checkpoint) != protocol['source_sha256']:
        raise ValueError('Source checkpoint changed')
    target_csv = exp/'manifests/target_eval.csv'
    audit = v.read(exp/'manifests/audit.json')
    if v.sha(target_csv) != audit['files']['target_eval.csv']:
        raise ValueError('Target manifest changed')
    raw_records = {r['sample_id']: r for r in csv_rows(target_csv)}
    if set(raw_records) != set(protocol['order']):
        raise ValueError('Target identities differ')
    # 此处只取raw字段，不检查、不打开、不传递reference路径。
    samples = []
    for name in ids:
        if any(c in name for c in '/\\:\x00') or name in ('.', '..'):
            raise ValueError('Unsafe sample_id')
        record = raw_records[name]
        path = Path(args.raw_dir)/name if args.raw_dir else Path(record['raw_path'])
        if v.sha(path) != record['raw_sha256']:
            raise ValueError('Raw image changed: '+name)
        samples.append(u.Sample(name, path.resolve(), None))
    data = u.TargetRawDataset(samples, protocol['config']['patch_size'], protocol['config']['seed'])
    originals = {r['id']: r for r in protocol['target']['samples']}
    for row in data.identity()['samples']:
        if row != originals[row['id']]:
            raise ValueError('Original crop identity differs')
    implementations = {
        name: v.sha(Path(__file__).parent/name) for name in
        ('pilot_physical_uie.py', 'physical_degradation.py', 'diagnose_visual_uie.py',
         'uie_tta.py', 'options.py', 'uie_backbones/nu2net.py')}
    for name in ('uie_tta.py', 'options.py', 'uie_backbones/nu2net.py'):
        historical = {k.replace('\\', '/'): x for k, x in protocol['environment']['implementation'].items()}
        if implementations[name] != historical[name]:
            raise ValueError('Original teacher/crop code changed: '+name)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'Pillow')}
    if device.type == 'cuda' and (any(versions[n] != protocol['environment']['packages'][n] for n in versions) or
                                   torch.version.cuda != protocol['environment']['cuda'] or
                                   torch.backends.cudnn.version() != protocol['environment']['cudnn']):
        raise ValueError('Use the original program-machine numerical environment')
    images = len(ids)*len(scopes)*(2+6*len(points))
    estimated = 20_000_000
    for crop in data.crops:
        for scope in scopes:
            h, w = (crop['height'], crop['width']) if scope == 'full' else (crop['patch_size'],)*2
            estimated += math.ceil(((3*w+1)*h*1.01+65536)*(2+6*len(points)))
            estimated += len(points)*(config.grid_size**2*3*8+65536)
    largest = max(data.crops, key=lambda x: x['height']*x['width'])
    estimated += math.ceil(3*((3*largest['width']+1)*largest['height']*1.01+65536))
    if estimated > args.max_gb*1e9:
        raise ValueError('Conservative storage estimate exceeds budget')
    manifest = dict(version=1, implementations=implementations, source_sha256=v.sha(checkpoint),
        protocol_sha256=v.sha(diag/'protocol.json'), selection_sha256=v.sha(visual/'selected_samples.csv'),
        target_manifest_sha256=v.sha(target_csv), samples=data.identity(), fit=config.identity(),
        points=points, scopes=scopes, expected_images=images, estimated_bytes=estimated,
        device=str(device), numerical_mode='strict_cuda' if device.type == 'cuda' else 'cpu_debug',
        environment=dict(packages=versions, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
            gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
        max_bytes=int(args.max_gb*1e9), roots=[str(x.resolve()) for x in roots],
        raw_paths=[str(s.raw_path) for s in samples],
        definition='Per-image reset; source teacher x8 frozen; no reference/no student updates; gamma=.5,1')
    out = v.Output(args.output_dir, manifest['max_bytes'])
    path = out.root/'manifest.json'
    if path.exists():
        if v.read(path) != manifest:
            raise ValueError('Run contract changed; use a new output directory')
    else:
        if list(out.root.iterdir()):
            raise ValueError('Output must be empty on first use')
        out.json('manifest.json', manifest)
        out.put('selected_samples.csv', (visual/'selected_samples.csv').read_bytes())
    print('Preflight PASSED: %d images; <= %.3f GB; %s' %
          (images, estimated/1e9, manifest['numerical_mode']), flush=True)
    return out, manifest, data, checkpoint, device


def image_bytes(value):
    pixels = (value.detach().cpu()[0].permute(1, 2, 0).clamp(0, 1)*255).round().to(torch.uint8).numpy()
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    return stream.getvalue()


def snapshot(out, prefix, raw, pseudo, model, trace_row):
    files = {}
    with torch.no_grad():
        t, b, _ = model.fields()
        _, _, _, reconstruction = model.objective(raw, pseudo)
        variants = {'reconstruction': reconstruction, 'x_sd_gamma05': model.render(raw, .5),
                    'x_sd_gamma10': model.render(raw, 1.)}
        for c, label in enumerate('rgb'):
            variants['T_'+label] = t[:, c:c+1].expand(-1, 3, -1, -1)
        row = dict(trace_row, initial_sha256=model.initial_hash, parameter_sha256=model.parameter_hash(),
                   t_min=float(t.min()), t_max=float(t.max()),
                   t_near_lower=float((t <= model.config.t_min+.01).float().mean()),
                   t_p01=float(torch.quantile(t.flatten(), .01)), t_p50=float(torch.quantile(t.flatten(), .5)),
                   t_p99=float(torch.quantile(t.flatten(), .99)),
                   b_r=float(b[0, 0, 0, 0]), b_g=float(b[0, 1, 0, 0]), b_b=float(b[0, 2, 0, 0]))
        raw_stats = p.channel_stats(raw)
        row.update({'raw_'+k: val for k, val in raw_stats.items()})
        for role in ('x_sd_gamma05', 'x_sd_gamma10'):
            stats = p.channel_stats(variants[role])
            row.update({role+'_'+k: val for k, val in stats.items()})
            for c in 'rgb':
                row[role+'_'+c+'_std_ratio'] = stats[c+'_std']/raw_stats[c+'_std'] if raw_stats[c+'_std'] else ''
            difference = variants[role]-raw
            row[role+'_perturbation_l1'] = float(difference.abs().mean())
            row[role+'_gradient_difference_l1'] = float(((difference[..., 1:, :]-difference[..., :-1, :]).abs().mean()+
                                                       (difference[..., :, 1:]-difference[..., :, :-1]).abs().mean())/2)
        for role, value in variants.items():
            name = prefix+'/'+role+'.png'
            files[name] = out.put(name, image_bytes(value))
        stream = io.BytesIO()
        torch.save(model.parameter_payload(), stream)
        name = prefix+'/parameters.pt'
        files[name] = out.put(name, stream.getvalue())
    return row, files


def fit_unit(out, manifest, teacher, raw, prefix):
    # 不读取reference；教师推理仅一次，随后只有a/b更新。
    begin = time.perf_counter()
    with torch.no_grad():
        pseudo = u.eight_view_mean(teacher, raw).detach()
    teacher_before = [x.detach().clone() for x in teacher.state_dict().values()]
    files = {}
    for role, value in [('raw', raw), ('source_pseudo', pseudo)]:
        name = prefix+'/'+role+'.png'
        files[name] = out.put(name, image_bytes(value))
    trace, metrics = [], []
    config = p.FitConfig(**manifest['fit'])
    for model, row in p.fit_steps(raw, pseudo, config, manifest['points'][-1]):
        trace.append(row)
        if row['step'] in manifest['points']:
            metric, images = snapshot(out, prefix+'/%06d'%row['step'], raw, pseudo, model, row)
            metrics.append(metric); files.update(images)
    if any(not torch.equal(a, b) for a, b in zip(teacher_before, teacher.state_dict().values())):
        raise RuntimeError('Pilot changed frozen teacher')
    if any(x.grad is not None for x in teacher.parameters()):
        raise RuntimeError('Pilot produced teacher gradients')
    name = prefix+'/trace.csv'
    files[name] = out.put(name, v.csv_bytes(trace))
    if raw.device.type == 'cuda':
        torch.cuda.synchronize(raw.device)
    elapsed = time.perf_counter()-begin
    name = prefix+'/timing.json'
    files[name] = out.json(name, dict(seconds=elapsed, fit_steps=manifest['points'][-1]))
    return metrics, files


def plots(out, rows, scopes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(scopes), figsize=(6*len(scopes), 4), squeeze=False)
    for ax, scope in zip(axes[0], scopes):
        for name in dict.fromkeys(r['sample_id'] for r in rows):
            chosen = [r for r in rows if r['scope']==scope and r['sample_id']==name]
            ax.plot([r['step'] for r in chosen], [r['reconstruction_l1'] for r in chosen], marker='.', label=name)
        ax.set(title=scope+' fit (no TTA)', xlabel='Physical fit steps', ylabel='Reconstruction L1')
        ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.tight_layout()
    stream = io.BytesIO(); fig.savefig(stream, format='png', dpi=150); plt.close(fig)
    return out.put('fit_curves.png', stream.getvalue())


def acceptance(out, manifest, data, teacher, device):
    """在实际设备上验收严格反向与最大整图；不预先确定正式K。"""
    path = out.root/'acceptance.json'
    identity = v.sha(out.root/'manifest.json')
    if path.exists():
        saved = v.read(path)
        if not saved['passed'] or saved['manifest_sha256'] != identity:
            raise ValueError('Acceptance identity differs')
        v.verify_files(out, saved['files'])
        return saved
    config = p.FitConfig(**manifest['fit'])
    generator = torch.Generator(device='cpu').manual_seed(73)
    pseudo = torch.rand(1, 3, 40, 48, generator=generator).to(device)
    background = pseudo.new_tensor([.15, .4, .7]).reshape(1, 3, 1, 1)
    raw = .7*pseudo+.3*background
    first, last = None, None
    for model, row in p.fit_steps(raw, pseudo, config, 100):
        if first is None: first = row['reconstruction_l1']
        last = row['reconstruction_l1']
    # 预实验数值配置必须至少能拟合一个已知简单合成关系。
    if not last < first*.5:
        raise RuntimeError('Synthetic fit did not reduce error sufficiently; review fit configuration')
    with torch.no_grad():
        if not torch.equal(model.render(raw, 0), raw):
            raise RuntimeError('gamma=0 is not identity')
        if not torch.allclose(model.render(raw, .5), .5*(raw+model.render(raw, 1)), atol=1e-6, rtol=1e-6):
            raise RuntimeError('Strength relation differs')
        t, _, grid = model.fields()
        expected = torch.nn.functional.interpolate(grid, size=raw.shape[-2:], mode='bilinear', align_corners=False)
        if not torch.allclose(t, (-expected).exp(), atol=2e-6, rtol=2e-6):
            raise RuntimeError('Deterministic interpolation differs')
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    i = max(range(len(data)), key=lambda j: data.crops[j]['height']*data.crops[j]['width'])
    sample = data.samples[i]
    raw = u.read_rgb(sample.raw_path).unsqueeze(0).to(device)
    with torch.no_grad():
        pseudo = u.eight_view_mean(teacher, raw)
    # 最大图前向+一次严格反向；正式步数仍按用户选定观察点。
    for model, _ in p.fit_steps(raw, pseudo, config, 1): pass
    files = {}
    with torch.no_grad():
        for role, value in [('source_pseudo', pseudo), ('x_sd_gamma05', model.render(raw, .5)),
                            ('x_sd_gamma10', model.render(raw, 1))]:
            p.check_image(value)
            name = 'acceptance/'+role+'.png'
            files[name] = out.put(name, image_bytes(value))
    record = dict(passed=True, manifest_sha256=identity, files=files,
                  synthetic_initial_l1=first, synthetic_final_l1=last,
                  largest_sample=sample.sample_id, largest_size=list(raw.shape[-2:]),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  interpolation='align_corners=False equivalent; matmul backward',
                  strict_deterministic=torch.are_deterministic_algorithms_enabled())
    out.json('acceptance.json', record)
    print('Device acceptance PASSED: synthetic fit, strengths, maximum full-frame forward/backward', flush=True)
    return record


def run(args):
    out, manifest, data, checkpoint, device = prepare(args)
    identity = v.sha(out.root/'manifest.json')
    progress_path = out.root/'progress.json'
    saved = v.read(progress_path) if progress_path.exists() else dict(manifest_sha256=identity, cursor=0, rows=[], files={})
    units = [(i, scope) for i in range(len(data)) for scope in manifest['scopes']]
    if (saved['manifest_sha256'] != identity or not 0 <= saved['cursor'] <= len(units) or
            len(saved['rows']) != saved['cursor']*len(manifest['points'])):
        raise ValueError('Resume identity/coverage differs')
    expected_rows = [(data.samples[i].sample_id, scope, step)
                     for i, scope in units[:saved['cursor']] for step in manifest['points']]
    if [(r['sample_id'], r['scope'], r['step']) for r in saved['rows']] != expected_rows:
        raise ValueError('Resume sample/scope/step identities differ')
    v.verify_files(out, saved['files'])
    status_path = out.root/'status.json'
    if status_path.exists():
        status = v.read(status_path)
        if status['status'] != 'complete' or status['manifest_sha256'] != identity:
            raise ValueError('Invalid completion identity')
        v.verify_files(out, status['files'])
        print('Already COMPLETE; skip inference/fitting', flush=True)
        return
    with u.isolated_rng():
        teacher = u.build_backbone('nu2net', checkpoint, device).eval()
    teacher.requires_grad_(False)
    accepted = acceptance(out, manifest, data, teacher, device)
    for cursor in range(saved['cursor'], len(units)):
        i, scope = units[cursor]
        sample = data.samples[i]
        if v.sha(sample.raw_path) != manifest['samples']['samples'][i]['raw_sha256']:
            raise ValueError('Raw changed during run')
        raw = data[i][1] if scope == 'patch' else u.read_rgb(sample.raw_path)
        raw = raw.unsqueeze(0).to(device)
        prefix = 'images/'+scope+'/'+sample.sample_id
        rows, files = fit_unit(out, manifest, teacher, raw, prefix)
        saved['rows'].extend(dict(sample_id=sample.sample_id, scope=scope, **r) for r in rows)
        saved['files'].update(files)
        saved['cursor'] = cursor+1
        out.json('progress.json', saved)
        print('Physical pilot committed %d/%d: %s %s; %.3f GB' %
              (cursor+1, len(units), sample.sample_id, scope, out.used()/1e9), flush=True)
        del raw
    if sum(name.endswith('.png') for name in saved['files']) != manifest['expected_images']:
        raise ValueError('PNG coverage differs')
    if len({r['initial_sha256'] for r in saved['rows']}) != 1:
        raise ValueError('Per-image initial parameters differ')
    files = dict(saved['files'])
    files.update(accepted['files'])
    files['acceptance.json'] = v.sha(out.root/'acceptance.json')
    files['metrics.csv'] = out.put('metrics.csv', v.csv_bytes(saved['rows']))
    files['fit_curves.png'] = plots(out, saved['rows'], manifest['scopes'])
    files['progress.json'] = v.sha(progress_path)
    v.verify_files(out, files)
    out.json('status.json', dict(status='complete', manifest_sha256=identity, files=files,
        images=manifest['expected_images'], metric_rows=len(saved['rows']),
        note='Pilot only; no reference PSNR/no student TTA; K must be frozen after reviewing pilot'))
    print('COMPLETE: %d diagnostic images, %d snapshot rows; no TTA run' %
          (manifest['expected_images'], len(saved['rows'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment_dir', 'diagnostic_dir', 'visual_dir', 'output_dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--raw_dir', help='Optional relocated raw folder; IDs/hashes remain fixed')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--phase', choices=['preflight', 'accept', 'run'], default='preflight')
    parser.add_argument('--steps', default='0,5,20,50,100,200')
    parser.add_argument('--scopes', default='patch,full')
    parser.add_argument('--grid_size', type=int, default=32)
    parser.add_argument('--t_min', type=float, default=.1)
    parser.add_argument('--fit_lr', type=float, default=.05)
    parser.add_argument('--tv_weight', type=float, default=.01)
    parser.add_argument('--fit_seed', type=int, default=23)
    parser.add_argument('--max_gb', type=float, default=10.)
    args = parser.parse_args()
    if not math.isfinite(args.max_gb) or not 0 < args.max_gb <= 10:
        parser.error('max_gb must be in (0,10]')
    if args.phase == 'preflight':
        prepare(args)
    elif args.phase == 'accept':
        out, manifest, data, checkpoint, device = prepare(args)
        with u.isolated_rng():
            teacher = u.build_backbone('nu2net', checkpoint, device).eval()
        teacher.requires_grad_(False)
        acceptance(out, manifest, data, teacher, device)
    else:
        run(args)


if __name__ == '__main__':
    main()
