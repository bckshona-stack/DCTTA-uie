"""G4/G5 再退化强度对照：独立入口，原算法文件保持不变。"""
import argparse
import gc
import io
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import torch
import numpy as np
import diagnose_uie as d
import diagnose_visual_uie as v
import run_factorial_uie as f
import uie_tta as u

GROUPS = {'G4': {'fixed_teacher': False, 'alpha': 0.5},
          'G5': {'fixed_teacher': True, 'alpha': 0.5}}
RUN_ORDER = ('G5', 'G4')
SELECTION = Path(__file__).parent/'docs/06_20260914_15_visual_diagnostic/UIEB_BLUE_VISUAL_SELECTED.csv'


def mix_input(raw, generated, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('Invalid mixing strength')
    if raw.shape != generated.shape or raw.device != generated.device or raw.dtype != generated.dtype:
        raise ValueError('Mix inputs must have identical shape/device/dtype')
    for x in (raw, generated):
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError('Mix inputs must be finite in [0,1]; no extra clamp allowed')
    if alpha == 0:
        return raw.detach()
    if alpha == 1:
        return generated.detach()
    return ((1-alpha)*raw.detach()+alpha*generated.detach()).detach()


def channel_stats(value):
    x = value.detach().float().permute(1, 0, 2, 3).reshape(3, -1)
    return dict(mean=x.mean(1).cpu().tolist(), std=x.std(1, unbiased=False).cpu().tolist(),
                quantiles=torch.quantile(x, x.new_tensor([0., .5, .9, 1.]), dim=1).cpu().tolist(),
                exact_zero=(x == 0).float().mean(1).cpu().tolist(),
                exact_one=(x == 1).float().mean(1).cpu().tolist(),
                near_zero=(x <= 1/255).float().mean(1).cpu().tolist(),
                near_one=(x >= 254/255).float().mean(1).cpu().tolist())


class MixedGenerator:
    def __init__(self, generator, alpha, measure=True):
        self.generator, self.alpha, self.measure = generator, alpha, measure
        self.last = {}

    def train(self, raw, pseudo, names):
        generated = self.generator.train(raw, pseudo, names)
        mixed = mix_input(raw, generated, self.alpha)
        if self.measure:
            self.last = dict(raw=channel_stats(raw), x_sd=channel_stats(generated),
                             x_mix=channel_stats(mixed), x_sd_sha256=d.digest(generated.detach().cpu()))
        return mixed


class CountingStream:
    def __init__(self): self.size = 0
    def write(self, data): self.size += len(data); return len(data)
    def tell(self): return self.size
    def flush(self): pass


class BudgetOutput(v.Output):
    def checkpoint(self, name, value):
        # 先计算torch实际序列化字节数；不在内存中额外堆积整个序列化副本。
        counter = CountingStream()
        torch.save(value, counter)
        target = self.root/name
        if self.root not in target.resolve().parents:
            raise ValueError('Checkpoint escapes output directory')
        pending = target.with_name(target.name+'.pending')
        previous = pending.stat().st_size if pending.exists() else 0
        if self.used()-previous+counter.size > self.limit:
            raise RuntimeError('Checkpoint peak exceeds total output budget')
        if shutil.disk_usage(self.root).free < counter.size+1024*1024:
            raise RuntimeError('Insufficient checkpoint disk space')
        target.parent.mkdir(parents=True, exist_ok=True)
        # file object使用相同归档前缀，保证计数与实际序列化长度一致。
        with pending.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if pending.stat().st_size != counter.size:
            raise RuntimeError('Checkpoint serialization size changed')
        os.replace(pending, target)
        return v.sha(target)


def git_identity():
    def command(*args):
        try:
            result = subprocess.run(['git', '-C', str(Path(__file__).parent), *args],
                                    capture_output=True, text=True, timeout=15)
            return result.stdout.strip() if result.returncode == 0 else 'unavailable'
        except (OSError, subprocess.TimeoutExpired):
            return 'unavailable'
    return dict(commit=command('rev-parse', 'HEAD'), worktree=command('status', '--short'))


def verify_index(root, files):
    root = Path(root).resolve()
    for name, digest in files.items():
        path = (root/name).resolve()
        if root not in path.parents or v.sha(path) != digest:
            raise ValueError('Committed artifact changed: '+str(path))


def old_results(args, protocol):
    diag, fac = Path(args.diagnostic_dir), Path(args.factorial_dir)
    for phase in ('run', 'evaluation'):
        if v.read(diag/phase/'status.json')['status'] != 'complete':
            raise ValueError('Original G0 incomplete')
    old_plan = v.read(fac/'plan.json')
    if old_plan['entry_sha256'] != v.sha(f.__file__):
        raise ValueError('Keep original factorial implementation')
    if old_plan['protocol_sha256'] != v.sha(diag/'protocol.json'):
        raise ValueError('G1-G3 use another G0 protocol')
    if old_plan['initial_sha256'] != v.sha(diag/'run/replay_states/000000.pth'):
        raise ValueError('G1-G3 initial state differs')
    verify_index(diag/'evaluation', old_plan['g0_evaluation_sha256'])
    hashes = {'factorial_plan': v.sha(fac/'plan.json')}
    for group in ('G1', 'G2', 'G3'):
        root = fac/group
        if v.read(root/'run_status.json')['status'] != 'complete' or v.read(root/'evaluation/status.json')['status'] != 'complete':
            raise ValueError('Original factorial group incomplete: '+group)
        verify_index(root/'evaluation', v.read(root/'evaluation/index.json'))
        # 正式对照复用评价；端点验收另读取第1步学生/教师小快照。
        hashes[group] = {name: v.sha(root/name) for name in
                         ('trajectory.json', 'snapshots/000001.pth', 'evaluation/index.json')}
        if group == 'G1' and (root/'resume.pth').exists():
            hashes[group]['resume.pth'] = v.sha(root/'resume.pth')
    return hashes


def storage_estimate(rows, chosen, stages, initial):
    def png(r): return math.ceil((3*r['width']+1)*r['height']*1.01)+65536
    images_per_group = 5*len(chosen)*len(stages)+len(rows)
    png_bytes = 2*(5*len(stages)*sum(map(png, chosen))+sum(map(png, rows)))
    # 每组1个成熟状态，原子替换时还需1份；初态尚无Adam动量，保守按4倍张量计。
    mature = 4*d.state_bytes(initial)+32_000_000
    return dict(images_per_group=images_per_group, formal_images=2*images_per_group,
                acceptance_images=10, png_conservative_bytes=png_bytes+10*max(map(png, rows)),
                checkpoint_peak_bytes=3*mature, metadata_bootstrap_reserve_bytes=150_000_000,
                estimated_peak_bytes=png_bytes+10*max(map(png, rows))+3*mature+150_000_000)


def preflight(args):
    v.runtime()
    v.independent_output(args.output_dir, [args.experiment_dir, args.diagnostic_dir, args.factorial_dir])
    out = BudgetOutput(args.output_dir, int(args.max_gb*1e9))
    manifest_path = out.root/'plan.json'
    if not manifest_path.exists() and any(out.root.iterdir()):
        raise FileExistsError('Use an empty output directory for a new experiment')
    d.require_strict()
    diag = Path(args.diagnostic_dir)
    protocol, rows = v.inventory(diag, args.experiment_dir)
    if protocol.get('diagnostic_protocol') != 'fixed_fisher_strict' or protocol['environment'] != d.environment():
        raise ValueError('Keep original strict v2 environment and implementation')
    d.verify_fixed_files(args.experiment_dir)
    old_hashes = old_results(args, protocol)
    source_report = v.read(Path(args.experiment_dir)/'source_only/run.json')
    if source_report['source_checkpoint_sha256'] != v.sha(Path(args.experiment_dir)/'source/checkpoints/best.pth') or source_report['metrics'] != 'RGB float clip[0,1], data_range=1, skimage defaults':
        raise ValueError('source_only checkpoint/metric identity differs')
    groups, audit, checkpoint = d.inputs(args.experiment_dir)
    samples = {s.sample_id: s for s in groups['target_eval']}
    if set(samples) != set(protocol['order']): raise ValueError('Target identity differs')
    chosen = v.selection(args.selection_csv, rows)
    stages = v.schedule(len(rows))
    initial = u.load_checkpoint(diag/'run/replay_states/000000.pth')
    if d.digest(initial) != d.digest(u.load_checkpoint(diag/'acceptance/initial.pth')):
        raise ValueError('Original initial state differs from acceptance')
    size = storage_estimate(rows, chosen, stages, initial)
    del initial
    if size['estimated_peak_bytes'] > out.limit:
        raise RuntimeError('Conservative total estimate exceeds budget; do not start or resize images')
    paths = [Path(__file__), Path(f.__file__), Path(v.__file__)]
    value = dict(version=1, groups=GROUPS, order=protocol['order'], stages=stages, selected=chosen,
                 inputs=dict(protocol=v.sha(diag/'protocol.json'), initial=v.sha(diag/'run/replay_states/000000.pth'),
                             source=v.sha(checkpoint), source_only=v.sha(Path(args.experiment_dir)/'source_only/run.json'),
                             selection=v.sha(args.selection_csv), old_results=old_hashes),
                 implementations={p.name: v.sha(p) for p in paths}, environment=d.environment(),
                 experiment_dir=str(Path(args.experiment_dir).resolve()), diagnostic_dir=str(diag.resolve()),
                 factorial_dir=str(Path(args.factorial_dir).resolve()), device=args.device, max_bytes=out.limit,
                 estimate=size, diagnostic_attention=dict(chunk_size=128, threshold=1024),
                 metric='RGB float clip[0,1], skimage defaults; teacher=eight-view; student=single-view')
    if manifest_path.exists():
        saved = v.read(manifest_path)
        if saved['contract'] != value: raise ValueError('Experiment contract changed; use a new directory')
    else:
        out.json('plan.json', dict(contract=value, creation_git=git_identity()))
        out.put('selected_samples.csv', v.csv_bytes(chosen))
    if not (out.root/'selected_samples.csv').exists():
        out.put('selected_samples.csv', v.csv_bytes(chosen))
    if v.selection(out.root/'selected_samples.csv', rows) != chosen:
        raise ValueError('Committed selection changed')
    out.json('storage_estimate.json', size)
    d.require_space(out.root, max(0, size['estimated_peak_bytes']-out.used()))
    return out, protocol, groups, chosen, stages


def build(args, out, protocol, group):
    # 固定bootstrap路径，重启不累积重复Fisher副本。其最大空间已包含在预检中。
    folder = out.root/group/'bootstrap'
    if out.used()+50_000_000 > out.limit: raise RuntimeError('Bootstrap exceeds budget')
    t, g, data, order, groups, opt = d.build(args.experiment_dir, args.device, folder, fixed_fisher=True)
    if vars(opt) != protocol['config'] or order != protocol['order'] or data.identity() != protocol['target']:
        raise ValueError('Configuration/order/crops differ from G0')
    initial = u.load_checkpoint(Path(args.diagnostic_dir)/'run/replay_states/000000.pth')
    if d.digest(d.capture(t, g, 0, order)) != d.digest(initial): raise ValueError('Initial state differs')
    d.restore(t, g, initial)
    return t, g, data, order, opt, initial


def image_name(sample_id):
    return sample_id if sample_id.lower().endswith('.png') else sample_id+'.png'


def stage_images(t, g, samples, chosen, step, seed, out, prefix, alpha):
    files, metrics = {}, []
    for item in chosen:
        sample = samples[item['sample_id']]
        name = sample.sample_id
        raw = u.read_rgb(sample.raw_path).unsqueeze(0).to(t.device)
        pseudo = u.eight_view_mean(t.model_teacher, raw)
        x_sd = v.generate_full(g, raw, name, v.sample_seed(seed, name))
        mixed = mix_input(raw, x_sd, alpha)
        y_mix, y_raw = t.model(mixed), t.model(raw)
        ref = u.read_rgb(sample.reference_path).unsqueeze(0)
        for role, tensor in [('teacher_pseudo', pseudo), ('x_sd', x_sd), ('x_mix', mixed),
                             ('y_mix', y_mix), ('student_raw', y_raw)]:
            path = '%s/images/%s/%06d/%s' % (prefix, role, step, image_name(name))
            digest, quantized = v.export_image(out, path, tensor)
            files[path] = digest
            if role not in ('x_sd', 'x_mix'):
                score = v.psnr(tensor, ref)
                metrics.append(dict(step=step, sample_id=name, role=role, psnr=score,
                                    png_psnr=v.psnr(quantized, ref), source_only_psnr=item['source_psnr'],
                                    delta_source_psnr=score-item['source_psnr'], path=path))
    return files, metrics


def observe(t, g, groups, chosen, step, order, seed, out, group, protocol, stages):
    files, image_metrics = {}, []
    with v.isolated_diagnostic(t, g, step, order), torch.no_grad():
        samples = {s.sample_id: s for split in groups.values() for s in split}
        full = step in stages
        selected = groups['target_eval']+groups['source_val'] if full else [samples[n] for ids in protocol['probes'].values() for n in ids]
        student = d.score(t.model, selected)
        name = '%s/evaluation/%06d_student.json' % (group, step)
        files[name] = out.json(name, student)
        if full:
            teacher = d.score(t.model_teacher, groups['target_eval'], ensemble=True)
            name = '%s/evaluation/%06d_teacher_pseudo.json' % (group, step)
            files[name] = out.json(name, teacher)
            images, image_metrics = stage_images(t, g, samples, chosen, step, seed, out, group, GROUPS[group]['alpha'])
            files.update(images)
        if step == len(order):
            for sample in groups['target_eval']:
                raw = u.read_rgb(sample.raw_path).unsqueeze(0).to(t.device)
                pred = t.model(raw)
                name = group+'/final/student_raw/'+image_name(sample.sample_id)
                digest, quantized = v.export_image(out, name, pred)
                files[name] = digest
                ref = u.read_rgb(sample.reference_path).unsqueeze(0)
                image_metrics.append(dict(step=step, sample_id=sample.sample_id, role='final_student_raw',
                                          psnr=v.psnr(pred, ref), png_psnr=v.psnr(quantized, ref),
                                          source_only_psnr='', delta_source_psnr='', path=name))
    return files, image_metrics


def adapt_one(t, g, data, order, index, alpha, measure=True):
    by_id = {s.sample_id: i for i, s in enumerate(data.samples)}
    name, image = data[by_id[order[index]]]
    proxy = MixedGenerator(g, alpha, measure)
    observer = f.UpdateSummary(t) if measure else None
    t.observer = observer
    try: losses = t.adapt(image.unsqueeze(0), [name], proxy)
    finally: t.observer = None
    d.require_strict()
    state_hash = d.digest(d.capture(t, g, index+1, order))
    return dict(step=index+1, sample_id=name, losses=losses, state_sha256=state_hash,
                **(observer.values if observer else {}), inputs=proxy.last)


def acceptance(args):
    out, protocol, groups, chosen, stages = preflight(args)
    report = out.root/'acceptance.json'
    contract_hash = v.sha(out.root/'plan.json')
    if report.exists():
        saved = v.read(report)
        if saved['plan_sha256'] != contract_hash or not saved['passed']: raise ValueError('Acceptance identity differs')
        verify_index(out.root, saved['files'])
        print('G4/G5 acceptance already passed; SKIP', flush=True)
        return
    t, g, data, order, opt, initial = build(args, out, protocol, 'acceptance')
    count = d.acceptance_length(22, data, order, opt, g.trainer.ema, 5)
    update_teacher = t.update_teacher
    findings, files = {}, {}
    # 四个端点只核对第1步，不重新运行旧组完整实验。
    for old_group, fixed, alpha in [('G0', False, 1), ('G1', True, 1), ('G2', False, 0), ('G3', True, 0)]:
        d.restore(t, g, initial)
        t.update_teacher = (lambda: None) if fixed else update_teacher
        row = adapt_one(t, g, data, order, 0, alpha)
        if old_group == 'G0':
            expected = v.read(Path(args.diagnostic_dir)/'run/trajectory.json')[0]
            if row['losses'] != expected['losses'] or row['state_sha256'] != expected['state_sha256']:
                raise ValueError('G0 endpoint mismatch')
        else:
            root = Path(args.factorial_dir)/old_group
            expected = v.read(root/'trajectory.json')[0]
            weights = u.load_checkpoint(root/'snapshots/000001.pth')
            if row['losses'] != expected['losses'] or d.digest(t.model.state_dict()) != d.digest(weights['student']) or d.digest(t.model_teacher.state_dict()) != d.digest(weights['teacher']):
                raise ValueError(old_group+' endpoint mismatch')
        findings[old_group] = dict(endpoint_passed=True)
    largest = [max(chosen, key=lambda r: r['width']*r['height'])]
    samples = {s.sample_id: s for s in groups['target_eval']}
    for group in RUN_ORDER:
        setting = GROUPS[group]
        t.update_teacher = (lambda: None) if setting['fixed_teacher'] else update_teacher
        d.restore(t, g, initial)
        reference = [adapt_one(t, g, data, order, i, .5, measure=False) for i in range(count)]
        d.restore(t, g, initial)
        for i in range(count):
            if i == count-1:
                # 真实完整序列化/恢复，覆盖Adam、RDDM EMA、buffer和RNG。
                state = d.capture(t, g, i, order)
                out.checkpoint('acceptance/resume.pth', state)
                d.restore(t, g, u.load_checkpoint(out.root/'acceptance/resume.pth'))
                if d.digest(d.capture(t, g, i, order)) != d.digest(state): raise ValueError('Acceptance resume mismatch')
                del state
            row = adapt_one(t, g, data, order, i, .5)
            if row['losses'] != reference[i]['losses'] or row['state_sha256'] != reference[i]['state_sha256']:
                raise ValueError(group+' observer/restore changed update')
        with v.isolated_diagnostic(t, g, count, order), torch.no_grad():
            extra, metrics = stage_images(t, g, samples, largest, count, opt.seed, out, 'acceptance/'+group, .5)
            files.update(extra)
            d.score(t.model, [samples[n] for n in protocol['probes']['target_eval']])
        if d.digest(d.capture(t, g, count, order)) != reference[-1]['state_sha256']:
            raise ValueError('Full-frame diagnostic changed state')
        if setting['fixed_teacher'] and d.digest(t.model_teacher.state_dict()) != d.digest(initial['teacher']['weights']):
            raise ValueError('Fixed teacher changed')
        findings[group] = dict(prefix=count, observer_exact=True, resume_exact=True, full_frame_isolated=True)
    # 单个明确路径只原子替换为轻量完成标记，不留下额外大型验收状态。
    out.checkpoint('acceptance/resume.pth', dict(passed=True, plan_sha256=contract_hash))
    files['acceptance/resume.pth'] = v.sha(out.root/'acceptance/resume.pth')
    out.json('acceptance.json', dict(passed=True, plan_sha256=contract_hash, checks=findings, files=files,
                                    git=git_identity()))
    del t, g, initial, update_teacher
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print('G4/G5 acceptance PASSED', flush=True)


def run_group(args, group):
    if group not in GROUPS: raise ValueError('Only G4/G5 may run')
    out, protocol, groups, chosen, stages = preflight(args)
    plan_hash = v.sha(out.root/'plan.json')
    passed = v.read(out.root/'acceptance.json')
    if not passed['passed'] or passed['plan_sha256'] != plan_hash: raise ValueError('Run acceptance first')
    verify_index(out.root, passed['files'])
    folder = out.root/group
    status_path = folder/'status.json'
    if status_path.exists():
        status = v.read(status_path)
        if status['status'] != 'complete' or status['plan_sha256'] != plan_hash: raise ValueError('Invalid completion status')
        verify_index(out.root, status['files'])
        print(group+': already complete; SKIP', flush=True)
        return
    t, g, data, order, opt, initial = build(args, out, protocol, group)
    source_teacher = d.digest(initial['teacher']['weights'])
    if GROUPS[group]['fixed_teacher']: t.update_teacher = lambda: None
    resume = folder/'resume.pth'
    rows, metrics, files, cursor = [], [], {}, 0
    if resume.exists():
        saved = u.load_checkpoint(resume)
        if saved['plan_sha256'] != plan_hash or saved['group'] != group: raise ValueError('Resume contract differs')
        rows, metrics, files = saved['rows'], saved['metrics'], saved['files']
        verify_index(out.root, files)
        cursor = saved['state']['cursor']
        if len(rows) != cursor or not 0 <= cursor <= len(order) or saved['state']['order'] != order:
            raise ValueError('Resume cursor/order differs')
        if d.digest(saved['state']) != saved['state_sha256']: raise ValueError('Resume state changed')
        d.restore(t, g, saved['state'])
        if d.digest(d.capture(t, g, cursor, order)) != saved['state_sha256']: raise ValueError('Resume restore differs')
        del saved
    del initial

    def commit(step):
        state = d.capture(t, g, step, order)
        out.checkpoint(group+'/resume.pth', dict(group=group, plan_sha256=plan_hash, state=state,
                       state_sha256=d.digest(state), rows=rows, metrics=metrics, files=files))
        out.json(group+'/progress.json', dict(committed_step=step, total=len(order), used_bytes=out.used()))

    if not resume.exists():
        extra, measured = observe(t, g, groups, chosen, 0, order, opt.seed, out, group, protocol, stages)
        files.update(extra); metrics.extend(measured)
        commit(0)
    for i in range(cursor, len(order)):
        row = adapt_one(t, g, data, order, i, .5)
        if GROUPS[group]['fixed_teacher'] and d.digest(t.model_teacher.state_dict()) != source_teacher:
            raise ValueError('G5 fixed teacher changed')
        if group == 'G5':
            prior = v.read(Path(args.factorial_dir)/'G1/trajectory.json')[i]
            if 'generated_rgb' in prior:
                if not np.allclose(row['inputs']['x_sd']['mean'], prior['generated_rgb'], rtol=0, atol=1e-6):
                    raise ValueError('G5 generator channel means differ from G1')
                row['g1_generated_rgb_match'] = True
        rows.append(row)
        extra, measured = observe(t, g, groups, chosen, i+1, order, opt.seed, out, group, protocol, stages)
        files.update(extra); metrics.extend(measured)
        # 每步提交，异常最多重算一个未提交步骤；不保存每步大权重快照。
        commit(i+1)
        print('%s: committed %d/%d; %.3f GB' % (group, i+1, len(order), out.used()/1e9), flush=True)
    generator_check = dict(channel_means_checked=sum(r.get('g1_generated_rgb_match', False) for r in rows))
    old_resume = Path(args.factorial_dir)/'G1/resume.pth'
    if group == 'G5' and old_resume.exists():
        old_state = u.load_checkpoint(old_resume)['state']
        current = d.capture(t, g, len(order), order)
        keys = ('rddm', 'rddm_ema', 'rddm_optimizer', 'rddm_local_step', 'scaler')
        generator_check['final_generator_exact'] = d.digest({k: current[k] for k in keys}) == d.digest({k: old_state[k] for k in keys})
        if not generator_check['final_generator_exact']: raise ValueError('G5 final generator differs from G1')
        del old_state, current
    expected = 5*len(chosen)*len(stages)+len(order)
    if sum(name.endswith('.png') for name in files) != expected: raise ValueError('Image coverage differs')
    verify_index(out.root, files)
    files[group+'/trajectory.json'] = out.json(group+'/trajectory.json', rows)
    files[group+'/metrics.csv'] = out.put(group+'/metrics.csv', v.csv_bytes(metrics))
    # resume内部索引只覆盖观测文件，完成状态同时保护恢复状态本身与最终汇总。
    files[group+'/resume.pth'] = v.sha(resume)
    out.json(group+'/status.json', dict(status='complete', group=group, steps=len(order), images=expected,
                                      plan_sha256=plan_hash, files=files, generator_check=generator_check, completion_git=git_identity()))
    del t, g
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def compare(args):
    out, protocol, groups, chosen, stages = preflight(args)
    order = protocol['order']; last = len(order)
    roots = {'G0': Path(args.diagnostic_dir)/'evaluation'}
    roots.update({g: Path(args.factorial_dir)/g/'evaluation' for g in ('G1', 'G2', 'G3')})
    roots.update({g: out.root/g/'evaluation' for g in GROUPS})
    for group in GROUPS:
        status = v.read(out.root/group/'status.json')
        if status['status'] != 'complete' or status['plan_sha256'] != v.sha(out.root/'plan.json'):
            raise ValueError('Both G4/G5 must complete before comparison')
        verify_index(out.root, status['files'])
    zero = {r['sample_id']: r for r in v.read(roots['G0']/'000000_student.json')}
    final = {g: {r['sample_id']: r for r in v.read(root/('%06d_student.json'%last))} for g, root in roots.items()}
    if any(set(rows) != set(zero) for rows in final.values()): raise ValueError('Final evaluation coverage differs')
    summary, paired, curves = {}, [], []
    target = set(order)
    for group, records in final.items():
        summary[group] = {}
        for split, ids in [('target', target), ('source_val', set(zero)-target)]:
            delta = np.array([records[n]['psnr']-zero[n]['psnr'] for n in sorted(ids)])
            summary[group][split] = dict(count=len(ids), psnr=float(np.mean([records[n]['psnr'] for n in ids])),
                ssim=float(np.mean([records[n]['ssim'] for n in ids])), mean_delta=float(delta.mean()),
                median_delta=float(np.median(delta)), improved=int((delta>0).sum()), worse=int((delta<0).sum()),
                worse_over_1db=int((delta < -1).sum()), worst_delta=float(delta.min()), p10_delta=float(np.quantile(delta,.1)))
    for name in sorted(zero):
        row = dict(sample_id=name, split='target' if name in target else 'source_val', source_psnr=zero[name]['psnr'],
                   **{g: rs[name]['psnr'] for g, rs in final.items()})
        row.update(G4_minus_G0=row['G4']-row['G0'], G5_minus_G1=row['G5']-row['G1'], G5_minus_G3=row['G5']-row['G3'])
        paired.append(row)
    for group, root in roots.items():
        for step in range(last+1):
            rows = {r['sample_id']: r for r in v.read(root/('%06d_student.json'%step))}
            for split, ids in protocol['probes'].items():
                curves.append(dict(group=group, step=step, split=split, psnr=float(np.mean([rows[n]['psnr'] for n in ids]))))
    out.json('comparison/summary.json', summary)
    out.put('comparison/paired.csv', v.csv_bytes(paired))
    out.put('comparison/probe_curves.csv', v.csv_bytes(curves))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(protocol['probes']), figsize=(6*len(protocol['probes']), 4), squeeze=False)
    for ax, (split, ids) in zip(axes[0], protocol['probes'].items()):
        for group in roots:
            subset = [r for r in curves if r['group'] == group and r['split'] == split]
            ax.plot([r['step'] for r in subset], [r['psnr'] for r in subset], label=group)
        ax.axhline(float(np.mean([zero[n]['psnr'] for n in ids])), color='grey', linestyle='--', label='source-only')
        ax.set(title=split, xlabel='Adaptation step', ylabel='PSNR (dB)')
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    image = io.BytesIO(); fig.savefig(image, format='png', dpi=150); plt.close(fig)
    out.put('comparison/G0_G5_probe_psnr.png', image.getvalue())
    out.json('comparison/status.json', dict(status='complete', note='G0-G3 reused; final student raw and fixed probes; no reference-based selection'))
    print('G0-G5 comparison COMPLETE', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment_dir', 'diagnostic_dir', 'factorial_dir', 'output_dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--selection_csv', default=str(SELECTION))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_gb', type=float, default=10.)
    parser.add_argument('--phase', choices=['preflight', 'accept', 'run', 'compare', 'all'], default='preflight')
    parser.add_argument('--group', choices=list(GROUPS))
    args = parser.parse_args(argv)
    if not 0 < args.max_gb <= 10: parser.error('--max_gb must be in (0,10]')
    if args.phase == 'run' and args.group is None: parser.error('--group is required for run')
    if os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8') not in (':4096:8', ':16:8'):
        raise ValueError('Invalid CUBLAS workspace setting')
    torch.use_deterministic_algorithms(True, warn_only=False)
    u.seed_everything(23)
    if args.phase == 'preflight':
        out, *_ = preflight(args)
        print('G4/G5 preflight PASSED; peak estimate %.3f GB' % (v.read(out.root/'storage_estimate.json')['estimated_peak_bytes']/1e9), flush=True)
    elif args.phase == 'accept': acceptance(args)
    elif args.phase == 'run': run_group(args, args.group)
    elif args.phase == 'compare': compare(args)
    else:
        acceptance(args)
        for group in RUN_ORDER: run_group(args, group)
        compare(args)


if __name__ == '__main__': main()
