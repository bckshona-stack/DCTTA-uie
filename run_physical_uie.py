"""G6/G7：固定源教师、逐图重置的200步物理再退化TTA。"""
import argparse
import gc
import io
import json
import math
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import diagnose_uie as d
import diagnose_visual_uie as v
import uie_tta as u
import physical_degradation as p
import pilot_physical_uie as pilot
import run_strength_uie as s
import run_factorial_uie as f

GROUPS = {'G6': .5, 'G7': 1.}
FIT_STEPS = 200
FIT = p.FitConfig()


def sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


class FixedPhysicalTTA(u.SRTTA):
    def update_teacher(self):
        # 原adapt仍依次执行Adam/Fisher；只屏蔽教师EMA。
        pass

    def _teacher_eight_view(self, image):
        sync(self.device); start = time.perf_counter()
        result = super()._teacher_eight_view(image)
        sync(self.device); self.teacher_seconds = time.perf_counter()-start
        return result


def capture(t, cursor, order):
    return dict(format_version='physical_v1', cursor=cursor, order=order, numerics=d.numerics(),
                fisher_origin=deepcopy(getattr(t, 'fisher_origin', None)),
                frozen_loss_sha256=d.digest([t.perceptual_loss.state_dict(), t.wt1.state_dict()]),
                student=d.model_state(t.model), teacher=d.model_state(t.model_teacher),
                source=d.cpu_tree(t.model_state), fisher=d.cpu_tree(t.fishers),
                fisher_identity=deepcopy(t.fisher_identity), student_optimizer=d.cpu_tree(t.optimizer.state_dict()),
                steps=t.steps, rng=d.cpu_tree(d.random_state()))


def restore(t, state):
    if state['numerics'] != d.numerics(): raise ValueError('Numerical mode differs')
    if state['frozen_loss_sha256'] != d.digest([t.perceptual_loss.state_dict(), t.wt1.state_dict()]):
        raise ValueError('Frozen losses differ')
    d.restore_model(t.model, state['student']); d.restore_model(t.model_teacher, state['teacher'])
    t.model_state = {n: x.to(t.device) for n, x in state['source'].items()}
    t.fishers = {n: [x.to(t.device) for x in values] for n, values in state['fisher'].items()}
    t.fisher_identity = deepcopy(state['fisher_identity']); t.fisher_origin = deepcopy(state.get('fisher_origin'))
    t._validate_fishers(t.fishers)
    t.optimizer.load_state_dict(state['student_optimizer']); t.steps = state['steps']
    d.restore_random_state(state['rng'])


@contextmanager
def diagnostic(t, cursor, order):
    state = capture(t, cursor, order); before = d.digest(state)
    try:
        t.model.eval(); t.model_teacher.eval()
        yield
    finally:
        restore(t, state)
        if d.digest(capture(t, cursor, order)) != before:
            raise RuntimeError('Diagnostic changed adaptation state')


class PhysicalGenerator:
    def __init__(self, gamma, config=FIT, steps=FIT_STEPS):
        self.gamma, self.config, self.steps = gamma, config, steps

    def train(self, raw, pseudo, names):
        sync(raw.device); started = time.perf_counter()
        self.trace = []
        for model, row in p.fit_steps(raw, pseudo, self.config, self.steps):
            self.trace.append(row)
        with torch.no_grad(): result = model.render(raw.detach(), self.gamma).detach()
        sync(raw.device); self.seconds = time.perf_counter()-started
        # 后续统计/参数复制不计入fit_seconds。
        self.payload = model.parameter_payload()
        with torch.no_grad():
            t, b, _ = model.fields()
            self.last = dict(fit=self.trace[-1], parameter_sha256=model.parameter_hash(),
                initial_sha256=model.initial_hash, gamma=self.gamma, fit_steps=self.steps,
                T=p.channel_stats(t), B=b.flatten().cpu().tolist(),
                raw=p.channel_stats(raw), pseudo=p.channel_stats(pseudo), x_sd=p.channel_stats(result),
                x_sd_sha256=p.tensor_hash(result), fit_seconds=self.seconds)
        self.raw = raw.detach().cpu(); self.pseudo = pseudo.detach().cpu(); self.x_sd = result.cpu()
        sync(raw.device); self.total_seconds = time.perf_counter()-started
        return result


def load_fit(payload, device):
    model = p.PhysicalFit(p.FitConfig(**payload['config']), payload['height'], payload['width'], device)
    with torch.no_grad(): model.a.copy_(payload['a']); model.b.copy_(payload['b'])
    if model.parameter_hash() != payload['parameter_sha256'] or model.initial_hash != payload['initial_sha256']:
        raise ValueError('Fit parameter identity differs')
    return model


def estimate(rows, chosen, stages, initial):
    def png(h, w): return math.ceil(((3*w+1)*h)*1.01)+65536
    full = 2*(4*len(stages)*sum(png(r['height'], r['width']) for r in chosen)+
              sum(png(r['height'], r['width']) for r in rows))
    patch = 2*len(rows)*3*png(320, 320)
    cache = sum(12*r['width']*r['height']+200000 for r in chosen)
    # 无RDDM权重/动量；成熟学生Adam+梯度+源/Fisher等保守按初态3倍。
    small = {k: initial[k] for k in ('student', 'teacher', 'source', 'fisher', 'student_optimizer')}
    mature = 3*d.state_bytes(small)+32_000_000
    peak = full+patch+cache+3*mature+150_000_000
    return dict(estimated_peak_bytes=peak, full_images_per_group=4*len(chosen)*len(stages)+len(rows),
                patch_images_per_group=3*len(rows), cache_bytes=cache, checkpoint_peak_bytes=3*mature)


def preflight(args):
    device = pilot.configure_runtime(args.device); v.runtime(); d.require_strict()
    roots = [args.experiment_dir, args.diagnostic_dir, args.factorial_dir, args.strength_dir,
             args.visual_dir, args.pilot_dir]
    v.independent_output(args.output_dir, roots)
    output_path = Path(args.output_dir)
    if output_path.exists() and not (output_path/'plan.json').exists() and any(output_path.iterdir()):
        raise ValueError('Output must initially be empty')
    diag = Path(args.diagnostic_dir); protocol, rows = v.inventory(diag, args.experiment_dir)
    if protocol['diagnostic_protocol'] != 'fixed_fisher_strict': raise ValueError('Strict v2 required')
    d.verify_fixed_files(args.experiment_dir)
    historical = {k.replace('\\', '/'): x for k, x in protocol['environment']['implementation'].items()}
    for name in ('uie_tta.py', 'options.py', 'uie_backbones/nu2net.py', 'utils/loss_utils.py', 'utils/val_utils.py'):
        if v.sha(Path(__file__).parent/name) != historical[name]: raise ValueError('Original implementation changed: '+name)
    env = d.environment()
    if device.type == 'cuda':
        for key in ('packages', 'cuda', 'cudnn', 'numerical_settings'):
            if env[key] != protocol['environment'][key]: raise ValueError('Original environment differs: '+key)
    for phase in ('run', 'evaluation'):
        if v.read(diag/phase/'status.json')['status'] != 'complete': raise ValueError('Original G0 incomplete')
    pilot_root = Path(args.pilot_dir); pm = v.read(pilot_root/'manifest.json'); ps = v.read(pilot_root/'status.json')
    if ps['status'] != 'complete' or ps['manifest_sha256'] != v.sha(pilot_root/'manifest.json'):
        raise ValueError('Completed physical pilot required')
    s.verify_index(pilot_root, ps['files'])
    if (pm['fit'] != FIT.identity() or pm['points'][-1] != FIT_STEPS or
        pm['implementations']['physical_degradation.py'] != v.sha(p.__file__) or
        pm['source_sha256'] != protocol['source_sha256'] or pm['protocol_sha256'] != v.sha(diag/'protocol.json')):
        raise ValueError('Approved pilot fit/source/protocol differs')
    chosen = v.selection(Path(args.visual_dir)/'selected_samples.csv', rows)
    if v.read(Path(args.visual_dir)/'status.json')['status'] != 'complete': raise ValueError('Completed visual_v2 required')
    if [r['sample_id'] for r in chosen] != [r['id'] for r in pm['samples']['samples']]:
        raise ValueError('Fixed ten images differ from pilot')
    groups, _, checkpoint = d.inputs(args.experiment_dir)
    data = u.TargetRawDataset([u.Sample(x.sample_id, x.raw_path, None) for x in groups['target_eval']],
                              protocol['config']['patch_size'], protocol['config']['seed'])
    if data.identity() != protocol['target']: raise ValueError('Target crop identity differs')
    source_report = v.read(Path(args.experiment_dir)/'source_only/run.json')
    if source_report['source_checkpoint_sha256'] != v.sha(checkpoint) or source_report['metrics'] != 'RGB float clip[0,1], data_range=1, skimage defaults':
        raise ValueError('Source-only identity differs')
    initial_path = diag/'run/replay_states/000000.pth'; initial = u.load_checkpoint(initial_path)
    if d.digest(initial) != d.digest(u.load_checkpoint(diag/'acceptance/initial.pth')):
        raise ValueError('Initial state differs')
    stages = v.schedule(len(rows)); size = estimate(rows, chosen, stages, initial); del initial
    if size['estimated_peak_bytes'] > args.max_gb*1e9: raise ValueError('Storage peak exceeds budget')
    old_plan = v.read(Path(args.factorial_dir)/'plan.json')
    if (old_plan['protocol_sha256'] != v.sha(diag/'protocol.json') or
        old_plan['initial_sha256'] != v.sha(initial_path) or old_plan['entry_sha256'] != v.sha(f.__file__)):
        raise ValueError('Old factorial protocol/initial/entry differs')
    for name in ('000000_student.json', '%06d_student.json'%len(rows)):
        if v.sha(diag/'evaluation'/name) != old_plan['g0_evaluation_sha256'][name]:
            raise ValueError('Old G0 evaluation changed')
    old = {}
    for group in ('G1', 'G2', 'G3', 'G4', 'G5'):
        root = Path(args.factorial_dir if group in ('G1', 'G2', 'G3') else args.strength_dir)/group
        status = v.read(root/('run_status.json' if group in ('G1', 'G2', 'G3') else 'status.json'))
        if status['status'] != 'complete': raise ValueError('Old group incomplete: '+group)
        final = root/('evaluation/%06d_student.json'%len(rows))
        # 回传可能省略旧大权重；这里只绑定已有评价及第1步G3端点。
        old[group] = v.sha(final)
        expected_sha = (v.read(root/'evaluation/index.json')[final.name] if group in ('G1', 'G2', 'G3')
                        else status['files'][group+'/evaluation/'+final.name])
        if old[group] != expected_sha: raise ValueError('Old final evaluation changed: '+group)
        if group in ('G1', 'G2', 'G3') and v.read(root/'evaluation/status.json')['status'] != 'complete':
            raise ValueError('Old evaluation incomplete: '+group)
    out = s.BudgetOutput(args.output_dir, int(args.max_gb*1e9))
    value = dict(version=1, groups=GROUPS, fit=FIT.identity(), fit_steps=FIT_STEPS, order=protocol['order'],
        stages=stages, chosen=chosen, estimate=size, inputs=dict(protocol=v.sha(diag/'protocol.json'),
        initial=v.sha(initial_path), source=v.sha(checkpoint), pilot_manifest=v.sha(pilot_root/'manifest.json'),
        pilot_status=v.sha(pilot_root/'status.json'), old_final=old,
        g3_endpoint=v.sha(Path(args.factorial_dir)/'G3/snapshots/000001.pth')),
        implementations={name: v.sha(Path(__file__).parent/name) for name in
            ('run_physical_uie.py', 'physical_degradation.py', 'pilot_physical_uie.py', 'diagnose_uie.py',
             'diagnose_visual_uie.py', 'run_strength_uie.py', 'run_factorial_uie.py', 'uie_tta.py', 'options.py',
             'uie_backbones/nu2net.py', 'utils/loss_utils.py', 'utils/val_utils.py', 'train_source_uie.py')},
        environment=env, numerics=d.numerics(), roots=[str(Path(x).resolve()) for x in roots],
        device=str(device), max_bytes=out.limit, source_only_sha256=v.sha(Path(args.experiment_dir)/'source_only/run.json'),
        definition='fixed source x8; per-image reset K200; cumulative student; no reference fitting; full-frame diagnostics independent')
    value = json.loads(json.dumps(value))
    if (out.root/'plan.json').exists():
        if v.read(out.root/'plan.json')['contract'] != value: raise ValueError('Experiment contract changed; use a new directory')
    else:
        if any(out.root.iterdir()): raise ValueError('Output must initially be empty')
        out.json('plan.json', dict(contract=value, creation_git=s.git_identity()))
        out.put('selected_samples.csv', v.csv_bytes(chosen))
    if v.sha(out.root/'selected_samples.csv') != __import__('hashlib').sha256(v.csv_bytes(chosen)).hexdigest():
        raise ValueError('Selection output changed')
    d.require_space(out.root, size['estimated_peak_bytes'])
    print('G6/G7 preflight PASSED; K200; storage peak <= %.3f GB'% (size['estimated_peak_bytes']/1e9), flush=True)
    return out, protocol, groups, chosen, stages


def build(args, protocol, perceptual_loss=None, wavelet=None):
    opt = SimpleNamespace(**protocol['config']); opt.device = str(u.resolve_device(args.device, 0))
    u.seed_everything(opt.seed)
    with u.isolated_rng(opt.seed): model = u.build_backbone('nu2net', Path(args.experiment_dir)/'source/checkpoints/best.pth', opt.device)
    u.configure_model(opt, model); params, _ = u.collect_params(model)
    t = FixedPhysicalTTA(opt, model, torch.optim.Adam(params, lr=opt.lr, betas=tuple(opt.betas)),
                         perceptual_loss=perceptual_loss, wavelet=wavelet)
    initial = u.load_checkpoint(Path(args.diagnostic_dir)/'run/replay_states/000000.pth')
    if d.digest(t.model.state_dict()) != d.digest(initial['student']['weights']): raise ValueError('Source student differs')
    restore(t, initial)
    if d.digest(t.fishers) != d.digest(u.load_checkpoint(Path(args.experiment_dir)/'dctta/fisher.pt')['fishers']):
        raise ValueError('Approved Fisher differs')
    data = u.TargetRawDataset([u.Sample(x.sample_id, x.raw_path, None) for x in d.inputs(args.experiment_dir)[0]['target_eval']],
                              opt.patch_size, opt.seed)
    if data.identity() != protocol['target'] or t.fisher_origin['policy'] != 'fixed_first_round':
        raise ValueError('Fixed Fisher/crop differs')
    initial = capture(t, 0, protocol['order'])
    if d.digest(initial['teacher']['weights']) != d.digest(initial['source']): raise ValueError('Fixed teacher differs from source')
    return t, data, initial


def adapt_one(t, data, order, index, gamma, measure=True):
    by_id = {x.sample_id: i for i, x in enumerate(data.samples)}
    name, image = data[by_id[order[index]]]; gen = PhysicalGenerator(gamma)
    observer = f.UpdateSummary(t) if measure else None; t.observer = observer
    sync(t.device); started = time.perf_counter()
    try: losses = t.adapt(image.unsqueeze(0), [name], gen)
    finally: t.observer = None
    sync(t.device); elapsed = time.perf_counter()-started
    d.require_strict()
    if d.digest(t.model_teacher.state_dict()) != d.digest(t.model_state): raise ValueError('Fixed teacher changed')
    if any(x.grad is not None for x in t.model_teacher.parameters()): raise ValueError('Teacher gradient appeared')
    row = dict(step=index+1, sample_id=name, crop=data.crops[by_id[name]], losses=losses,
        inputs=gen.last, **(observer.values if observer else {}),
        timing=dict(teacher_seconds=t.teacher_seconds, fit_seconds=gen.seconds,
                    fit_statistics_seconds=gen.total_seconds-gen.seconds, adaptation_seconds=elapsed,
                    student_with_observer_seconds=elapsed-t.teacher_seconds-gen.total_seconds))
    return row, gen


def full_cached(t, sample, out):
    name = 'full_cache/'+sample.sample_id+'.pth'; path = out.root/name
    key = dict(raw_sha256=v.sha(sample.raw_path), source_sha256=d.digest(t.model_teacher.state_dict()),
               fit=FIT.identity(), fit_steps=FIT_STEPS)
    if path.exists():
        packet = u.load_checkpoint(path)
        if packet['key'] != key or p.tensor_hash(packet['pseudo']) != packet['pseudo_sha256']:
            raise ValueError('Full cache identity differs')
    else:
        raw = u.read_rgb(sample.raw_path).unsqueeze(0).to(t.device)
        with torch.no_grad(): pseudo = u.eight_view_mean(t.model_teacher, raw).detach()
        for model, row in p.fit_steps(raw, pseudo, FIT, FIT_STEPS): pass
        packet = dict(key=key, parameters=model.parameter_payload(), pseudo=pseudo.cpu(),
                      pseudo_sha256=p.tensor_hash(pseudo), last_fit=row)
        out.checkpoint(name, packet)
    model = load_fit(packet['parameters'], t.device)
    raw = u.read_rgb(sample.raw_path).unsqueeze(0).to(t.device)
    return raw, packet['pseudo'].to(t.device), model, {name: v.sha(path)}


def stage_images(t, samples, chosen, step, out, prefix, gamma):
    files, metrics = {}, []
    for item in chosen:
        sample = samples[item['sample_id']]
        raw, pseudo, fitted, cache = full_cached(t, sample, out); files.update(cache)
        with torch.no_grad():
            xsd = fitted.render(raw, gamma).detach(); ysd = t.model(xsd); yin = t.model(raw)
        # reference仅在所有拟合与预测完成后进入离线指标。
        ref = u.read_rgb(sample.reference_path).unsqueeze(0)
        for role, value in [('teacher_pseudo', pseudo), ('x_sd', xsd), ('y_sd', ysd), ('y_in', yin)]:
            path = '%s/images/%s/%06d/%s'%(prefix, role, step, s.image_name(sample.sample_id))
            sha, quant = v.export_image(out, path, value); files[path] = sha
            score = v.psnr(value, ref)
            metrics.append(dict(step=step, sample_id=sample.sample_id, role=role, psnr=score,
                png_psnr=v.psnr(quant, ref), source_only_psnr=item['source_psnr'],
                delta_source_psnr=score-item['source_psnr'], path=path, scope='full_independent_fit'))
    return files, metrics


def observe(t, groups, chosen, step, order, out, group, protocol, stages):
    files, metrics = {}, []
    sync(t.device); start = time.perf_counter()
    with diagnostic(t, step, order):
        samples = {x.sample_id: x for split in groups.values() for x in split}
        selected = groups['target_eval']+groups['source_val'] if step in stages else [samples[n] for ids in protocol['probes'].values() for n in ids]
        with torch.no_grad(): records = d.score(t.model, selected)
        path = '%s/evaluation/%06d_student.json'%(group, step); files[path] = out.json(path, records)
        if step in stages:
            extra, metrics = stage_images(t, samples, chosen, step, out, group, GROUPS[group]); files.update(extra)
        if step == len(order):
            with torch.no_grad():
                for sample in groups['target_eval']:
                    pred = t.model(u.read_rgb(sample.raw_path).unsqueeze(0).to(t.device))
                    path = group+'/final/y_in/'+s.image_name(sample.sample_id)
                    sha, quant = v.export_image(out, path, pred); files[path] = sha
                    ref = u.read_rgb(sample.reference_path).unsqueeze(0)
                    metrics.append(dict(step=step, sample_id=sample.sample_id, role='final_y_in', psnr=v.psnr(pred, ref),
                        png_psnr=v.psnr(quant, ref), source_only_psnr='', delta_source_psnr='', path=path, scope='full_student_raw'))
    sync(t.device)
    return files, metrics, time.perf_counter()-start


def acceptance(args):
    out, protocol, groups, chosen, stages = preflight(args); plan_sha = v.sha(out.root/'plan.json')
    path = out.root/'acceptance.json'
    if path.exists():
        value = v.read(path)
        if not value['passed'] or value['plan_sha256'] != plan_sha: raise ValueError('Acceptance identity differs')
        s.verify_index(out.root, value['files']); print('G6/G7 acceptance already PASSED', flush=True); return
    t, data, initial = build(args, protocol); order = protocol['order']; findings, files = {}, {}
    # gamma=0等价原固定教师raw组G3；精确核对真实第一步损失和权重。
    row, _ = adapt_one(t, data, order, 0, 0., measure=False)
    expected = v.read(Path(args.factorial_dir)/'G3/trajectory.json')[0]
    weights = u.load_checkpoint(Path(args.factorial_dir)/'G3/snapshots/000001.pth')
    if row['losses'] != expected['losses'] or d.digest(t.model.state_dict()) != d.digest(weights['student']):
        raise ValueError('Gamma-zero G3 endpoint mismatch')
    findings['G3_gamma_zero'] = True; n = min(3, len(order))
    for group, gamma in GROUPS.items():
        restore(t, initial); reference = []
        for i in range(n):
            row, gen = adapt_one(t, data, order, i, gamma, measure=False)
            reference.append((row['losses'], d.digest(capture(t, i+1, order)), gen.payload['parameter_sha256']))
        restore(t, initial)
        for i in range(n):
            if i == n-1:
                state = capture(t, i, order); out.checkpoint('acceptance/resume.pth', state)
                restore(t, u.load_checkpoint(out.root/'acceptance/resume.pth'))
            row, gen = adapt_one(t, data, order, i, gamma)
            if (row['losses'], d.digest(capture(t, i+1, order)), gen.payload['parameter_sha256']) != reference[i]:
                raise ValueError('Observer/resume changed physical adaptation')
            if i == 0:
                with diagnostic(t, i+1, order):
                    largest = [max(chosen, key=lambda x: x['height']*x['width'])]
                    extra, _ = stage_images(t, {x.sample_id:x for x in groups['target_eval']}, largest, i+1, out, 'acceptance/'+group, gamma)
                    files.update(extra)
                if d.digest(capture(t, i+1, order)) != reference[i][1]: raise ValueError('Full diagnostic changed state')
        findings[group] = dict(prefix=n, observer_exact=True, resume_exact=True, full_frame_isolated=True)
    out.checkpoint('acceptance/resume.pth', dict(passed=True, plan_sha256=plan_sha))
    files['acceptance/resume.pth'] = v.sha(out.root/'acceptance/resume.pth')
    out.json('acceptance.json', dict(passed=True, plan_sha256=plan_sha, checks=findings, files=files))
    del t, initial; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print('G6/G7 device acceptance PASSED: G3 endpoint, observation, resume, full-frame', flush=True)


def run_group(args, group):
    if group not in GROUPS: raise ValueError('Only G6/G7')
    out, protocol, groups, chosen, stages = preflight(args); plan_sha = v.sha(out.root/'plan.json')
    passed = v.read(out.root/'acceptance.json')
    if not passed['passed'] or passed['plan_sha256'] != plan_sha: raise ValueError('Run acceptance first')
    s.verify_index(out.root, passed['files']); folder = out.root/group
    if (folder/'status.json').exists():
        status = v.read(folder/'status.json')
        if status['status'] != 'complete' or status['plan_sha256'] != plan_sha: raise ValueError('Completion identity differs')
        s.verify_index(out.root, status['files']); print(group+': already complete; SKIP', flush=True); return
    t, data, initial = build(args, protocol); order = protocol['order']; cursor = 0
    rows, metrics, files = [], [], {}; resume = folder/'resume.pth'
    if resume.exists():
        saved = u.load_checkpoint(resume)
        if saved['group'] != group or saved['plan_sha256'] != plan_sha: raise ValueError('Resume contract differs')
        rows, metrics, files = saved['rows'], saved['metrics'], saved['files']; s.verify_index(out.root, files)
        cursor = saved['state']['cursor']
        if len(rows) != cursor or not 0 <= cursor <= len(order) or saved['state']['order'] != order or saved['state_sha256'] != d.digest(saved['state']):
            raise ValueError('Resume state/cursor/order differs')
        restore(t, saved['state'])
        if d.digest(capture(t, cursor, order)) != saved['state_sha256']: raise ValueError('Resume restoration differs')
        del saved
    del initial
    def commit(step):
        state = capture(t, step, order)
        out.checkpoint(group+'/resume.pth', dict(group=group, plan_sha256=plan_sha, state=state,
            state_sha256=d.digest(state), rows=rows, metrics=metrics, files=files))
        out.json(group+'/progress.json', dict(committed_step=step, total=len(order), used_bytes=out.used()))
    if not resume.exists():
        extra, measured, _ = observe(t, groups, chosen, 0, order, out, group, protocol, stages)
        files.update(extra); metrics.extend(measured); commit(0)
    for i in range(cursor, len(order)):
        row, gen = adapt_one(t, data, order, i, GROUPS[group])
        prefix = group+'/training/%06d'%(i+1); export_start = time.perf_counter()
        for role, value in [('raw', gen.raw), ('teacher_pseudo', gen.pseudo), ('x_sd', gen.x_sd)]:
            name = prefix+'/'+role+'.png'; files[name] = v.export_image(out, name, value)[0]
        name = prefix+'/parameters.pth'; files[name] = out.checkpoint(name, gen.payload)
        name = prefix+'/fit_trace.csv'; files[name] = out.put(name, v.csv_bytes(gen.trace))
        row['timing']['training_export_seconds'] = time.perf_counter()-export_start
        extra, measured, seconds = observe(t, groups, chosen, i+1, order, out, group, protocol, stages)
        files.update(extra); metrics.extend(measured); row['timing']['observation_seconds'] = seconds; rows.append(row)
        commit(i+1)
        print('%s: committed %d/%d; %.3f GB'%(group, i+1, len(order), out.used()/1e9), flush=True)
    expected = 4*len(chosen)*len(stages)+len(order)+3*len(order)
    if sum(x.startswith(group+'/') and x.endswith('.png') for x in files) != expected:
        raise ValueError('Image coverage differs')
    files[group+'/trajectory.json'] = out.json(group+'/trajectory.json', rows)
    files[group+'/metrics.csv'] = out.put(group+'/metrics.csv', v.csv_bytes(metrics))
    files[group+'/resume.pth'] = v.sha(resume)
    s.verify_index(out.root, files)
    out.json(group+'/status.json', dict(status='complete', group=group, steps=len(order), images=expected,
                                      plan_sha256=plan_sha, files=files, completion_git=s.git_identity()))
    del t; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def compare(args):
    out, protocol, groups, chosen, stages = preflight(args); last = len(protocol['order'])
    roots = {'G0': Path(args.diagnostic_dir)/'evaluation'}
    roots.update({g: Path(args.factorial_dir)/g/'evaluation' for g in ('G1', 'G2', 'G3')})
    roots.update({g: Path(args.strength_dir)/g/'evaluation' for g in ('G4', 'G5')})
    roots.update({g: out.root/g/'evaluation' for g in GROUPS})
    for group in GROUPS:
        status = v.read(out.root/group/'status.json')
        if status['status'] != 'complete' or status['plan_sha256'] != v.sha(out.root/'plan.json'): raise ValueError('Both new groups must complete')
        s.verify_index(out.root, status['files'])
    traces = {g: v.read(out.root/g/'trajectory.json') for g in GROUPS}
    if len(traces['G6']) != last or len(traces['G7']) != last: raise ValueError('Trace coverage differs')
    for a, b in zip(traces['G6'], traces['G7']):
        if (a['sample_id'], a['inputs']['parameter_sha256'], a['inputs']['initial_sha256']) != (b['sample_id'], b['inputs']['parameter_sha256'], b['inputs']['initial_sha256']):
            raise ValueError('G6/G7 common physical fit differs')
    zero = {r['sample_id']:r for r in v.read(roots['G0']/'000000_student.json')}
    final = {g:{r['sample_id']:r for r in v.read(root/('%06d_student.json'%last))} for g, root in roots.items()}
    if any(set(x) != set(zero) for x in final.values()): raise ValueError('Final evaluation coverage differs')
    summary, paired = {}, []; target = set(protocol['order'])
    for group, records in final.items():
        summary[group] = {}
        for split, ids in [('target', target), ('source_val', set(zero)-target)]:
            delta = np.array([records[n]['psnr']-zero[n]['psnr'] for n in sorted(ids)])
            summary[group][split] = dict(count=len(ids), psnr=float(np.mean([records[n]['psnr'] for n in ids])),
                ssim=float(np.mean([records[n]['ssim'] for n in ids])), mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
                improved=int((delta>0).sum()), worse=int((delta<0).sum()), worse_over_1db=int((delta < -1).sum()), worst_delta=float(delta.min()))
    for n in sorted(zero):
        row = dict(sample_id=n, split='target' if n in target else 'source_val', source_psnr=zero[n]['psnr'], **{g:r[n]['psnr'] for g,r in final.items()})
        row.update(G6_minus_G5=row['G6']-row['G5'], G7_minus_G5=row['G7']-row['G5'], G7_minus_G6=row['G7']-row['G6']); paired.append(row)
    out.json('comparison/summary.json', summary); out.put('comparison/paired.csv', v.csv_bytes(paired))
    out.json('comparison/status.json', dict(status='complete', common_fit_exact=True, note='G0-G5 reused; no reference-based fit/stopping'))
    print('G0-G7 comparison COMPLETE', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment_dir', 'diagnostic_dir', 'factorial_dir', 'strength_dir', 'visual_dir', 'pilot_dir', 'output_dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--max_gb', type=float, default=10.)
    parser.add_argument('--phase', choices=['preflight', 'accept', 'run', 'compare', 'all'], default='preflight')
    parser.add_argument('--group', choices=list(GROUPS)); args = parser.parse_args(argv)
    if not 0 < args.max_gb <= 10: parser.error('max_gb must be in (0,10]')
    if args.phase == 'run' and not args.group: parser.error('run requires group')
    if args.phase == 'preflight': preflight(args)
    elif args.phase == 'accept': acceptance(args)
    elif args.phase == 'run': run_group(args, args.group)
    elif args.phase == 'compare': compare(args)
    else:
        acceptance(args)
        for group in GROUPS: run_group(args, group)
        compare(args)


if __name__ == '__main__': main()
