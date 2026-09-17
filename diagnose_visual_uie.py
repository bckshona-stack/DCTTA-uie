"""G0 整图诊断：精确重放原轨迹，导出教师伪标签、x_sd、y_sd。"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_bytes(rows):
    text = io.StringIO(newline='')
    if rows:
        writer = csv.DictWriter(text, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return text.getvalue().encode('utf-8-sig')


class Output:
    """所有写入均受容量约束；只替换同一路径，不删除文件。"""
    def __init__(self, root, limit):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = limit

    def used(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def put(self, name, blob):
        path = self.root / name
        if self.root not in path.resolve().parents:
            raise ValueError('Output path escapes root')
        pending = path.with_name(path.name + '.pending')
        # 临时文件也计入峰值；成功后以原子替换提交。
        previous = pending.stat().st_size if pending.exists() else 0
        if self.used() - previous + len(blob) > self.limit:
            raise RuntimeError('Output exceeds 10 GB/budget; stop without deleting or resizing')
        if shutil.disk_usage(self.root).free < len(blob) + 1024 * 1024:
            raise RuntimeError('Insufficient disk space')
        path.parent.mkdir(parents=True, exist_ok=True)
        pending.write_bytes(blob)
        os.replace(pending, path)
        return sha(path)

    def json(self, name, value):
        return self.put(name, json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8'))


def independent_output(output, originals):
    target = Path(output).resolve()
    for value in originals:
        original = Path(value).resolve()
        if target == original or original in target.parents or target in original.parents:
            raise ValueError('Use a separate sibling output directory, outside original results')


def schedule(n):
    if n < 1:
        raise ValueError('Empty target set')
    return sorted({0, *[math.ceil(n * k / 10) for k in range(1, 11)]})


def inventory(diag, experiment=None):
    diag = Path(diag)
    protocol = read(diag / 'protocol.json')
    order = protocol['order']
    if len(set(order)) != len(order):
        raise ValueError('Duplicate target IDs')
    def indexed(path):
        rows = read(path)
        result = {r['sample_id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate evaluation IDs')
        return result
    source = indexed(diag / 'evaluation/000000_student.json')
    if experiment is not None:
        report = read(Path(experiment)/'source_only/run.json')
        if report['status'] != 'complete' or report['configuration']['mode'] != 'baseline':
            raise ValueError('Completed source_only baseline required')
        records = report['evaluation_records']
        source = {r['image']:r for r in records}
        if len(source) != len(records) or set(source) != set(order):
            raise ValueError('source_only target identities differ')
    final = indexed(diag / ('evaluation/%06d_student.json' % len(order)))
    dims = {r['id']: r for r in protocol['target']['samples']}
    rows = []
    for name in sorted(order):
        if any(c in name for c in '/\\:') or name in ('.', '..'):
            raise ValueError('sample_id must be a plain filename')
        a, b = float(source[name]['psnr']), float(final[name]['psnr'])
        if not all(math.isfinite(v) for v in (a, b)):
            raise ValueError('Selection requires finite source/G0 PSNR')
        rows.append(dict(sample_id=name, reason='', width=int(dims[name]['width']),
                         height=int(dims[name]['height']), source_psnr=a,
                         g0_psnr=b, delta_psnr=b-a))
    if len({r['sample_id'].casefold() for r in rows}) != len(rows):
        raise ValueError('Case-insensitive filename collision')
    output_names=[(r['sample_id'] if r['sample_id'].lower().endswith('.png') else r['sample_id']+'.png').casefold() for r in rows]
    if len(set(output_names)) != len(rows):
        raise ValueError('PNG output filename collision')
    return protocol, rows


def select_ten(rows):
    if len(rows) < 10:
        raise ValueError('At least 10 target images required')
    chosen, used = [], set()
    best = sorted(rows, key=lambda r:(-r['delta_psnr'], r['sample_id']))[:2]
    reserved = {r['sample_id'] for r in best}
    def take(candidates, count, reason):
        for row in candidates:
            if row['sample_id'] in used:
                continue
            chosen.append(dict(row, reason=reason(row)))
            used.add(row['sample_id'])
            count -= 1
            if count == 0:
                return
    take(sorted([r for r in rows if r['sample_id'] not in reserved], key=lambda r:(r['delta_psnr'], r['sample_id'])), 3,
         lambda r:'最大退化' if r['delta_psnr'] < 0 else '最低增益（无足够退化）')
    take(sorted([r for r in rows if r['sample_id'] not in reserved], key=lambda r:(abs(r['delta_psnr']), r['sample_id'])), 3, lambda r:'变化最小')
    take(sorted(rows, key=lambda r:(-r['delta_psnr'], r['sample_id'])), 2,
         lambda r:'最大改善' if r['delta_psnr'] > 0 else '下降最少（改善样本不足）')
    take(sorted(rows, key=lambda r:(-r['width']*r['height'], r['sample_id'])), 1, lambda r:'剩余最大分辨率')
    take(sorted(rows, key=lambda r:(r['width']*r['height'], r['sample_id'])), 1, lambda r:'剩余最小分辨率')
    return chosen


def selection(path, rows):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        selected = list(csv.DictReader(f))
    ids = [r['sample_id'] for r in selected]
    lookup = {r['sample_id']: r for r in rows}
    if len(ids) != 10 or len(set(ids)) != 10 or not set(ids) <= set(lookup):
        raise ValueError('CSV must select exactly 10 unique target sample_id values')
    # 修改选择时只信任 sample_id/reason；尺寸和分数从原结果重新获取。
    return [dict(lookup[r['sample_id']], reason=r.get('reason') or '手动选择') for r in selected]


def estimate(rows, chosen, stages):
    def png_bound(r):
        # RGB8 每行滤波字节，预留压缩流/PNG块的保守开销。
        return math.ceil((3*r['width']+1)*r['height']*1.01)+65536
    images = 5*len(rows) + 3*len(chosen)*len(stages)
    upper = 5*sum(map(png_bound, rows))+3*len(stages)*sum(map(png_bound, chosen))
    return dict(images=images, png_conservative_bytes=upper, metadata_reserve_bytes=20_000_000,
                estimated_bytes=upper+20_000_000, snapshot_steps=stages)


def prepare(args):
    independent_output(args.output_dir, [args.experiment_dir, args.diagnostic_dir])
    output = Output(args.output_dir, int(args.max_gb*1_000_000_000))
    if not (output.root/'run_manifest.json').exists():
        unexpected = [p.name for p in output.root.iterdir() if p.name not in {'selected_samples.csv','storage_estimate.json'}]
        if unexpected:
            raise FileExistsError('Output contains unrelated files; choose a new directory: '+str(unexpected))
    protocol, rows = inventory(args.diagnostic_dir, args.experiment_dir)
    path = output.root / 'selected_samples.csv'
    if args.selection_csv:
        chosen = selection(args.selection_csv, rows)
    elif path.exists():
        chosen = selection(path, rows)
    else:
        chosen = select_ten(rows)
    stages = schedule(len(rows))
    size = estimate(rows, chosen, stages)
    if size['estimated_bytes'] > output.limit:
        raise RuntimeError('Conservative output estimate exceeds budget; no resizing/deletion performed')
    if (output.root / 'run_manifest.json').exists():
        saved = read(output.root / 'run_manifest.json')
        if saved['selected'] != chosen:
            raise ValueError('Selection frozen after preflight; use a new output directory')
        if sha(path) != saved['selection_csv_sha256']:
            raise ValueError('Selection CSV changed after preflight')
    else:
        if not path.exists() or args.selection_csv:
            output.put('selected_samples.csv', csv_bytes(chosen))
    output.json('storage_estimate.json', size)
    print('Selection ready: %d images; %.3f GB conservative estimate' %
          (size['images'], size['estimated_bytes']/1e9), flush=True)
    return output, protocol, rows, chosen, stages


def runtime():
    global torch, F, np, Image, d, u, tqdm
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    import diagnose_uie as d
    import uie_tta as u
    from tqdm import tqdm


def preflight(args):
    output, protocol, rows, chosen, stages = prepare(args)
    runtime()
    d.require_strict()
    if protocol.get('diagnostic_protocol') != 'fixed_fisher_strict':
        raise ValueError('Original strict v2 G0 required')
    if protocol['environment'] != d.environment():
        raise ValueError('Keep the original strict v2 machine/environment/code/numerical settings')
    for phase in ('run', 'evaluation'):
        if read(Path(args.diagnostic_dir)/phase/'status.json')['status'] != 'complete':
            raise ValueError('Original G0 incomplete')
    d.verify_fixed_files(args.experiment_dir)
    source_report = read(Path(args.experiment_dir)/'source_only/run.json')
    if source_report['source_checkpoint_sha256'] != sha(Path(args.experiment_dir)/'source/checkpoints/best.pth'):
        raise ValueError('source_only checkpoint differs')
    if source_report['metrics'] != 'RGB float clip[0,1], data_range=1, skimage defaults':
        raise ValueError('source_only metric protocol differs')
    groups, _, _ = d.inputs(args.experiment_dir)
    samples = {s.sample_id:s for s in groups['target_eval']}
    if set(samples) != set(protocol['order']):
        raise ValueError('Target set differs')
    for row in rows:
        s = samples[row['sample_id']]
        with Image.open(s.raw_path) as raw, Image.open(s.reference_path) as ref:
            if raw.size != ref.size or raw.size != (row['width'],row['height']):
                raise ValueError('Full image dimensions differ')
    root = Path(args.diagnostic_dir)
    paths = ['protocol.json','run/trajectory.json','run/replay_states/000000.pth',
             'evaluation/000000_student.json','evaluation/%06d_student.json' % len(rows)]
    initial = u.load_checkpoint(root/'run/replay_states/000000.pth')
    acceptance = root/'acceptance/initial.pth'
    if d.digest(initial) != d.digest(u.load_checkpoint(acceptance)):
        raise ValueError('Original initial state differs from acceptance')
    manifest = dict(version=2, entry_sha256=sha(__file__), selected=chosen,
        selection_csv_sha256=sha(output.root/'selected_samples.csv'), snapshot_steps=stages,
        experiment_dir=str(Path(args.experiment_dir).resolve()), diagnostic_dir=str(root.resolve()),
        device=args.device, max_bytes=output.limit, inputs={p:sha(root/p) for p in paths},
        acceptance_sha256=sha(acceptance), environment=d.environment(),
        source_only_report_sha256=sha(Path(args.experiment_dir)/'source_only/run.json'),
        definition='full-frame eval; raw-conditioned RDDM EMA sampling only; no diagnostic generator training',
        psnr='RGB clipped float32 before PNG; data_range=1; additionally PNG-quantized PSNR',
        seed_policy='sha256(original_seed:sample_id), constant across stages',
        diagnostic_attention=dict(mode='full_keys_query_chunks',chunk_size=128,threshold=1024))
    del initial
    if (output.root/'run_manifest.json').exists():
        if read(output.root/'run_manifest.json') != json.loads(json.dumps(manifest)):
            raise ValueError('Run contract changed; use a new output directory')
    else:
        output.json('run_manifest.json', manifest)
    d.require_space(output.root, estimate(rows, chosen, stages)['estimated_bytes'])
    print('Preflight PASSED', flush=True)
    return output, protocol, rows, chosen, stages, samples


@contextmanager
def isolated_diagnostic(t, g, cursor, order):
    state = d.capture(t,g,cursor,order)
    before = d.digest(state)
    tr = g.trainer
    old_save = tr.save_results
    # test() 改动扩散 buffer；完整恢复也包含 buffer 原设备、模式与梯度。
    try:
        tr.save_results = False
        t.model.eval()
        t.model_teacher.eval()
        yield
    finally:
        tr.save_results = old_save
        d.restore(t,g,state)
        if d.digest(d.capture(t,g,cursor,order)) != before:
            raise RuntimeError('Diagnostic changed full adaptation state')


def sample_seed(seed, name):
    return int.from_bytes(hashlib.sha256(('%s:%s'%(seed,name)).encode()).digest()[:4], 'big')


def query_chunk_attention(module, x, chunk_size=128):
    """仅诊断前向：分批查询，但每个查询仍对全部 key 做同一个 softmax。"""
    if torch.is_grad_enabled():
        raise RuntimeError('Chunked attention is diagnostic inference only')
    b, _, height, width = x.shape
    n = height * width
    qkv = module.to_qkv(x).chunk(3, dim=1)
    q, k, v = [value.reshape(b, module.heads, -1, n) for value in qkv]
    q = q * module.scale
    out = x.new_empty((b, module.heads, n, v.shape[2]))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = torch.einsum('b h d i, b h d j -> b h i j', q[..., start:end], k)
        attn = sim.softmax(dim=-1)
        del sim
        out[:, :, start:end] = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        del attn
    return module.to_out(out.permute(0, 1, 3, 2).reshape(b, -1, height, width))


@contextmanager
def bounded_sampling_attention(generator, chunk_size=128, threshold=1024):
    """只临时替换生成器 EMA 的前向，不改源码、权重或真实适配路径。"""
    from types import MethodType
    from RDDM.src.residual_denoising_diffusion_pytorch import Attention
    changed = []
    try:
        for module in generator.trainer.ema.ema_model.modules():
            if isinstance(module, Attention):
                previous = module.__dict__.get('forward')
                original = module.forward
                def forward(this, x, original=original):
                    if x.shape[-2] * x.shape[-1] <= threshold:
                        return original(x)
                    return query_chunk_attention(this, x, chunk_size)
                changed.append((module, previous))
                module.forward = MethodType(forward, module)
        yield
    finally:
        for module, previous in reversed(changed):
            if previous is None:
                delattr(module, 'forward')
            else:
                module.forward = previous


def generate_full(g, raw, name, seed):
    height, width = raw.shape[-2:]
    bottom = max(32, math.ceil(height/8)*8)-height
    right = max(32, math.ceil(width/8)*8)-width
    padded = F.pad(raw,(0,right,0,bottom),mode='reflect' if bottom<height and right<width else 'replicate')
    with u.isolated_rng(seed), torch.no_grad(), bounded_sampling_attention(g):
        generated = g.sample(padded,[name])
    if generated.shape != padded.shape or not torch.isfinite(generated).all():
        raise ValueError('Invalid full-frame RDDM output')
    return generated[...,:height,:width].detach()


def psnr(pred, ref):
    # 与 utils.val_utils 相同：先 float32 截断及平方，再 float64 求均值。
    if pred.shape != ref.shape or not torch.isfinite(pred).all() or not torch.isfinite(ref).all():
        raise ValueError('Invalid PSNR inputs')
    a = pred.detach().cpu().numpy().clip(0,1)
    b = ref.detach().cpu().numpy().clip(0,1)
    mse = np.mean((a-b)**2, dtype=np.float64)
    return float('inf') if mse == 0 else float(10*np.log10(1/mse))


def export_image(output, name, tensor):
    if not torch.isfinite(tensor).all() or tensor.shape[0] != 1:
        raise ValueError('Invalid image tensor')
    clipped = tensor.detach().cpu().clamp(0,1)
    pixels = (clipped[0].permute(1,2,0)*255).round().to(torch.uint8).numpy()
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    digest = output.put(name,stream.getvalue())
    quantized = torch.from_numpy(pixels.copy()).permute(2,0,1).unsqueeze(0).float()/255
    return digest, quantized


def observe(t,g,sample,source,output,prefix,seed,fixed_x=None,prior=None):
    # 不读取 reference，直到所有诊断前向已完成。
    with torch.no_grad():
        raw = u.read_rgb(sample.raw_path).unsqueeze(0).to(t.device)
        pseudo = u.eight_view_mean(t.model_teacher,raw)
        x = generate_full(g,raw,sample.sample_id,seed) if fixed_x is None else fixed_x.to(t.device)
        y = t.model(x)
    ref = u.read_rgb(sample.reference_path).unsqueeze(0)
    name = sample.sample_id if sample.sample_id.lower().endswith('.png') else sample.sample_id+'.png'
    files, metrics = {}, []
    for role,value in [('teacher_pseudo',pseudo),('x_sd',x),('y_sd',y)]:
        if role=='x_sd' and fixed_x is not None:
            continue
        path = '%s/%s/%s'%(role,prefix,name)
        digest,quantized = export_image(output,path,value)
        files[path]=digest
        if role!='x_sd':
            score=psnr(value,ref)
            metrics.append(dict(sample_id=sample.sample_id,role=role,path=path,
                psnr=score,png_psnr=psnr(quantized,ref),source_only_psnr=source,
                delta_source_psnr=score-source,
                delta_before_psnr='' if prior is None else score-prior[role],
                width=raw.shape[-1],height=raw.shape[-2],sampling_seed=seed,
                x_sd_tensor_sha256=d.digest(x.detach().cpu())))
    return x.detach().cpu(),metrics,files


def verify_files(output, files):
    for name,digest in files.items():
        if sha(output.root/name)!=digest:
            raise ValueError('Committed output changed: '+name)


def execute(args):
    output,protocol,rows,chosen,stages,samples = preflight(args)
    manifest_hash=sha(output.root/'run_manifest.json')
    progress_path=output.root/'progress.json'
    progress=read(progress_path) if progress_path.exists() else dict(
        cursor=0,initial_snapshot=False,metrics=[],files={},trajectory=[],manifest_sha256=manifest_hash)
    if progress['manifest_sha256'] != manifest_hash:
        raise ValueError('Resume manifest differs')
    verify_files(output,progress['files'])
    if (output.root/'status.json').exists():
        status=read(output.root/'status.json')
        if status.get('status')=='complete':
            verify_files(output,status['files'])
            print('Already COMPLETE; skip all inference/adaptation',flush=True)
            return
    # 不新增大状态文件；恢复已有 G0 边界，重放到已提交输出游标。
    bootstrap=output.root/'bootstrap'/uuid.uuid4().hex
    fisher_size=(Path(args.experiment_dir)/'dctta/fisher.pt').stat().st_size
    if output.used()+fisher_size+20_000_000>output.limit:
        raise RuntimeError('Bootstrap exceeds budget')
    t,g,data,order,_,opt=d.build(args.experiment_dir,args.device,bootstrap,fixed_fisher=True)
    if order!=protocol['order'] or vars(opt)!=protocol['config'] or data.identity()!=protocol['target']:
        raise ValueError('Original configuration/order/crops differ')
    original=read(Path(args.diagnostic_dir)/'run/trajectory.json')
    if len(original)!=len(order) or [r['step'] for r in original]!=list(range(1,len(order)+1)):
        raise ValueError('Incomplete original trajectory')
    cursor=progress['cursor']
    if not 0<=cursor<=len(order):
        raise ValueError('Invalid resume cursor')
    boundaries={int(p.stem):p for p in (Path(args.diagnostic_dir)/'run/replay_states').glob('*.pth')}
    start=max(k for k in boundaries if k<=cursor)
    state=u.load_checkpoint(boundaries[start])
    expected=original[start-1]['state_sha256'] if start else d.digest(u.load_checkpoint(Path(args.diagnostic_dir)/'acceptance/initial.pth'))
    if d.digest(state)!=expected:
        raise ValueError('Original replay boundary changed')
    d.restore(t,g,state)
    del state
    if d.digest(d.capture(t,g,start,order))!=expected:
        raise ValueError('Replay restore differs')
    by_id={s.sample_id:i for i,s in enumerate(data.samples)}
    sources={r['sample_id']:r['source_psnr'] for r in rows}

    def adapt(i):
        name,image=data[by_id[order[i]]]
        loss=t.adapt(image.unsqueeze(0),[name],g)
        d.require_strict()
        digest=d.digest(d.capture(t,g,i+1,order))
        if digest!=original[i]['state_sha256'] or loss!=original[i]['losses']:
            raise ValueError('Exact G0 replay failed at step %d; stop'% (i+1))
        return dict(step=i+1,sample_id=name,losses=loss,state_sha256=digest)

    def snapshots(step):
        with isolated_diagnostic(t,g,step,order):
            for row in tqdm(chosen,desc='Snapshot %d'%step,unit='img'):
                name=row['sample_id']
                _,metrics,files=observe(t,g,samples[name],sources[name],output,
                    'snapshots/%06d'%step,sample_seed(opt.seed,name))
                progress['metrics'].extend(dict(r,kind='snapshot',step=step,timing='stage') for r in metrics)
                progress['files'].update(files)

    def commit():
        output.put('metrics.csv',csv_bytes(progress['metrics']))
        output.json('progress.json',progress)

    for i in tqdm(range(start,cursor),desc='Resume replay',unit='img'):
        adapt(i)
    if not progress['initial_snapshot']:
        if cursor!=0:
            raise ValueError('Missing initial snapshot')
        snapshots(0)
        progress['initial_snapshot']=True
        commit()
    for i in tqdm(range(cursor,len(order)),initial=cursor,total=len(order),desc='G0 visual diagnosis',unit='img'):
        name=order[i]; seed=sample_seed(opt.seed,name)
        with isolated_diagnostic(t,g,i,order):
            x,before,files=observe(t,g,samples[name],sources[name],output,'online/%06d/before'%(i+1),seed)
        trajectory=adapt(i)
        with isolated_diagnostic(t,g,i+1,order):
            _,after,after_files=observe(t,g,samples[name],sources[name],output,
                'online/%06d/after'%(i+1),seed,fixed_x=x,prior={r['role']:r['psnr'] for r in before})
        if {r['x_sd_tensor_sha256'] for r in before+after} != {d.digest(x)}:
            raise ValueError('Before/after x_sd differs')
        del x
        progress['files'].update(files); progress['files'].update(after_files)
        for timing,metrics in [('before',before),('after',after)]:
            progress['metrics'].extend(dict(r,kind='online',step=i+1,timing=timing) for r in metrics)
        progress['trajectory'].append(trajectory)
        if i+1 in stages:
            snapshots(i+1)
        progress['cursor']=i+1
        commit()
        print('Committed %d/%d; output %.3f GB'%(i+1,len(order),output.used()/1e9),flush=True)
    expected_images=estimate(rows,chosen,stages)['images']
    if len(progress['files'])!=expected_images:
        raise ValueError('Image coverage incomplete')
    verify_files(output,progress['files'])
    summary=[]
    for step in stages:
        for role in ['teacher_pseudo','y_sd']:
            subset=[r for r in progress['metrics'] if r['kind']=='snapshot' and r['step']==step and r['role']==role]
            summary.append(dict(step=step,role=role,count=len(subset),
                mean_psnr=sum(r['psnr'] for r in subset)/len(subset),
                mean_delta_source_psnr=sum(r['delta_source_psnr'] for r in subset)/len(subset)))
    output.put('snapshot_summary.csv',csv_bytes(summary))
    output.json('trajectory.json',progress['trajectory'])
    output.json('status.json',dict(status='complete',images=expected_images,steps=len(order),
        metric_rows=len(progress['metrics']),exact_g0_replay=True,files={
            p:sha(output.root/p) for p in ['metrics.csv','snapshot_summary.csv','trajectory.json','progress.json']}))
    print('Visual diagnosis COMPLETE',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['experiment_dir','diagnostic_dir','output_dir']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--phase',choices=['select','preflight','run'],default='select')
    parser.add_argument('--selection_csv')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max_gb',type=float,default=10.)
    args=parser.parse_args(argv)
    if not 0<args.max_gb<=10:
        parser.error('--max_gb must be in (0,10]')
    if args.phase=='select':
        prepare(args)
        return
    runtime()
    if os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8') not in [':4096:8',':16:8']:
        raise ValueError('Invalid CUBLAS_WORKSPACE_CONFIG')
    torch.use_deterministic_algorithms(True,warn_only=False)
    u.seed_everything(23)
    if args.phase=='preflight':
        preflight(args)
    else:
        execute(args)


if __name__=='__main__':
    main()
