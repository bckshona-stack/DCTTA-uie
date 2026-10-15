"""UIEB4 新源模型 source_only + 严格 G0；不依赖旧蓝色域产物。"""
import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import diagnose_uie as d
import diagnose_visual_uie as v
import run_strength_uie as s
import uie_tta as u
from options import parse_args
from prepare_uieb4_experiment import CATEGORIES, ROOT, sha, read_csv, verify_runtime
from train_source_uie import checked_manifest


def selection(rows):
    chosen=[]
    for c in CATEGORIES:
        pool=sorted([r for r in rows if r['primary_category']==c],
                    key=lambda r:(int(r['width'])*int(r['height']),r['sample_id']))
        chosen.extend([pool[0]['sample_id'],pool[-1]['sample_id']])
    rest=sorted([r for r in rows if r['sample_id'] not in chosen],
                key=lambda r:sha_name(r['sample_id']))
    chosen.extend(r['sample_id'] for r in rest[:10-len(chosen)])
    if len(chosen)!=10 or len(set(chosen))!=10: raise ValueError('Need ten unique diagnostic images')
    return chosen


def sha_name(name):
    import hashlib
    return hashlib.sha256(name.encode()).hexdigest()


def aggregate(rows, metadata):
    result={}
    for category in ('all',*CATEGORIES):
        selected=[r for r in rows if category=='all' or metadata[r['sample_id']]['primary_category']==category]
        if selected:
            result[category]=dict(count=len(selected),**{k:sum(r[k] for r in selected)/len(selected) for k in ('psnr','ssim')})
    result['macro_category']={k:sum(result[c][k] for c in CATEGORIES)/len(CATEGORIES) for k in ('psnr','ssim')}
    return result


def inputs(exp):
    folder=exp/'manifests'
    audit=v.read(folder/'audit.json')
    if audit['protocol']!='UIEB4_light_to_severe_v1': raise ValueError('New UIEB4 manifests required')
    groups={k:checked_manifest(folder,k+'.csv',audit) for k in ('source_train','source_val','target_eval')}
    rows=read_csv(folder/'target_eval.csv')
    checkpoint=exp/'source/checkpoints/best.pth'
    payload=u.load_checkpoint(checkpoint)
    if payload.get('artifact_type')!='source_training': raise ValueError('Trained source best required')
    contract=payload['training_contract']
    if contract['data']!={n:audit['files'][n] for n in ('source_train.csv','source_val.csv')}:
        raise ValueError('Source checkpoint belongs to different source manifests')
    last=u.load_checkpoint(exp/'source/checkpoints/last.pth')
    if last['epoch']!=contract['epochs'] or last['training_contract']!=contract or last['best_epoch']!=payload['epoch']:
        raise ValueError('Complete source training before G0; keep matching best and last')
    del payload,last
    return groups,rows,checkpoint


def build(args, out, groups, checkpoint):
    from RDDM.net import ResidualDiffusionModel
    opt=parse_args([])
    opt.device=str(u.resolve_device(args.device)); opt.backbone='nu2net'
    opt.model_path=str(checkpoint); opt.verbose=False; opt.save_rddm_results=False
    u.seed_everything(opt.seed)
    with u.isolated_rng(opt.seed): model=u.build_backbone('nu2net',checkpoint,opt.device)
    u.configure_model(opt,model)
    t=u.SRTTA(opt,model,torch.optim.Adam(u.collect_params(model)[0],lr=opt.lr,betas=tuple(opt.betas)))
    data=u.TargetRawDataset(groups['target_eval'],opt.patch_size,opt.seed)
    t.fisher_identity=t.cache_identity(data)
    fisher=out.root/'fisher.pt'
    if fisher.exists(): t.load_fisher(fisher)
    else:
        print('Computing new target Fisher (raw only)',flush=True)
        t.compute_fisher(u.make_raw_loader(data,0,False,opt.seed))
        out.checkpoint('fisher.pt',dict(identity=t.fisher_identity,fishers=d.cpu_tree(t.fishers)))
    t.fisher_origin=dict(policy='new_UIEB4_fixed',sha256=sha(fisher))
    with u.isolated_rng(opt.seed+1):
        g=ResidualDiffusionModel(opt,image_size=opt.patch_size,device=opt.device,results_folder=out.root/'rddm')
    order=[names[0] for names,_ in u.make_raw_loader(data,0,True,opt.seed)]
    return t,g,data,order,opt


def adapt(t,g,data,order,index):
    by_id={sample.sample_id:i for i,sample in enumerate(data.samples)}
    name,image=data[by_id[order[index]]]
    start=time.perf_counter()
    loss=t.adapt(image.unsqueeze(0).to(t.device),[name],g)
    return dict(step=index+1,sample_id=name,losses=loss,seconds=time.perf_counter()-start)


def signature(t,g,cursor,order): return d.digest(d.capture(t,g,cursor,order))


def acceptance(t,g,data,order,opt,out,chosen,groups,contract):
    path=out.root/'acceptance.json'
    if path.exists():
        report=v.read(path)
        if report.get('contract')!=contract or not report.get('passed'): raise ValueError('Acceptance identity differs')
        return
    # 必须覆盖小图/320裁剪及EMA warmup，不改变正式随机顺序。
    n=min(22,len(order))
    while n<=len(order):
        try:
            d.acceptance_length(n,data,order,opt,g.trainer.ema,g.trainer.train_num_steps)
            break
        except ValueError:
            n+=1
    if n>len(order): raise ValueError('No valid mixed-size acceptance prefix')
    initial=u.load_checkpoint(out.root/'initial.pth')
    expected=[]; boundary=None
    for i in range(n):
        adapt(t,g,data,order,i)
        expected.append(signature(t,g,i+1,order))
        if i==0: boundary=d.capture(t,g,1,order)
        print('Acceptance control %d/%d'%(i+1,n),flush=True)
    d.restore(t,g,initial)
    for i in range(n):
        adapt(t,g,data,order,i)
        if i in (0,n-1):
            with v.isolated_diagnostic(t,g,i+1,order),torch.no_grad():
                sample=next(x for x in groups['target_eval'] if x.sample_id==chosen[-1])
                raw=u.read_rgb(sample.raw_path).unsqueeze(0).to(t.device)
                x=v.generate_full(g,raw,sample.sample_id,v.sample_seed(opt.seed,sample.sample_id))
                t.model(x);t.model(raw);u.eight_view_mean(t.model_teacher,raw)
        if signature(t,g,i+1,order)!=expected[i]: raise RuntimeError('Repeat/observation acceptance failed')
        print('Acceptance repeat %d/%d'%(i+1,n),flush=True)
    d.restore(t,g,boundary)
    for i in range(1,n):
        adapt(t,g,data,order,i)
        if signature(t,g,i+1,order)!=expected[i]: raise RuntimeError('Resume acceptance failed')
    d.restore(t,g,initial)
    out.json('acceptance.json',dict(passed=True,steps=n,contract=contract,
        repeat_and_observation_equal=True,resume_equal=True))
    print('Strict acceptance PASSED',flush=True)


def observe(t,g,groups,chosen,step,order,opt,out,baseline):
    records=[]; files={}
    lookup={x.sample_id:x for x in groups['target_eval']}
    with v.isolated_diagnostic(t,g,step,order),torch.no_grad():
        for name in chosen:
            _,metrics,exported=v.observe(t,g,lookup[name],baseline[name],out,
                'snapshots/%06d'%step,v.sample_seed(opt.seed,name))
            records.extend(metrics);files.update(exported)
        evaluation={k:d.score(t.model,groups[k]) for k in ('target_eval','source_val')}
        teacher=d.score(t.model_teacher,groups['target_eval'],ensemble=True)
    files['evaluation/%06d.json'%step]=out.json('evaluation/%06d.json'%step,
        dict(student=evaluation,teacher=teacher,image_metrics=records))
    files['snapshots/%06d.pth'%step]=out.checkpoint('snapshots/%06d.pth'%step,
        dict(student=d.cpu_tree(t.model.state_dict()),teacher=d.cpu_tree(t.model_teacher.state_dict())))
    return files


def commit(out,state,progress):
    slot=state['cursor']%2
    name='resume_%d.pth'%slot
    digest=out.checkpoint(name,dict(state=state,progress=progress))
    out.json('progress.json',dict(checkpoint=name,sha256=digest,cursor=state['cursor']))


def execute(args):
    verify_runtime()
    if args.max_gb<=0 or args.max_gb>10: raise ValueError('Budget must be in (0,10] GB')
    v.runtime();torch.use_deterministic_algorithms(True,warn_only=False)
    u.resolve_device(args.device)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    exp=Path(args.experiment_dir).resolve()
    groups,rows,checkpoint=inputs(exp)
    chosen=selection(rows)
    # 最大图作为诊断验收样本，正式选择不根据目标GT或评分。
    largest=max(rows,key=lambda r:int(r['width'])*int(r['height']))['sample_id']
    if largest not in chosen: chosen[-1]=largest
    else: chosen.remove(largest);chosen.append(largest)
    out=s.BudgetOutput(exp/'G0',int(args.max_gb*1e9))
    stages=v.schedule(len(rows))
    def bound(r): return math.ceil((3*int(r['width'])+1)*int(r['height'])*1.01)+65536
    selected=[r for r in rows if r['sample_id'] in chosen]
    # 初态+双槽成熟状态+原子保存临时副本，阶段双模型权重及元数据。
    estimate=4_800_000_000+4*len(stages)*sum(map(bound,selected))+2*sum(map(bound,rows))
    if estimate>out.limit: raise RuntimeError('Conservative G0 peak exceeds budget: %.3f GB'%(estimate/1e9))
    tracked=('run_uieb4_g0.py','prepare_uieb4_experiment.py','uie_tta.py','options.py',
             'train_source_uie.py','diagnose_uie.py','diagnose_visual_uie.py','run_strength_uie.py',
             'run_factorial_uie.py','uie_backbones/nu2net.py','RDDM/net.py','RDDM/src/residual_denoising_diffusion_pytorch.py')
    contract=dict(version=1,source=sha(checkpoint),manifests=sha(exp/'manifests/audit.json'),
        implementation={n:sha(ROOT/n) for n in tracked},environment=d.environment(),numerics=d.numerics(),
        device=args.device,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        chosen=chosen,stages=stages,max_bytes=out.limit,estimated_peak_bytes=estimate)
    path=out.root/'contract.json'
    if path.exists():
        if v.read(path)!=contract: raise ValueError('Code/data/source/environment changed; use a new experiment')
    elif any(out.root.iterdir()): raise FileExistsError('G0 folder must be empty for a new protocol')
    else: out.json('contract.json',contract)
    if (out.root/'complete.json').exists():
        status=v.read(out.root/'complete.json')
        v.verify_files(out,status['files'])
        print('G0 already COMPLETE',flush=True);return
    d.require_space(out.root,max(0,estimate-out.used()))
    print('G0 preflight: %d target; peak <= %.3f GB'%(len(rows),estimate/1e9),flush=True)
    if args.phase=='preflight': return
    t,g,data,order,opt=build(args,out,groups,checkpoint)
    initial_path=out.root/'initial.pth'
    if initial_path.exists():
        initial=u.load_checkpoint(initial_path)
        if signature(t,g,0,order)!=d.digest(initial): raise ValueError('Initial state rebuild differs')
        del initial
    else:
        out.checkpoint('initial.pth',d.capture(t,g,0,order))
    acceptance(t,g,data,order,opt,out,chosen,groups,contract)
    if args.phase=='accept': return
    metadata={r['sample_id']:r for r in rows}
    baseline_files={}
    with v.isolated_diagnostic(t,g,0,order),torch.no_grad():
        baseline=d.score(t.model,groups['target_eval'])
        baseline_by_id={r['sample_id']:r['psnr'] for r in baseline}
        for sample in groups['target_eval']:
            raw=u.read_rgb(sample.raw_path).unsqueeze(0).to(t.device)
            name='source_only/'+s.image_name(sample.sample_id)
            digest,_=v.export_image(out,name,t.model(raw));baseline_files[name]=digest
    baseline_files['source_only.json']=out.json('source_only.json',dict(records=baseline,summary=aggregate(baseline,metadata)))
    pointer=out.root/'progress.json'
    if pointer.exists():
        p=v.read(pointer)
        if p['checkpoint'] not in ('resume_0.pth','resume_1.pth') or sha(out.root/p['checkpoint'])!=p['sha256']:
            raise ValueError('Committed resume checkpoint differs')
        payload=u.load_checkpoint(out.root/p['checkpoint'])
        progress=payload['progress'];state=payload['state']
        if state['order']!=order or state['cursor']!=p['cursor']: raise ValueError('Resume cursor/order mismatch')
        v.verify_files(out,progress['files']);d.restore(t,g,state)
        cursor=state['cursor'];del payload,state
    else:
        progress=dict(files={**baseline_files,**observe(t,g,groups,chosen,0,order,opt,out,baseline_by_id)},trajectory=[])
        commit(out,d.capture(t,g,0,order),progress);cursor=0
    for i in range(cursor,len(order)):
        row=adapt(t,g,data,order,i)
        progress['trajectory'].append(row)
        if i+1 in stages: progress['files'].update(observe(t,g,groups,chosen,i+1,order,opt,out,baseline_by_id))
        commit(out,d.capture(t,g,i+1,order),progress)
        print('G0 committed %d/%d; %.3f GB'%(i+1,len(order),out.used()/1e9),flush=True)
    with v.isolated_diagnostic(t,g,len(order),order),torch.no_grad():
        final=d.score(t.model,groups['target_eval'])
        for sample in groups['target_eval']:
            raw=u.read_rgb(sample.raw_path).unsqueeze(0).to(t.device)
            path='final_y_in/'+s.image_name(sample.sample_id)
            digest,_=v.export_image(out,path,t.model(raw));progress['files'][path]=digest
    before={r['sample_id']:r for r in baseline}
    difference=[dict(sample_id=r['sample_id'],category=metadata[r['sample_id']]['primary_category'],
        source_psnr=before[r['sample_id']]['psnr'],g0_psnr=r['psnr'],delta_psnr=r['psnr']-before[r['sample_id']]['psnr'],
        source_ssim=before[r['sample_id']]['ssim'],g0_ssim=r['ssim'],delta_ssim=r['ssim']-before[r['sample_id']]['ssim']) for r in final]
    progress['files']['comparison.csv']=out.put('comparison.csv',v.csv_bytes(difference))
    progress['files']['comparison.json']=out.json('comparison.json',dict(source_only=aggregate(baseline,metadata),
        G0=aggregate(final,metadata),per_image=difference))
    progress['files']['trajectory.json']=out.json('trajectory.json',progress['trajectory'])
    progress['files']['adapted_model.pth']=out.checkpoint('adapted_model.pth',dict(
        net=d.cpu_tree(t.model.backbone.state_dict()),model_config=t.model.model_config,
        backbone='nu2net',artifact_type='adapted_inference_only',source_sha256=sha(checkpoint),
        adaptation_steps=t.steps))
    out.json('complete.json',dict(status='complete',files=progress['files'],steps=len(order)))
    print('source_only / G0 COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment_dir',default=str(ROOT/'experiments/UIEB4_light_to_severe_NU2Net_v1'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max_gb',type=float,default=10)
    p.add_argument('--phase',choices=('preflight','accept','run'),default='run')
    args=p.parse_args();execute(args)


if __name__=='__main__': main()
