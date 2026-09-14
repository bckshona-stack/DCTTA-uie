"""G1/G2/G3 路径对照。复用 v2 G0，不修改其实现，不重复已提交的工作。"""
import argparse
import gc
import json
import os
import uuid
from pathlib import Path

import torch
from tqdm import tqdm
import diagnose_uie as d
import uie_tta as u
from train_source_uie import save_checkpoint

GROUPS={'G1':(True,False),'G2':(False,True),'G3':(True,True)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(pending,path)


def contract(args):
    old=Path(args.diagnostic_dir); protocol=read(old/'protocol.json')
    d.require_strict(); d.verify_fixed_files(args.experiment_dir)
    if protocol['environment']!=d.environment(): raise ValueError('Keep original v2 code, environment and numerical settings')
    if protocol.get('diagnostic_protocol')!='fixed_fisher_strict': raise ValueError('Strict v2 required')
    for phase in ['run','evaluation']:
        if read(old/phase/'status.json')['status']!='complete': raise ValueError('G0 incomplete')
    value=dict(entry_sha256=u.sha256_file(__file__),protocol_sha256=u.sha256_file(old/'protocol.json'),
        initial_sha256=u.sha256_file(old/'run/replay_states/000000.pth'),environment=d.environment(),
        experiment_dir=str(Path(args.experiment_dir).resolve()),diagnostic_dir=str(old.resolve()),
        groups=GROUPS,order=protocol['order'],config=protocol['config'],probes=protocol['probes'],
        g0_evaluation_sha256={p.name:u.sha256_file(p) for p in sorted((old/'evaluation').glob('*_*.json'))})
    # JSON 归一化 tuple/list，保证跨进程身份一致。
    value=json.loads(json.dumps(value)); path=Path(args.output_dir)/'plan.json'
    if path.exists():
        if read(path)!=value: raise ValueError('Experiment contract changed; use a new output directory')
    else: write(path,value)
    return protocol


class GeneratorInput:
    """始终执行原 RDDM，raw 组只替换送给学生的第二路输入。"""
    def __init__(self,g,use_raw): self.g=g; self.use_raw=use_raw; self.last=None; self.calls=0
    def train(self,raw,pseudo,names):
        generated=self.g.train(raw,pseudo,names)
        self.last=dict(raw_rgb=raw.detach().mean((0,2,3)).cpu().tolist(),
            generated_rgb=generated.detach().mean((0,2,3)).cpu().tolist())
        self.calls+=1
        return raw.detach() if self.use_raw else generated


class UpdateSummary:
    def __init__(self,t): self.t=t; self.before={}; self.values={}
    def __call__(self,event,values):
        if event=='student_before':
            self.before={n:p.detach().clone() for n,p in self.t.candidate_parameters().items()}
        elif event in ['student_adam','student_restored']:
            ps=self.t.candidate_parameters()
            norm=lambda terms:float(torch.sqrt(sum(terms)))
            self.values[event+'_delta_l2']=norm([(p.detach()-self.before[n]).double().square().sum() for n,p in ps.items()])
            if event=='student_restored':
                self.values['distance_from_source_l2']=norm([(p.detach()-self.t.model_state[n]).double().square().sum() for n,p in ps.items()])
                for n,p in ps.items():
                    mask=self.t.fishers[n][1].bool()
                    if not torch.equal(p.detach()[mask],self.t.model_state[n][mask]): raise ValueError('Fisher restoration failed')


def group_paths(args,group):
    if group not in GROUPS: raise ValueError('Only new groups G1/G2/G3 can run; G0 is read-only')
    return Path(args.output_dir)/group


def run_group(args,group):
    protocol=contract(args); folder=group_paths(args,group); folder.mkdir(parents=True,exist_ok=True)
    status_path=folder/'run_status.json'
    if status_path.exists() and read(status_path)['status']=='complete':
        for name,sha in read(status_path)['files'].items():
            if u.sha256_file(folder/name)!=sha: raise ValueError('Completed group artifact changed')
        print(f'{group}: already complete; SKIP adaptation',flush=True); return
    # 每次构建只在新 bootstrap 子目录写兼容报告，不触碰旧结果。
    t,g,data,order,_,opt=d.build(args.experiment_dir,args.device,folder/'bootstrap'/uuid.uuid4().hex,fixed_fisher=True)
    if vars(opt)!=protocol['config'] or order!=protocol['order'] or data.identity()!=protocol['target']:
        raise ValueError('Group inputs differ from G0')
    initial=u.load_checkpoint(Path(args.diagnostic_dir)/'run/replay_states/000000.pth')
    if d.digest(d.capture(t,g,0,order))!=d.digest(initial): raise ValueError('Group initial state differs from G0')
    resume=folder/'resume.pth'; rows=[]; hashes={}; start=0
    if resume.exists():
        saved=u.load_checkpoint(resume)
        if saved['plan_sha256']!=u.sha256_file(Path(args.output_dir)/'plan.json'): raise ValueError('Resume contract differs')
        state=saved['state']; rows=saved['rows']; hashes=saved['files']; start=state['cursor']
        for name,sha in hashes.items():
            if u.sha256_file(folder/name)!=sha: raise ValueError('Committed snapshot changed')
        d.restore(t,g,state)
        if d.digest(d.capture(t,g,start,order))!=d.digest(state): raise ValueError('Resume restoration differs')
        del saved,state
    else: d.restore(t,g,initial)
    fixed,use_raw=GROUPS[group]
    source_teacher=d.digest(initial['teacher']['weights']); del initial
    if fixed:
        # 只屏蔽教师 EMA，原 adapt / loss / Adam / Fisher 代码保持原样。
        t.update_teacher=lambda:None
        if d.digest(t.model_teacher.state_dict())!=source_teacher: raise ValueError('Fixed teacher differs from source')
    proxy=GeneratorInput(g,use_raw); by_id={s.sample_id:i for i,s in enumerate(data.samples)}
    remaining=len(order)-start
    # 只保留一个原子续跑完整状态；预留写入临时副本及成熟优化器空间。
    d.require_space(folder,8*d.state_bytes(d.capture(t,g,start,order))+remaining*2*sum(p.numel()*p.element_size() for p in t.model.parameters()))
    for i in tqdm(range(start,len(order)),initial=start,total=len(order),desc=f'{group} adaptation',unit='img'):
        observer=UpdateSummary(t); t.observer=observer
        name,image=data[by_id[order[i]]]
        try: losses=t.adapt(image.unsqueeze(0),[name],proxy)
        finally: t.observer=None
        d.require_strict()
        if fixed and d.digest(t.model_teacher.state_dict())!=source_teacher: raise ValueError('Fixed teacher changed')
        path=folder/'snapshots'/f'{i+1:06d}.pth'
        save_checkpoint(path,dict(student=d.cpu_tree(t.model.state_dict()),teacher=d.cpu_tree(t.model_teacher.state_dict())))
        hashes[str(path.relative_to(folder))]=u.sha256_file(path)
        rows.append(dict(step=i+1,sample_id=name,losses=losses,**observer.values,**proxy.last))
        if i+1 in d.BOUNDARIES or i+1==len(order):
            # 快照与日志的提交点由一个原子 resume 文件决定。未提交区间允许确定性重算覆盖。
            save_checkpoint(resume,dict(state=d.capture(t,g,i+1,order),rows=rows,files=hashes,
                plan_sha256=u.sha256_file(Path(args.output_dir)/'plan.json')))
            write(folder/'progress.json',dict(committed_step=i+1,total=len(order)))
    write(folder/'trajectory.json',rows); hashes['trajectory.json']=u.sha256_file(folder/'trajectory.json')
    write(status_path,dict(status='complete',steps=len(order),files=hashes))
    del t,g,proxy; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f'{group}: adaptation COMPLETE',flush=True)


def evaluate_group(args,group):
    protocol=contract(args); folder=group_paths(args,group)
    status=read(folder/'run_status.json')
    if status['status']!='complete': raise ValueError('Group run incomplete')
    for name,sha in status['files'].items():
        if u.sha256_file(folder/name)!=sha: raise ValueError('Group artifact changed')
    output=folder/'evaluation'; output.mkdir(parents=True,exist_ok=True)
    index_path=output/'index.json'; done=read(index_path) if index_path.exists() else {}
    for name,sha in done.items():
        if u.sha256_file(output/name)!=sha: raise ValueError('Committed evaluation changed')
    if (output/'status.json').exists() and read(output/'status.json')['status']=='complete':
        print(f'{group}: evaluation already complete; SKIP',flush=True); return
    groups,_,checkpoint=d.inputs(args.experiment_dir); order=protocol['order']
    model=u.build_backbone('nu2net',checkpoint,u.resolve_device(args.device,0))
    source_state_sha=d.digest(model.state_dict())
    source_rows={r['sample_id']:r for r in read(Path(args.diagnostic_dir)/'evaluation/000000_student.json')}
    samples={s.sample_id:s for v in groups.values() for s in v}
    for step in tqdm(range(len(order)+1),desc=f'{group} offline evaluation',unit='step'):
        names=[f'{step:06d}_{role}.json' for role in ['student','teacher']]
        if all(n in done for n in names): continue
        if step:
            packet=u.load_checkpoint(folder/'snapshots'/f'{step:06d}.pth')
            if GROUPS[group][0] and d.digest(packet['teacher'])!=source_state_sha:
                raise ValueError('Cannot reuse source evaluation: fixed teacher changed')
            chosen=groups['target_eval']+groups['source_val'] if step in d.BOUNDARIES else [samples[n] for v in protocol['probes'].values() for n in v]
            if order[step-1] not in {s.sample_id for s in chosen}: chosen.append(samples[order[step-1]])
        for role,name in zip(['student','teacher'],names):
            if name in done: continue
            if step==0: result=read(Path(args.diagnostic_dir)/'evaluation'/name)
            elif role=='teacher' and GROUPS[group][0]:
                result=[source_rows[s.sample_id] for s in chosen]
            else:
                model.load_state_dict(packet[role]); result=d.score(model,chosen)
            write(output/name,result); done[name]=u.sha256_file(output/name); write(index_path,done)
    write(output/'status.json',dict(status='complete',files=len(done)))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f'{group}: evaluation COMPLETE',flush=True)


def compare(args):
    protocol=contract(args); output=Path(args.output_dir)/'comparison'; output.mkdir(parents=True,exist_ok=True)
    folders={'G0':Path(args.diagnostic_dir)/'evaluation'}
    for group in GROUPS:
        folder=group_paths(args,group)/'evaluation'
        if read(folder/'status.json')['status']!='complete': raise ValueError('All three evaluations required')
        for name,sha in read(folder/'index.json').items():
            if u.sha256_file(folder/name)!=sha: raise ValueError('Evaluation input changed')
        folders[group]=folder
    zero={r['sample_id']:r for r in read(folders['G0']/'000000_student.json')}; targets=set(protocol['order'])
    summary={}; paired={}
    for group,folder in folders.items():
        rows=read(folder/f'{len(targets):06d}_student.json'); paired[group]={r['sample_id']:r for r in rows}
        summary[group]={}
        for split in ['target','source_val']:
            subset=[r for r in rows if (r['sample_id'] in targets)==(split=='target')]
            summary[group][split]={k:sum(r[k] for r in subset)/len(subset) for k in ['psnr','ssim']}
            summary[group][split].update(count=len(subset),worse=sum(r['psnr']<zero[r['sample_id']]['psnr'] for r in subset))
    differences=[]
    for name in zero:
        scores={g:paired[g][name]['psnr'] for g in folders}
        differences.append(dict(sample_id=name,source_psnr=zero[name]['psnr'],**scores,
            G1_minus_G0=scores['G1']-scores['G0'],G2_minus_G0=scores['G2']-scores['G0'],
            interaction=scores['G3']-scores['G1']-scores['G2']+scores['G0']))
    write(output/'summary.json',summary); write(output/'paired.json',differences)
    curves=[]
    for group,folder in folders.items():
        for step in range(len(targets)+1):
            rows={r['sample_id']:r for r in read(folder/f'{step:06d}_student.json')}
            for split,ids in protocol['probes'].items():
                curves.append(dict(group=group,step=step,split=split,psnr=sum(rows[n]['psnr'] for n in ids)/len(ids)))
    write(output/'curves.json',curves)
    # G0更新范数直接读原日志，无需重放；旧日志用float32逐张量范数，
    # 新组用float64归约，数值精度略有区别，不据极小差值作因果判断。
    updates={}
    with (Path(args.diagnostic_dir)/'run/steps.jsonl').open(encoding='utf-8') as stream:
        original=[]
        for line in stream:
            row=json.loads(line); item={'step':row['step']}
            for event in row['events']:
                if event['event'] in ['student_adam','student_restored']:
                    item[event['event']+'_delta_l2']=sum(v['delta_norm']**2 for v in event['parameters'].values())**.5
            original.append(item)
    updates['G0']=original
    for group in GROUPS: updates[group]=read(group_paths(args,group)/'trajectory.json')
    write(output/'updates.json',updates)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axes,protocol['probes']):
        for group in folders:
            r=[r for r in curves if r['group']==group and r['split']==split]
            ax.plot([v['step'] for v in r],[v['psnr'] for v in r],label=group)
        ids=protocol['probes'][split]; ax.axhline(sum(zero[n]['psnr'] for n in ids)/len(ids),ls='--',color='grey',label='source')
        ax.set(title=split,xlabel='Adaptation step',ylabel='PSNR'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(output/'psnr.png',dpi=150); plt.close(fig)
    write(output/'status.json',dict(status='complete',note='Fixed probe curves; final statistics use complete sets. Reference never selects a model.'))
    print('Comparison COMPLETE; G0 reused without adaptation or inference',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['experiment_dir','diagnostic_dir','output_dir']: p.add_argument('--'+name,required=True)
    p.add_argument('--phase',choices=['preflight','run','evaluate','compare','all'],default='all')
    p.add_argument('--group',choices=list(GROUPS)); p.add_argument('--device',default='cuda:0')
    args=p.parse_args(argv)
    out=Path(args.output_dir).resolve()
    for path in [args.experiment_dir,args.diagnostic_dir]:
        original=Path(path).resolve()
        if out==original or out in original.parents or original in out.parents: p.error('Use an independent sibling output directory')
    if args.phase in ['run','evaluate'] and args.group is None: p.error('--group required')
    if os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8') not in [':4096:8',':16:8']: raise ValueError('Invalid CUBLAS workspace')
    torch.use_deterministic_algorithms(True,warn_only=False); u.seed_everything(23)
    out.mkdir(parents=True,exist_ok=True)
    try:
        contract(args)
        if args.phase=='preflight':
            d.inputs(args.experiment_dir)
            print('Preflight PASSED; G0 will be reused',flush=True)
        elif args.phase=='run': run_group(args,args.group)
        elif args.phase=='evaluate': evaluate_group(args,args.group)
        elif args.phase=='compare': compare(args)
        else:
            for group in GROUPS: run_group(args,group)
            for group in GROUPS: evaluate_group(args,group)
            compare(args)
    except Exception as exc:
        write(out/f'failure_{uuid.uuid4().hex}.json',dict(phase=args.phase,group=args.group,error=repr(exc)))
        raise


if __name__=='__main__': main()
