"""单步因果诊断：原轨迹精确重放后，离线检查 Adam / Fisher 的实际作用。

独立入口是为了保持已验收的 v2 源码哈希不变。reference 不进入重放更新。
"""
import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

import diagnose_uie as d
import uie_tta as u
from train_source_uie import save_checkpoint, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_packet(path,expected_hash):
    if u.sha256_file(path)!=expected_hash: raise ValueError('Replay packet changed')
    return u.load_checkpoint(path)


def verified_protocol(args):
    root=Path(args.diagnostic_dir)
    protocol=read(root/'protocol.json')
    if protocol['environment']!=d.environment():
        raise ValueError('Environment or original v2 code differs; retain the accepted v2 files')
    if protocol.get('diagnostic_protocol')!='fixed_fisher_strict':
        raise ValueError('A strict fixed-Fisher v2 run is required')
    if read(root/'run/status.json')['status']!='complete':
        raise ValueError('Original diagnostic run incomplete')
    d.verify_fixed_files(args.experiment_dir)
    return protocol


class UpdateObserver:
    """只复制真实更新中的值；不执行额外前向、随机采样或反向。"""
    def __init__(self):
        self.states={}; self.values={}; self.ranges=[]; self.gradients={}

    def __call__(self,event,values):
        if event=='student_before':
            self.states['before']=d.cpu_tree(values['model'].state_dict())
            self.states['teacher']=d.cpu_tree(values['teacher'].state_dict())
            self.values.update({k:d.cpu_tree(values[k]) for k in ['raw','pseudo','confidence']})
        elif event=='student_outputs':
            self.values.update({k:d.cpu_tree(values[k]) for k in ['raw_output','regenerated','regenerated_output']})
        elif event in ['student_adam','student_restored']:
            label='adam' if event=='student_adam' else 'restored'
            self.states[label]=d.cpu_tree(values['model'].state_dict())
            if label=='adam':
                self.gradients={n:d.cpu_tree(p.grad) for n,p in values['model'].named_parameters() if p.requires_grad}
        elif event=='sample_unclipped_start':
            # RDDM 的 x_start 在 [-1,1]；不能把 below_zero 当成截断率。
            a=values['value'].detach()
            self.ranges.append(dict(min=float(a.min()),max=float(a.max()),
                below_minus_one=float((a < -1).float().mean()),above_one=float((a > 1).float().mean())))


def attach(t,g,observer):
    t.observer=observer; g.trainer.observer=observer
    g.trainer.model.observer=observer; g.trainer.ema.ema_model.observer=observer


def replay(args):
    protocol=verified_protocol(args)
    root=Path(args.diagnostic_dir); out=Path(args.output_dir)/'replay'
    out.mkdir(parents=True,exist_ok=False)
    steps=sorted(set(args.steps)); order=protocol['order']
    if not steps or any(s<1 or s>len(order) for s in steps): raise ValueError('Invalid requested steps')
    trajectory=read(root/'run/trajectory.json')
    records={}
    with (root/'run/steps.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['step'] in steps: records[row['step']]=row
    print('Building original models and loading fixed Fisher...',flush=True)
    t,g,data,actual_order,_,opt=d.build(args.experiment_dir,args.device,out/'bootstrap',fixed_fisher=True)
    if actual_order!=order or vars(opt)!=protocol['config'] or data.identity()!=protocol['target']:
        raise ValueError('Original configuration/order/data changed')
    d.require_space(out,2*1024**3+len(steps)*5*sum(p.numel()*p.element_size() for p in t.model.parameters()))
    index={s.sample_id:i for i,s in enumerate(data.samples)}
    boundaries={int(p.stem):p for p in (root/'run/replay_states').glob('*.pth')}
    manifest={}; boundary_hashes={}
    for step in steps:
        eligible=[i for i in boundaries if i<step]
        if not eligible: raise FileNotFoundError('Missing original replay boundary before requested step')
        start=max(eligible); state=u.load_checkpoint(boundaries[start])
        expected=trajectory[start-1]['state_sha256'] if start else d.digest(u.load_checkpoint(root/'acceptance/initial.pth'))
        if d.digest(state)!=expected: raise ValueError(f'Original boundary state changed: {start}')
        boundary_hashes[str(start)]=u.sha256_file(boundaries[start])
        d.restore(t,g,state); del state
        if d.digest(d.capture(t,g,start,order))!=expected: raise ValueError('Boundary restoration is not exact')
        observer=None
        for cursor in tqdm(range(start,step),desc=f'Replay to step {step}',unit='img'):
            if cursor==step-1:
                observer=UpdateObserver(); attach(t,g,observer)
            name,image=data[index[order[cursor]]]
            try:
                losses=t.adapt(image.unsqueeze(0),[name],g)
            finally:
                if observer is not None: attach(t,g,None)
            d.require_strict()
            actual=d.digest(d.capture(t,g,cursor+1,order))
            if actual!=trajectory[cursor]['state_sha256'] or losses!=trajectory[cursor]['losses']:
                raise ValueError(f'Exact replay failed at step {cursor+1}; offline analysis blocked')
        # 在进行 reference 评价前，先确认输入和在线实际输出与原观测逐张量相同。
        original={}
        for event in records[step]['events']:
            if event['event'] in ['student_before','student_outputs']:
                original.update({k:v['sha256'] for k,v in event.items() if isinstance(v,dict) and 'sha256' in v})
        for key,value in observer.values.items():
            if d.digest(value)!=original[key]: raise ValueError(f'Observed tensor differs: {step}/{key}')
        path=out/f'{step:06d}.pth'
        save_checkpoint(path,dict(step=step,sample_id=order[step-1],crop=records[step]['crop'],
            states=observer.states,values=observer.values,online_gradients=observer.gradients,
            masks={n:d.cpu_tree(v[1]) for n,v in t.fishers.items()},
            sampling_ranges=observer.ranges,losses=losses))
        manifest[path.name]=u.sha256_file(path)
        print(f'Step {step}: exact replay PASSED; before/Adam/restored states saved',flush=True)
    write_json(out/'status.json',dict(status='complete',steps=steps,files=manifest,
        boundary_sha256=boundary_hashes,environment=d.environment(),
        protocol_sha256=u.sha256_file(root/'protocol.json'),entry_sha256=u.sha256_file(__file__)))
    del t,g; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def flatten(values,names):
    return torch.cat([values[n].detach().cpu().reshape(-1).double() for n in names])


def alignment(a,b):
    """b 是实际 delta 或梯度；名称注明含义。对实际 delta，dot<0 表示一阶改善。"""
    a=a.double(); b=b.double()
    dot=float(torch.dot(a,b)); denom=float(a.norm()*b.norm())
    return dict(dot=dot,cosine=None if denom==0 else max(-1.,min(1.,dot/denom)),
                first_norm=float(a.norm()),second_norm=float(b.norm()))


def gradient(loss,params):
    gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().cpu().reshape(-1).double()
                      for p,g in zip(params,gs)])


def metrics(pred,ref):
    from utils.val_utils import compute_psnr_ssim
    psnr,ssim=compute_psnr_ssim(pred,ref)
    diff=pred.detach()-ref
    return dict(psnr=float(psnr),ssim=float(ssim),l1=float(diff.abs().mean()),mse=float(diff.square().mean()),
                rgb_bias=diff.mean((0,2,3)).cpu().tolist())


def inspect_one(t,packet,sample,probes,output):
    device=t.device; states=packet['states']; names=list(t.candidate_parameters())
    params=list(t.candidate_parameters().values()); values={k:v.to(device) for k,v in packet['values'].items()}
    raw_full=u.read_rgb(sample.raw_path).unsqueeze(0).to(device)
    ref_full=u.read_rgb(sample.reference_path).unsqueeze(0).to(device)
    c=packet['crop']; y,x,s=c['top'],c['left'],c['patch_size']
    ref=ref_full[...,y:y+s,x:x+s]
    if not torch.equal(raw_full[...,y:y+s,x:x+s],values['raw']): raise ValueError('Offline raw crop differs')
    t.model.load_state_dict(states['before']); t.model.train()
    pred=t.model(values['regenerated'])
    if not torch.equal(pred.detach(),values['regenerated_output']): raise ValueError('Offline regenerated forward differs')
    low=t.wt1(pred.permute(0,2,3,1))[:,0]
    losses={}
    for branch,key,weight in [('s','raw_output',1.),('t','pseudo',t.opt.teacher_weight)]:
        target=values[key]; target_low=t.wt1(target.permute(0,2,3,1))[:,0]
        losses[branch+'_pixel']=weight*t.G_lossfn_weight*F.l1_loss(values['confidence']*pred,values['confidence']*target)
        losses[branch+'_perceptual']=weight*t.P_lossfn_weight*t.perceptual_loss(pred,target)
        losses[branch+'_low']=weight*t.RL_lossfn_weight*F.l1_loss(low,target_low)
    # 保留原实现的运算顺序：教师权重在三项求和后相乘。
    # 将权重分配到各项再求和会产生浮点差异，不能用它做精确重放验收。
    raw_low=t.wt1(values['raw_output'].permute(0,2,3,1))[:,0]
    pseudo_low=t.wt1(values['pseudo'].permute(0,2,3,1))[:,0]
    total=t.compute_loss2(pred,values['raw_output'],low,raw_low,values['confidence'])
    total=total+t.opt.teacher_weight*t.compute_loss2(pred,values['pseudo'],low,pseudo_low,values['confidence'])
    grads={k:gradient(v,params) for k,v in losses.items()}
    grads['total']=gradient(total,params)
    online=flatten(packet['online_gradients'],names)
    # 同图同权重反向必须与真实更新梯度一致，不能只看 loss 接近。
    if not torch.equal(grads['total'],online): raise ValueError('Offline adaptation gradient differs from actual Adam input')
    loss_values={k:float(v.detach()) for k,v in losses.items()}
    del losses,total,pred,low
    grads['s']=grads['s_pixel']+grads['s_perceptual']+grads['s_low']
    grads['t']=grads['t_pixel']+grads['t_perceptual']+grads['t_low']
    oracle={}
    for label,image,target in [('raw_crop',values['raw'],ref),('raw_full',raw_full,ref_full)]:
        pred=t.model(image)
        for loss_name,loss_fn in [('l1',F.l1_loss),('mse',F.mse_loss)]:
            oracle[label+'_'+loss_name]=gradient(loss_fn(pred,target),params)
        del pred
    before=flatten(states['before'],names)
    updates={k:flatten(states[k],names)-before for k in ['adam','restored']}
    updates['fisher_correction']=updates['restored']-updates['adam']
    mask=flatten(packet['masks'],names).bool(); free=~mask
    result=dict(step=packet['step'],sample_id=packet['sample_id'],losses=loss_values,
                actual_gradient_exact=True,gradient_comparisons={},state_metrics={},
                sampling_ranges=packet['sampling_ranges'],input_statistics={})
    for label,g in oracle.items():
        result['gradient_comparisons'][label]={}
        for name,v in dict(**{'gradient_'+k:v for k,v in grads.items()},**{'delta_'+k:v for k,v in updates.items()}).items():
            result['gradient_comparisons'][label][name]=dict(all=alignment(g,v),free=alignment(g[free],v[free]))
    for name in ['raw','regenerated']:
        a=values[name]
        result['input_statistics'][name]=dict(rgb_mean=a.mean((0,2,3)).cpu().tolist(),
            zero_fraction=(a==0).float().mean((0,2,3)).cpu().tolist(),
            one_fraction=(a==1).float().mean((0,2,3)).cpu().tolist())
    images={}; source=d.cpu_tree(t.model_state)
    for label,state in tqdm(list(states.items())+[('source',source)],desc=f'Evaluate step {packet["step"]}',unit='state'):
        t.model.load_state_dict(state); t.model.eval()
        with torch.no_grad():
            a=t.model(values['raw']); b=t.model(raw_full); regen=t.model(values['regenerated'])
            result['state_metrics'][label]=dict(raw_crop=metrics(a,ref),raw_full=metrics(b,ref_full),
                regenerated_crop=metrics(regen,ref),
                crop_vs_full_mae=float((a-b[...,y:y+s,x:x+s]).abs().mean()),
                probes=d.score(t.model,probes))
            images[label]=dict(raw_crop=d.cpu_tree(a),raw_full=d.cpu_tree(b),regenerated_crop=d.cpu_tree(regen))
    # 固定观察集逐图结果完整保存；不根据 reference 决定是否应用更新。
    write_json(output/f'{packet["step"]:06d}.json',result)
    save_checkpoint(output/f'{packet["step"]:06d}_outputs.pt',images)


def evaluate(args):
    protocol=verified_protocol(args); root=Path(args.output_dir)
    status=read(root/'replay/status.json')
    if (status['status']!='complete' or status['environment']!=d.environment()
        or status['entry_sha256']!=u.sha256_file(__file__)
        or status['protocol_sha256']!=u.sha256_file(Path(args.diagnostic_dir)/'protocol.json')):
        raise ValueError('Replay contract changed')
    output=root/'evaluation'; output.mkdir(parents=True,exist_ok=False)
    groups,_,checkpoint=d.inputs(args.experiment_dir)
    samples={s.sample_id:s for v in groups.values() for s in v}
    probes=[samples[n] for v in protocol['probes'].values() for n in v]
    from options import parse_args
    opt=parse_args([])
    for k,v in protocol['config'].items(): setattr(opt,k,v)
    opt.device=str(u.resolve_device(args.device,0))
    model=u.build_backbone('nu2net',checkpoint,opt.device); u.configure_model(opt,model)
    t=u.SRTTA(opt,model,torch.optim.Adam(u.collect_params(model)[0],lr=opt.lr,betas=tuple(opt.betas)))
    for step in status['steps']:
        path=root/'replay'/f'{step:06d}.pth'
        packet=load_packet(path,status['files'][path.name])
        inspect_one(t,packet,samples[packet['sample_id']],probes,output)
        print(f'Step {step}: offline diagnosis complete',flush=True)
    write_json(output/'status.json',dict(status='complete',steps=status['steps'],environment=d.environment()))
    print('Update diagnosis COMPLETE. Return evaluation/ and replay/status.json.',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment_dir',required=True)
    p.add_argument('--diagnostic_dir',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--phase',choices=['replay','evaluate','all'],default='all')
    p.add_argument('--steps',type=int,nargs='+',default=[1,20,133])
    p.add_argument('--device',default='cuda:0'); args=p.parse_args(argv)
    target=Path(args.output_dir).resolve()
    for original in [Path(args.experiment_dir).resolve(),Path(args.diagnostic_dir).resolve()]:
        if target==original or original in target.parents or target in original.parents:
            p.error('Output must be a separate sibling directory, outside original results')
    workspace=os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if workspace not in [':4096:8',':16:8']: raise ValueError('Unsupported CUBLAS_WORKSPACE_CONFIG')
    torch.use_deterministic_algorithms(True,warn_only=False); u.seed_everything(23)
    target.mkdir(parents=True,exist_ok=True)
    try:
        if args.phase in ['replay','all']: replay(args)
        if args.phase in ['evaluate','all']: evaluate(args)
    except Exception as exc:
        import time
        write_json(target/f'failure_{time.time_ns()}.json',dict(error=repr(exc),phase=args.phase,environment=d.environment()))
        raise


if __name__=='__main__': main()
