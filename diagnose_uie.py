"""UIE 诊断编排、只读观测与离线评价。不会自动修改适配超参数。"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
import subprocess
import time
import os
from contextlib import contextmanager
from pathlib import Path
from copy import deepcopy

import numpy as np
from PIL import Image
import torch

import uie_tta as u
from train_source_uie import (read_csv, write_csv, write_json, checked_manifest,
                              random_state, restore_random_state, save_checkpoint)

BOUNDARIES = (0, 1, 5, 10, 20, 40, 80, 133)

# 用户确认的第一轮产物兼容记录：只允许这一份已核查的历史Fisher。
# 旧源码身份保留；当前数据/参数身份仍必须逐项相同，不给任意旧缓存放行。
FIXED_FISHER_CONTRACT = dict(
    files={'dctta/fisher.pt':'b998a58d7a84aac7654ca881102a87a4c1b14b19ab0424e54c158617079ac1cc',
           'dctta/run.json':'690222eabeacad584ddf4d528645ad866e3e62bd7328af44028968700ad6ece0',
           'source/checkpoints/best.pth':'390600b2982289d4972ec40dcd14443832d4776c834a495ba35573da46464b05'},
    legacy_implementation={'uie_tta.py':'eb14edc921ffc1c0d51fd730b3dd038cbe6c2fc09960c424b706590546b850d1',
        'options.py':'241f90547692b245d837797a7ef46ceb53d63bc93b40d6e0243e50b41ccf8474',
        'nu2net.py':'040c0dc550c5d9899279ea0fd037372b23b2fb73ca0e09c2b991f1cb09a91b10'},
    current_implementation={'uie_tta.py':'a0b5c7cb0ebff9afaa10df1e29d9de787a19bfdc9f6e13a53afb553411f5271a',
        'uie_backbones/nu2net.py':'040c0dc550c5d9899279ea0fd037372b23b2fb73ca0e09c2b991f1cb09a91b10'})


def fixed_protocol(args):
    return getattr(args,'protocol','legacy')=='fixed_fisher_strict'


def require_strict():
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError('Strict deterministic algorithms must remain enabled')


def numerics():
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


@contextmanager
def numerical_mode(strict):
    enabled=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(strict,warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled,warn_only=warn)


def verify_fixed_files(exp):
    for name,expected in FIXED_FISHER_CONTRACT['files'].items():
        if u.sha256_file(Path(exp)/name)!=expected:
            raise ValueError(f'Approved first-round artifact changed: {name}')
    for name,expected in FIXED_FISHER_CONTRACT['current_implementation'].items():
        if u.sha256_file(Path(__file__).parent/name)!=expected:
            raise ValueError(f'Unreviewed Fisher implementation: {name}')


def load_fixed_fisher(tta, exp, folder):
    verify_fixed_files(exp)
    path=Path(exp)/'dctta/fisher.pt'; payload=u.load_checkpoint(path)
    original=payload['identity']; current=tta.fisher_identity
    if original.get('implementation')!=FIXED_FISHER_CONTRACT['legacy_implementation']:
        raise ValueError('Unreviewed historical Fisher implementation')
    if {k:v for k,v in original.items() if k!='implementation'}!={k:v for k,v in current.items() if k!='implementation'}:
        raise ValueError('Fixed Fisher source/data/crop/parameter identity mismatch')
    tta._validate_fishers(payload['fishers'])
    tta.fishers={n:[f.to(tta.device),m.to(tta.device)] for n,(f,m) in payload['fishers'].items()}
    tta.fisher_identity=deepcopy(original)
    tta.fisher_origin=dict(policy='fixed_first_round',sha256=u.sha256_file(path),
                          original_identity=original,current_identity=current)
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fisher.pt').exists(): raise FileExistsError(folder/'fisher.pt')
    shutil.copyfile(path,folder/'fisher.pt')
    write_json(folder/'fisher_compatibility.json',dict(**tta.fisher_origin,contract=FIXED_FISHER_CONTRACT))


def state_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,np.ndarray): return value.nbytes
    if isinstance(value,dict): return sum(state_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)): return sum(state_bytes(v) for v in value)
    return 0


def require_space(folder, estimated_bytes):
    required=int(estimated_bytes*1.2)
    if shutil.disk_usage(folder).free<required:
        raise OSError(f'Insufficient diagnostic disk space: need {required/1e9:.1f} GB free in {folder}')


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return deepcopy(value)


def digest(value):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            h.update(str((t.dtype, tuple(t.shape))).encode()); h.update(t.numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((v.dtype, v.shape)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(repr(k).encode()); visit(v[k])
        elif isinstance(v, (tuple, list)):
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def tensor_stats(t):
    a = t.detach().cpu().float().numpy()
    finite = np.isfinite(a)
    out = dict(shape=list(a.shape), dtype=str(t.dtype), sha256=digest(t), finite=bool(finite.all()))
    if not finite.all():
        raise FloatingPointError('Non-finite diagnostic tensor')
    out.update(min=float(a.min()), max=float(a.max()), mean=float(a.mean()), std=float(a.std()),
               below_zero=float((a < 0).mean()), above_one=float((a > 1).mean()))
    if a.ndim == 4 and a.shape[1] == 3:
        flat = a.transpose(1, 0, 2, 3).reshape(3, -1)
        out.update(rgb_mean=flat.mean(1).tolist(), rgb_std=flat.std(1).tolist(),
                   rgb_quantiles=np.quantile(flat, [.01, .5, .99], axis=1).tolist(),
                   luma_proxy=float((flat.mean(1)*[.2126,.7152,.0722]).sum()))
    return out


def model_state(model):
    return dict(weights=cpu_tree(model.state_dict()),
                modes={n:m.training for n,m in model.named_modules()},
                grads={n:cpu_tree(p.grad) for n,p in model.named_parameters()},
                buffer_devices={n:str(b.device) for n,b in model.named_buffers()})


def restore_model(model, state):
    model.load_state_dict(state['weights'], strict=True)
    modules = dict(model.named_modules())
    for name, mode in state['modes'].items():
        modules[name].training = mode
    for name, device in state['buffer_devices'].items():
        prefix, _, key = name.rpartition('.')
        modules[prefix]._buffers[key] = state['weights'][name].to(device).clone()
    for name, parameter in model.named_parameters():
        grad = state['grads'][name]
        parameter.grad = None if grad is None else grad.to(parameter.device).clone()


def capture(tta, generator, cursor, order):
    tr = generator.trainer
    return dict(format_version=2, cursor=cursor, order=order,numerics=numerics(),
                fisher_origin=deepcopy(getattr(tta,'fisher_origin',None)),
                loader_policy='materialized_original_loader_order_num_workers_0',
                frozen_loss_sha256=digest([tta.perceptual_loss.state_dict(),tta.wt1.state_dict()]),
                student=model_state(tta.model),
                teacher=model_state(tta.model_teacher), source=cpu_tree(tta.model_state),
                fisher=cpu_tree(tta.fishers), fisher_identity=tta.fisher_identity,
                student_optimizer=cpu_tree(tta.optimizer.state_dict()), steps=tta.steps,
                rddm=model_state(tr.model), rddm_ema=model_state(tr.ema),
                rddm_optimizer=cpu_tree(tr.opt0.state_dict()), rddm_local_step=tr.step,
                scaler=cpu_tree(tr.accelerator.scaler.state_dict()) if tr.accelerator.scaler else None,
                rng=cpu_tree(random_state()))


def restore(tta, generator, state):
    tr = generator.trainer
    if state.get('numerics')!=numerics(): raise ValueError('Replay numerical mode differs')
    if digest([tta.perceptual_loss.state_dict(),tta.wt1.state_dict()])!=state['frozen_loss_sha256']:
        raise ValueError('Frozen perceptual/wavelet implementation differs from replay state')
    restore_model(tta.model, state['student']); restore_model(tta.model_teacher, state['teacher'])
    tta.model_state = {n:t.to(tta.device) for n,t in state['source'].items()}
    tta.fishers = {n:[f.to(tta.device),m.to(tta.device)] for n,(f,m) in state['fisher'].items()}
    tta.fisher_identity = state['fisher_identity']
    tta.fisher_origin=deepcopy(state.get('fisher_origin'))
    tta.optimizer.load_state_dict(state['student_optimizer']); tta.steps = state['steps']
    restore_model(tr.model, state['rddm']); restore_model(tr.ema, state['rddm_ema'])
    tr.opt0.load_state_dict(state['rddm_optimizer']); tr.step = state['rddm_local_step']
    if state['scaler'] is not None:
        tr.accelerator.scaler.load_state_dict(state['scaler'])
    restore_random_state(state['rng'])


def parameter_summary(model, before=None):
    result = {}
    for name, p in model.named_parameters():
        t = p.detach().cpu()
        if not torch.isfinite(t).all():
            raise FloatingPointError(f'Non-finite parameter: {name}')
        item = dict(norm=float(t.norm()), grad_norm=None if p.grad is None else float(p.grad.detach().norm()))
        if p.grad is not None:
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                raise FloatingPointError(f'Non-finite gradient: {name}')
            item.update(grad_finite=bool(torch.isfinite(g).all()), grad_zero=float((g==0).float().mean()))
        if before is not None:
            delta = t-before[name]
            item.update(delta_norm=float(delta.norm()), delta_max=float(delta.abs().max()),
                        relative_delta=float(delta.norm()/(before[name].norm()+1e-12)))
        result[name] = item
    return result


class Recorder:
    """只读取真实执行中的张量；不增加随机前向、采样或 backward。"""
    def __init__(self, folder):
        self.folder = Path(folder); self.events = []; self.handles = []
        self.before = {}; self.step = 0; self.serial = 0; self.rddm_updates = 0
        self.adam = {}; self.ema_before = None

    def begin(self, step, sample, crop):
        self.step=step; self.serial=0; self.events=[]; self.before={}; self.adam={}
        self.header=dict(step=step, sample_id=sample, crop=crop)
        (self.folder/'tensors'/f'{step:06d}').mkdir(parents=True, exist_ok=False)

    def __call__(self, event, values):
        record=dict(event=event, step=self.step, serial=self.serial)
        tensors={}
        for key, value in values.items():
            if torch.is_tensor(value):
                record[key]=tensor_stats(value); tensors[key]=cpu_tree(value)
            elif key == 'buffers':
                record[key]={n:digest(t) for n,t in value.items()}
            elif key == 'model':
                family='student' if event.startswith('student') else 'rddm'
                record['parameters']=parameter_summary(value, self.before.get(family))
                if event.endswith('before'):
                    self.before[family]={n:cpu_tree(p) for n,p in value.named_parameters()}
                if event=='student_adam':
                    self.adam={n:cpu_tree(p) for n,p in value.named_parameters()}
                if event=='student_restored':
                    record['fisher_correction']=parameter_summary(value,self.adam)
            elif key == 'teacher':
                record['teacher_parameters']=parameter_summary(value)
            elif key == 'optimizer':
                record['optimizer']=[dict(lr=g['lr'], betas=g.get('betas')) for g in value.param_groups]
                record['optimizer_state']={str(i):{k:float(v.norm()) if torch.is_tensor(v) else v
                    for k,v in state.items()} for i,state in value.state_dict()['state'].items()}
            elif key == 'ema':
                record['ema']={k:float(getattr(value,k)) for k in ['step','beta','update_every','update_after_step','initted'] if hasattr(value,k)}
                record['ema']['effective_decay']=float(value.get_current_decay())
                record['ema']['weights_sha256']=digest(value.ema_model.state_dict())
                if event=='rddm_before':
                    self.ema_before=record['ema'].copy()
                if event=='rddm_after':
                    record['ema']['changed']=record['ema']['weights_sha256']!=self.ema_before['weights_sha256']
                    record['ema']['before']=self.ema_before
                    record['ema']['online_ema_parameter_l2']=float(torch.sqrt(sum(
                        (a.detach()-b.detach().to(a.device)).square().sum()
                        for a,b in zip(value.model.parameters(),value.ema_model.parameters()))))
            else:
                record[key]=value
        if tensors:
            path=self.folder/'tensors'/f'{self.step:06d}'/f'{self.serial:03d}_{event}.pt'
            save_checkpoint(path,tensors); record['tensor_file']=str(path.relative_to(self.folder))
        self.events.append(record); self.serial+=1
        if event=='rddm_after': self.rddm_updates+=1

    def attach(self, tta, generator):
        tta.observer=self
        generator.trainer.observer=self
        generator.trainer.model.observer=self
        generator.trainer.ema.ema_model.observer=self
        for label, adapter in [('student',tta.model),('teacher',tta.model_teacher)]:
            if hasattr(adapter.backbone, 'decoder'):
                def pre_tail(module, args, output, label=label):
                    self(label+'_tail_before',dict(value=output,
                         normalization_trigger=bool((output.max()>1) | (output.min()<0))))
                def post_tail(module, args, output, label=label):
                    self(label+'_tail_after',dict(value=output))
                self.handles.append(adapter.backbone.decoder.register_forward_hook(pre_tail))
                self.handles.append(adapter.backbone.register_forward_hook(post_tail))

    def close(self, tta, generator):
        tta.observer=None; generator.trainer.observer=None
        generator.trainer.model.observer=None; generator.trainer.ema.ema_model.observer=None
        for handle in self.handles: handle.remove()
        self.handles=[]

    def end(self, losses, tta):
        protected={}
        for name, parameter in tta.candidate_parameters().items():
            mask=tta.fishers[name][1].bool()
            protected[name]=bool(torch.equal(parameter.detach()[mask],tta.model_state[name][mask]))
        if not all(protected.values()): raise RuntimeError('Fisher restoration failed')
        components=[e for e in self.events if e['event']=='loss_components']
        for i,event in enumerate(components):
            event['branch']='student_consistency' if i==0 else 'teacher_constraint'
            event['outer_weight']=1. if i==0 else tta.opt.teacher_weight
        with (self.folder/'steps.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(**self.header,losses=losses,protected=protected,events=self.events))+'\n')
        sub=[]
        with (self.folder/'rddm_steps.jsonl').open('a',encoding='utf-8') as f:
            for event in self.events:
                if event['event']=='rddm_before': sub=[]
                if event['event'].startswith('rddm_'): sub.append(event)
                if event['event']=='rddm_after':
                    f.write(json.dumps(dict(step=self.step,sample_id=self.header['sample_id'],events=sub))+'\n')
        write_json(self.folder/'progress.json',dict(step=self.step,rddm_updates=self.rddm_updates,status='running'))
        self.events=[]; self.before={}


def environment():
    versions={}
    for name in ['torch','torchvision','numpy','Pillow','scikit-image','ema-pytorch','accelerate']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    files=[Path(__file__),Path(u.__file__),Path(__file__).with_name('options.py'),
           Path(__file__).parent/'RDDM/net.py',Path(__file__).parent/'RDDM/src/residual_denoising_diffusion_pytorch.py',
           Path(__file__).parent/'uie_backbones/nu2net.py',Path(__file__).parent/'utils/loss_utils.py',
           Path(__file__).parent/'uie_backbones/__init__.py',Path(__file__).parent/'train_source_uie.py',
           Path(__file__).parent/'utils/val_utils.py']
    try:
        driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                              capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired): driver='unavailable'
    return dict(python=sys.version,platform=platform.platform(),packages=versions,cuda=torch.version.cuda,driver=driver,
                cudnn=torch.backends.cudnn.version(),deterministic=torch.are_deterministic_algorithms_enabled(),
                numerical_settings=numerics(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                implementation={str(p.relative_to(Path(__file__).parent)):u.sha256_file(p) for p in files})


def inputs(exp, check_images=True):
    folder=Path(exp)/'manifests'; audit=json.loads((folder/'audit.json').read_text())
    result={}
    for name in ['source_train','source_val','target_eval']:
        path=folder/(name+'.csv')
        if u.sha256_file(path)!=audit['files'][path.name]: raise ValueError('Manifest mismatch')
        result[name]=checked_manifest(folder,path.name,audit) if check_images else u.select_samples(manifest=path)
    checkpoint=Path(exp)/'source/checkpoints/best.pth'
    payload=u.load_checkpoint(checkpoint)
    if payload['training_contract']['data']!={k:audit['files'][k] for k in ['source_train.csv','source_val.csv']}:
        raise ValueError('Source checkpoint data mismatch')
    return result,audit,checkpoint


def compare_fishers(old, new):
    """区分数值微扰与保护集合变化；这里不设置放行容差。"""
    if set(old)!=set(new): raise ValueError('Fisher parameter names differ')
    rows=[]
    for name,(f,m) in old.items():
        g,n=new[name]
        f,m,g,n=[v.detach().cpu() for v in (f,m,g,n)]
        if f.shape!=g.shape or m.shape!=n.shape: raise ValueError(f'Fisher shape mismatch: {name}')
        if not all(torch.isfinite(v).all() for v in (f,m,g,n)):
            raise FloatingPointError(f'Non-finite Fisher: {name}')
        delta=f.double()-g.double(); changed=(m!=n).flatten()
        rows.append(dict(name=name,count=f.numel(),changed_values=int((f!=g).sum()),
            max_absolute_error=float(delta.abs().max()),
            relative_l2=float(delta.norm()/f.double().norm()) if f.double().norm()>0 else None,
            mask_changed=int(changed.sum()),protected_old=int(m.sum()),protected_new=int(n.sum()),
            changed_mask_old_max=float(f[changed].max()) if changed.any() else None,
            changed_mask_new_max=float(g[changed].max()) if changed.any() else None))
    return dict(exact_match=digest(old)==digest(new),
                total=sum(r['count'] for r in rows),mask_changed=sum(r['mask_changed'] for r in rows),
                changed_values=sum(r['changed_values'] for r in rows),
                max_absolute_error=max(r['max_absolute_error'] for r in rows),layers=rows)


def build(exp, device, folder, perceptual_loss=None, wavelet=None, fisher_only=False, fixed_fisher=False):
    from options import parse_args
    from RDDM.net import ResidualDiffusionModel
    if fixed_fisher:
        require_strict()
        verify_fixed_files(exp)
    groups,audit,checkpoint=inputs(exp)
    prior=json.loads((Path(exp)/'dctta/run.json').read_text())
    if prior['source_checkpoint_sha256']!=u.sha256_file(checkpoint):
        raise ValueError('Original source checkpoint changed')
    opt=parse_args([])
    for key in ['patch_size','seed','shuffle','iterations','lr','betas','teacher_weight','ema_decay','fisher_ratio']:
        setattr(opt,key,prior['configuration'][key])
    if opt.iterations!=1: raise ValueError('Diagnostic protocol requires one update per image')
    opt.device=str(u.resolve_device(device,0)); opt.compute_fisher=1
    u.seed_everything(opt.seed)
    with u.isolated_rng(opt.seed): model=u.build_backbone('nu2net',checkpoint,opt.device)
    u.configure_model(opt,model); parameters,_=u.collect_params(model)
    tta=u.SRTTA(opt,model,torch.optim.Adam(parameters,lr=opt.lr,betas=tuple(opt.betas)),
                perceptual_loss=perceptual_loss,wavelet=wavelet)
    data=u.TargetRawDataset(groups['target_eval'],opt.patch_size,opt.seed)
    if data.identity()!=prior['target_identity']: raise ValueError('Crops differ from original run')
    tta.fisher_identity=tta.cache_identity(data)
    if fixed_fisher:
        if fisher_only: raise ValueError('Fisher probe must recompute')
        load_fixed_fisher(tta,exp,folder)
    else:
        # legacy/probe保留原重算行为；固定模式另有显式产物兼容检查。
        tta.compute_fisher(u.make_raw_loader(data,0,False,opt.seed))
    old=u.load_checkpoint(Path(exp)/'dctta/fisher.pt')['fishers']
    same=digest(cpu_tree(tta.fishers))==digest(old)
    if not fixed_fisher: tta.save_fisher(Path(folder)/'fisher.pt')
    fisher_stats={}
    for name,(f,mask) in tta.fishers.items():
        a=f.detach().cpu().numpy(); selected=a[mask.detach().cpu().numpy().reshape(-1).astype(bool)]
        cutoff=float(selected.min()) if len(selected) else None
        fisher_stats[name]=dict(count=a.size,nonzero=int(np.count_nonzero(a)),protected=int(mask.sum()),
                     quantiles=np.quantile(a,[0,.5,.9,.99,1]).tolist(),cutoff=cutoff,
                     tied_at_cutoff=0 if cutoff is None else int(np.count_nonzero(a==cutoff)))
    write_json(Path(folder)/'fisher_statistics.json',fisher_stats)
    write_json(Path(folder)/'fisher_comparison.json',compare_fishers(old,tta.fishers))
    # 独立探针只保存证据，不创建RDDM、不更新学生，也不生成验收通过标记。
    if fisher_only: return tta,None,data,None,groups,opt
    if not same: raise ValueError('Recomputed Fisher differs; diagnose environment before replay')
    with u.isolated_rng(opt.seed+1):
        generator=ResidualDiffusionModel(opt,image_size=opt.patch_size,device=opt.device,results_folder=Path(folder)/'rddm')
    # 使用真实 DataLoader 产生原顺序并消费其独立 RNG；取完后主循环只按记录索引读取。
    order=[names[0] for names,_ in u.make_raw_loader(data,0,bool(opt.shuffle),opt.seed)]
    if order!=prior['adaptation_order']: raise ValueError('Adaptation order mismatch')
    if fixed_fisher: require_strict()
    return tta,generator,data,order,groups,opt


def fisher_probe(args):
    """相同配置独立重建两次，再用严格确定性模式检查单图反向的算子支持。"""
    root=Path(args.output_dir); folder=root/'fisher_probe'
    folder.mkdir(parents=True,exist_ok=False)
    require_space(folder,512*1024**2)
    write_json(folder/'environment.json',environment())
    exp=Path(args.experiment_dir)
    previous=u.load_checkpoint(exp/'dctta/fisher.pt')
    returned_path=root/'acceptance/fisher.pt'
    returned=u.load_checkpoint(returned_path) if returned_path.exists() else None
    payloads=[]
    try:
        for run in (1,2):
            step_folder=folder/f'repeat_{run}'; step_folder.mkdir()
            print(f'Fisher probe {run}/2: recomputing full target Fisher',flush=True)
            t,_,data,_,groups,opt=build(exp,args.device,step_folder,fisher_only=True)
            payloads.append(u.load_checkpoint(step_folder/'fisher.pt'))
            # 不保留两份NU2Net/VGG及其GPU内存。
            if run==1: del t,data,groups,opt
        report=dict(status='measured_not_accepted',environment=environment(),
            original_sha256=u.sha256_file(exp/'dctta/fisher.pt'),
            repeats=compare_fishers(payloads[0]['fishers'],payloads[1]['fishers']),
            original_to_repeat=[compare_fishers(previous['fishers'],p['fishers']) for p in payloads],
            identities_equal_between_repeats=payloads[0]['identity']==payloads[1]['identity'])
        if returned is not None:
            report['returned_sha256']=u.sha256_file(returned_path)
            report['returned_to_repeat']=[compare_fishers(returned['fishers'],p['fishers']) for p in payloads]
        # 仅在独立探针末尾开启严格模式；不修改原算法配置，不用于计算上面的两次Fisher。
        enabled=torch.are_deterministic_algorithms_enabled()
        warn=torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            one=torch.utils.data.Subset(data,[0])
            t.compute_fisher(u.make_raw_loader(one,0,False,opt.seed))
            report['strict_single_image']=dict(status='completed',
                note='One image only; not certification of full adaptation or RDDM determinism')
        except RuntimeError as exc:
            report['strict_single_image']=dict(status='runtime_error',error=str(exc))
        finally:
            torch.use_deterministic_algorithms(enabled,warn_only=warn)
        write_json(folder/'report.json',report)
        print(f'Fisher repeat mask changes: {report["repeats"]["mask_changed"]}; '
              f'strict single-image check: {report["strict_single_image"]["status"]}',flush=True)
    except Exception as exc:
        write_json(folder/'status.json',dict(status='incomplete',error=repr(exc))); raise
    write_json(folder/'status.json',dict(status='measured_not_accepted'))


def run_segment(tta, generator, data, order, folder, start, end, record=False, full=False, save_models=False):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    by_id={s.sample_id:i for i,s in enumerate(data.samples)}
    rec=Recorder(folder) if record else None
    if rec: rec.attach(tta,generator)
    states=[]; expected_numerics=numerics()
    try:
        for cursor in range(start,end):
            started=time.perf_counter()
            device=next(tta.model.parameters()).device
            if record and device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            index=by_id[order[cursor]]; name,image=data[index]
            if rec: rec.begin(cursor+1,name,data.crops[index])
            losses=tta.adapt(image.unsqueeze(0),[name],generator)
            if numerics()!=expected_numerics: raise RuntimeError('Numerical mode changed during adaptation')
            if rec: rec.end(losses,tta)
            snapshot=capture(tta,generator,cursor+1,order)
            states.append(dict(step=cursor+1,losses=losses,state_sha256=digest(snapshot)))
            if full or save_models:
                save_checkpoint(folder/'snapshots'/f'{cursor+1:06d}.pth',dict(student=snapshot['student']['weights'],teacher=snapshot['teacher']['weights']))
                if full and (cursor+1 in BOUNDARIES or cursor+1==end):
                    save_checkpoint(folder/'replay_states'/f'{cursor+1:06d}.pth',snapshot)
            if record:
                resources=dict(step=cursor+1,seconds=time.perf_counter()-started,
                    cuda_peak_allocated=None if device.type!='cuda' else torch.cuda.max_memory_allocated(device),
                    cuda_peak_reserved=None if device.type!='cuda' else torch.cuda.max_memory_reserved(device))
                with (folder/'resources.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(resources)+'\n')
            print(f'Diagnostic step {cursor+1}/{end}: loss={losses["loss"]:.6f}',flush=True)
            del snapshot
    finally:
        if rec: rec.close(tta,generator)
    write_json(folder/'trajectory.json',states)
    return states


@torch.no_grad()
def score(model,samples,ensemble=False,strict=None):
    if strict is not None:
        with numerical_mode(strict): return score(model,samples,ensemble)
    from utils.val_utils import compute_psnr_ssim
    model.eval(); device=next(model.parameters()).device
    rows=[]
    for sample in samples:
        raw=u.read_rgb(sample.raw_path).unsqueeze(0).to(device)
        ref=u.read_rgb(sample.reference_path).unsqueeze(0).to(device)
        pred=u.eight_view_mean(model,raw) if ensemble else model(raw)
        psnr,ssim=compute_psnr_ssim(pred,ref); raw_psnr,raw_ssim=compute_psnr_ssim(raw,ref)
        clipped=pred.clamp(0,1); diff=clipped-ref
        rows.append(dict(sample_id=sample.sample_id,psnr=float(psnr),ssim=float(ssim),raw_psnr=float(raw_psnr),
                         raw_ssim=float(raw_ssim),gain_psnr=float(psnr-raw_psnr),
                         rgb_mae=diff.abs().mean((0,2,3)).cpu().tolist(),
                         luma_bias=float((diff.mean((0,2,3))*diff.new_tensor([.2126,.7152,.0722])).sum())))
    return rows


def static_audit(args):
    if fixed_protocol(args): verify_fixed_files(args.experiment_dir)
    groups,audit,checkpoint=inputs(args.experiment_dir)
    output=Path(args.output_dir)/'static_audit'; output.mkdir(parents=True,exist_ok=False)
    write_json(output/'environment.json',environment())
    summaries={}; comparisons={}
    for label,path in [('source_only',checkpoint),('source_only_x8',checkpoint),('dctta',Path(args.experiment_dir)/'dctta/adapted_model.pth')]:
        model=u.build_backbone('nu2net',path,u.resolve_device(args.device,0))
        result=score(model,groups['target_eval'],label=='source_only_x8',strict=False if fixed_protocol(args) else None)
        write_json(output/(label+'.json'),result)
        if fixed_protocol(args):
            strict_result=score(model,groups['target_eval'],label=='source_only_x8',strict=True)
            write_json(output/(label+'_strict.json'),strict_result)
            write_json(output/(label+'_numeric_difference.json'),
                {k:max(abs(a[k]-b[k]) for a,b in zip(result,strict_result)) for k in ['psnr','ssim']})
        prior=json.loads((Path(args.experiment_dir)/label/'run.json').read_text())
        old={r['image']:r for r in prior['evaluation_records']}
        comparisons[label]={k:max(abs(r[k]-old[r['sample_id']][k]) for r in result) for k in ['psnr','ssim']}
        summaries[label]={k:float(np.mean([r[k] for r in result])) for k in ['psnr','ssim','raw_psnr','gain_psnr']}
        if label=='source_only':
            for split in ['source_train','source_val']:
                write_json(output/(split+'.json'),score(model,groups[split],strict=False if fixed_protocol(args) else None))
            spatial=[]
            for split,samples in groups.items():
                data=u.TargetRawDataset(samples,320,23)
                for i,sample in enumerate(samples):
                    crop=data.crops[i]; raw=u.read_rgb(sample.raw_path).unsqueeze(0).to(next(model.parameters()).device)
                    top,left,size=crop['top'],crop['left'],crop['patch_size']
                    with torch.no_grad():
                        full=model(raw)[...,top:top+size,left:left+size]
                        patch=model(raw[...,top:top+size,left:left+size])
                    spatial.append(dict(split=split,sample_id=sample.sample_id,crop=crop,
                           mean_absolute_difference=float((full-patch).abs().mean()),
                           max_absolute_difference=float((full-patch).abs().max())))
            write_json(output/'source_crop_vs_full.json',spatial)
        del model
    write_json(output/'summary.json',dict(means=summaries,reload_max_error=comparisons,
               note='Reload differences must be reviewed before accepting first-round reproduction.'))
    if args.severity_scores:
        records=read_csv(args.severity_scores)
        write_json(output/'severity_input.json',dict(source=str(Path(args.severity_scores).resolve()),
                   sha256=u.sha256_file(args.severity_scores),records=records,
                   review='Scores retained for independent review; not used to change manifests or adaptation.'))
    passed=all(v['psnr']<=1e-5 and v['ssim']<=1e-6 for v in comparisons.values())
    write_json(output/'status.json',dict(status='complete',reload_passed=passed,
               tolerance=dict(psnr=1e-5,ssim=1e-6),severity_review='pending',reference_visual_review='pending',
               environment=environment(),protocol=getattr(args,'protocol','legacy'),
               source_sha256=u.sha256_file(checkpoint),manifest_files=audit['files']))
    if not passed: raise ValueError('Independent metric reload differs beyond predefined tolerance')


def acceptance_length(requested,data,order,opt,ema,train_num_steps):
    n=min(requested,len(order))
    if n<22: raise ValueError('Formal acceptance needs at least 22 images')
    sizes={s.sample_id:c['patch_size'] for s,c in zip(data.samples,data.crops)}
    if not any(sizes[name]<opt.patch_size for name in order[:n]) or not any(sizes[name]==opt.patch_size for name in order[:n]):
        raise ValueError('Acceptance prefix must include small and full-sized crops; increase --acceptance_steps')
    ema_boundary=int(ema.update_after_step)//train_num_steps+2
    if n<ema_boundary: raise ValueError(f'Acceptance must cross actual EMA warmup: at least {ema_boundary} images')
    return n


def acceptance(args):
    audit_status=json.loads((Path(args.output_dir)/'static_audit/status.json').read_text())
    if (not audit_status.get('reload_passed') or audit_status['environment']!=environment()
        or audit_status.get('protocol','legacy')!=getattr(args,'protocol','legacy')):
        raise ValueError('Complete static audit in this environment before acceptance')
    folder=Path(args.output_dir)/'acceptance'; folder.mkdir(parents=True,exist_ok=False)
    t,g,data,order,groups,opt=build(args.experiment_dir,args.device,folder,fixed_fisher=fixed_protocol(args))
    n=acceptance_length(args.acceptance_steps,data,order,opt,g.trainer.ema,g.trainer.train_num_steps)
    initial=capture(t,g,0,order)
    # 短程验收本身也保存大型RDDM状态，不能等完整重跑才检查磁盘。
    boundary_count=len({s for s in BOUNDARIES if 0<s<=n}|{n})
    # Adam/RAdam在首次更新后才创建动量；初态大小不能当作成熟状态大小。
    require_space(folder,state_bytes(initial)*4*(boundary_count+1)+n*256*1024**2)
    save_checkpoint(folder/'initial.pth',initial)
    write_json(folder/'environment.json',environment())
    write_json(folder/'probe_plan.json',dict(probes=select_probes(groups),order=order[:n]))
    control=run_segment(t,g,data,order,folder/'control',0,n)
    original=[json.loads(line) for line in (Path(args.experiment_dir)/'dctta/adaptation.jsonl').read_text().splitlines()]
    prior_loss_error={k:max(abs(control[i]['losses'][k]-original[i][k]) for i in range(n)) for k in ['loss_s','loss_t','loss']}
    restore(t,g,initial)
    repeat=run_segment(t,g,data,order,folder/'repeat',0,n)
    restore(t,g,initial)
    observed=run_segment(t,g,data,order,folder/'observed',0,n,True,True)
    # 从第1步边界开始恢复，覆盖前缀中的尺寸切换和EMA预热边界，不能只检查最后一步。
    state=u.load_checkpoint(folder/'observed/replay_states/000001.pth')
    restore(t,g,state)
    resumed=run_segment(t,g,data,order,folder/'resumed',1,n)
    passed=control==repeat==observed and resumed==control[1:]
    write_json(folder/'acceptance.json',dict(passed=passed,steps=n,control_repeat=control==repeat,
         observation_equal=control==observed,replay_equal=resumed==control[1:],
         original_prefix_loss_max_error=prior_loss_error,
         environment=environment(),protocol=getattr(args,'protocol','legacy'),
         fisher_origin=getattr(t,'fisher_origin',None),
         source=u.sha256_file(Path(args.experiment_dir)/'source/checkpoints/best.pth'),
         target=data.identity(),config=vars(opt)))
    if not passed: raise RuntimeError('Exact trajectory acceptance failed; full run is blocked')
    print('Acceptance PASSED',flush=True)


def select_probes(groups):
    probes={}
    for split,count in [('source_val',4),('target_eval',12)]:
        ranked=sorted(groups[split],key=lambda s:hashlib.sha256(s.sample_id.encode()).hexdigest())
        strata=[[],[]]
        for sample in ranked:
            with Image.open(sample.raw_path) as im: strata[int(min(im.size)>=320)].append(sample.sample_id)
        selected=strata[0][:count//2]+strata[1][:count//2]
        selected += [s.sample_id for s in ranked if s.sample_id not in selected][:count-len(selected)]
        probes[split]=selected
    return probes


def accepted_gate(args):
    require_strict()
    gate=json.loads((Path(args.output_dir)/'acceptance/acceptance.json').read_text())
    if not gate['passed'] or gate['environment']!=environment() or gate.get('protocol')!='fixed_fisher_strict':
        raise ValueError('A matching strict fixed-Fisher acceptance is required')
    verify_fixed_files(args.experiment_dir)
    return gate


def numeric_control(args):
    """从验收初态出发，仅关闭全局严格模式；保留完整原更新过程。"""
    gate=accepted_gate(args); root=Path(args.output_dir)
    folder=root/'numeric_control'; folder.mkdir(parents=True,exist_ok=False)
    t,g,data,order,groups,opt=build(args.experiment_dir,args.device,folder,fixed_fisher=True)
    initial=u.load_checkpoint(root/'acceptance/initial.pth')
    if data.identity()!=gate['target'] or vars(opt)!=gate['config']:
        raise ValueError('Numerical-control protocol changed')
    if digest(capture(t,g,0,order))!=digest(initial): raise ValueError('Control initial state changed')
    restore(t,g,initial)
    require_space(folder,gate['steps']*2*sum(p.numel()*p.element_size() for p in t.model.parameters())+128*1024**2)
    save_checkpoint(folder/'snapshots/000000.pth',dict(student=initial['student']['weights'],teacher=initial['teacher']['weights']))
    initial_sha=digest(initial); del initial
    strict_env=environment()
    with numerical_mode(False):
        write_json(folder/'environment.json',environment())
        rows=run_segment(t,g,data,order,folder,0,gate['steps'],save_models=True)
    reference=json.loads((root/'acceptance/control/trajectory.json').read_text())
    differences=[dict(step=a['step'],**{k:b['losses'][k]-a['losses'][k] for k in ['loss_s','loss_t','loss']})
                 for a,b in zip(reference,rows)]
    write_json(folder/'loss_difference.json',differences)
    write_json(folder/'status.json',dict(status='complete',steps=gate['steps'],environment=strict_env,
        initial_state_sha256=initial_sha,note='Default global deterministic flag; shared CUBLAS workspace. Not a bitwise replay of round one.'))


@torch.no_grad()
def compare_numeric(args):
    """两种更新数值模式的快照统一在严格模式下离线评价，不用reference选择模式。"""
    gate=accepted_gate(args); root=Path(args.output_dir)
    status=json.loads((root/'numeric_control/status.json').read_text())
    if status['status']!='complete' or status['environment']!=environment() or status['steps']!=gate['steps']:
        raise ValueError('Numerical control incomplete or changed')
    if status['initial_state_sha256']!=digest(u.load_checkpoint(root/'acceptance/initial.pth')):
        raise ValueError('Numerical-control initial state changed')
    groups,_,checkpoint=inputs(args.experiment_dir)
    plan=json.loads((root/'acceptance/probe_plan.json').read_text())
    output=root/'numeric_comparison'; output.mkdir(parents=True,exist_ok=False)
    by_id={s.sample_id:s for samples in groups.values() for s in samples}
    ids=[n for names in plan['probes'].values() for n in names]
    models=[u.build_backbone('nu2net',checkpoint,u.resolve_device(args.device,0)) for _ in range(2)]
    curves=[]; inputs_sha={name:u.sha256_file(root/name) for name in
        ['acceptance/acceptance.json','acceptance/probe_plan.json','numeric_control/status.json']}
    for step in range(gate['steps']+1):
        strict_path=root/'acceptance/observed/snapshots'/f'{step:06d}.pth'
        native_path=root/'numeric_control/snapshots'/f'{step:06d}.pth'
        if step==0:
            initial=u.load_checkpoint(root/'acceptance/initial.pth')
            strict_state={k:initial[k]['weights'] for k in ['student','teacher']}; del initial
            inputs_sha['acceptance/initial.pth']=u.sha256_file(root/'acceptance/initial.pth')
        else:
            strict_state=u.load_checkpoint(strict_path)
            inputs_sha[str(strict_path.relative_to(root))]=u.sha256_file(strict_path)
        native_state=u.load_checkpoint(native_path)
        inputs_sha[str(native_path.relative_to(root))]=u.sha256_file(native_path)
        selected=list(ids)
        if step and plan['order'][step-1] not in selected: selected.append(plan['order'][step-1])
        samples=[by_id[n] for n in selected]
        for role in ['student','teacher']:
            for model,state in zip(models,[strict_state,native_state]):
                model.load_state_dict(state[role]); model.eval()
            scored=[score(model,samples) for model in models]
            differences=[]
            for sample in samples:
                raw=u.read_rgb(sample.raw_path).unsqueeze(0).to(next(models[0].parameters()).device)
                a,b=[model(raw) for model in models]
                differences.append(dict(sample_id=sample.sample_id,mae=float((a-b).abs().mean()),
                                        max_absolute=float((a-b).abs().max())))
            write_json(output/f'{step:06d}_{role}.json',dict(strict=scored[0],default=scored[1],outputs=differences))
            for split,names in plan['probes'].items():
                for label,rows in zip(['strict','default'],scored):
                    indexed={r['sample_id']:r for r in rows}
                    curves.append(dict(step=step,role=role,split=split,mode=label,
                        **{k:float(np.mean([indexed[n][k] for n in names])) for k in ['psnr','ssim']}))
        print(f'Numerical-mode comparison {step}/{gate["steps"]}',flush=True)
    write_csv(output/'curves.csv',curves,list(curves[0]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axes,plan['probes']):
        for role in ['student','teacher']:
            for mode in ['strict','default']:
                rows=[r for r in curves if r['split']==split and r['role']==role and r['mode']==mode]
                ax.plot([r['step'] for r in rows],[r['psnr'] for r in rows],label=f'{role}/{mode}')
        ax.set(title=split,xlabel='Adaptation step',ylabel='PSNR (dB)'); ax.legend(); ax.grid(alpha=.3)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.tight_layout(); fig.savefig(output/'psnr.png',dpi=150); plt.close(fig)
    write_json(output/'status.json',dict(status='complete',environment=environment(),input_sha256=inputs_sha,
        note='Descriptive comparison only; no reference-based threshold or mode selection.'))


def full_run(args):
    root=Path(args.output_dir)
    gate=json.loads((root/'acceptance/acceptance.json').read_text())
    if (not gate['passed'] or gate['environment']!=environment()
        or gate.get('protocol','legacy')!=getattr(args,'protocol','legacy')):
        raise ValueError('Missing acceptance or environment/code changed')
    if fixed_protocol(args):
        check=json.loads((root/'numeric_comparison/status.json').read_text())
        if check['status']!='complete' or check['environment']!=environment():
            raise ValueError('Complete the numerical-mode control and comparison before full run')
        for name,expected in check['input_sha256'].items():
            if u.sha256_file(root/name)!=expected:
                raise ValueError(f'Numerical-comparison input changed: {name}')
    folder=root/'run'; folder.mkdir(parents=True,exist_ok=False)
    t,g,data,order,groups,opt=build(args.experiment_dir,args.device,folder,fixed_fisher=fixed_protocol(args))
    if data.identity()!=gate['target'] or vars(opt)!=gate['config']:
        raise ValueError('Accepted protocol changed')
    initial=capture(t,g,0,order)
    if digest(initial)!=digest(u.load_checkpoint(root/'acceptance/initial.pth')):
        raise ValueError('Initial state differs from accepted control')
    # 基于实测状态和已采集前缀计算容量，不能只按学生模型估算。
    measured=max(sum(p.stat().st_size for p in step.glob('*.pt'))
                 for step in (root/'acceptance/observed/tensors').iterdir() if step.is_dir())
    measured_state_bytes=max(p.stat().st_size for p in (root/'acceptance/observed/replay_states').glob('*.pth'))
    estimate=measured*len(order)+measured_state_bytes*len(BOUNDARIES)+len(order)*2*sum(p.numel()*p.element_size() for p in t.model.parameters())
    require_space(root,estimate)
    save_checkpoint(folder/'replay_states/000000.pth',initial)
    save_checkpoint(folder/'snapshots/000000.pth',dict(student=initial['student']['weights'],teacher=initial['teacher']['weights']))
    probes=select_probes(groups)
    write_json(root/'protocol.json',dict(config=vars(opt),environment=environment(),order=order,
        target=data.identity(),probes=probes,source_sha256=u.sha256_file(Path(args.experiment_dir)/'source/checkpoints/best.pth'),
        estimated_bytes=estimate,status='running',diagnostic_protocol=getattr(args,'protocol','legacy')))
    try:
        run_segment(t,g,data,order,folder,0,len(order),True,True)
        logged=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        rddm_lines=(folder/'rddm_steps.jsonl').read_text().splitlines()
        if len(logged)!=len(order) or len(rddm_lines)!=5*len(order):
            raise ValueError('Missing student/RDDM trace records')
        for record in logged:
            if sum(e['event']=='sample_prediction' for e in record['events'])!=5:
                raise ValueError('Missing sampling trajectory')
    except Exception as exc:
        write_json(folder/'status.json',dict(status='incomplete',error=repr(exc))); raise
    write_json(folder/'status.json',dict(status='complete',steps=len(order)))
    original=u.load_checkpoint(Path(args.experiment_dir)/'dctta/adapted_model.pth')['net']
    new=t.model.backbone.state_dict()
    write_json(folder/'original_comparison.json',dict(weights_exact=digest(new)==digest(original),
               max_absolute_weight_difference=max(float((new[k].detach().cpu()-original[k]).abs().max()) for k in original),
               note='If different, diagnosis explains the newly accepted control run; do not claim exact first-round reproduction.'))


def offline_evaluate(args):
    root=Path(args.output_dir); groups,_,checkpoint=inputs(args.experiment_dir)
    protocol=json.loads((root/'protocol.json').read_text())
    if protocol['environment']!=environment(): raise ValueError('Evaluation environment differs from diagnostic run')
    if json.loads((root/'run/status.json').read_text())['status']!='complete': raise ValueError('Run incomplete')
    output=root/'evaluation'; output.mkdir(parents=True,exist_ok=False)
    model=u.build_backbone('nu2net',checkpoint,u.resolve_device(args.device,0))
    by_id={s.sample_id:s for samples in groups.values() for s in samples}
    for path in sorted((root/'run/snapshots').glob('*.pth')):
        step=int(path.stem); payload=u.load_checkpoint(path)
        samples=groups['target_eval']+groups['source_val'] if step in BOUNDARIES else [by_id[n] for names in protocol['probes'].values() for n in names]
        if step and protocol['order'][step-1] not in {s.sample_id for s in samples}:
            samples=samples+[by_id[protocol['order'][step-1]]]
        for role in ['student','teacher']:
            model.load_state_dict(payload[role],strict=True)
            write_json(output/f'{step:06d}_{role}.json',score(model,samples))
        print(f'Offline evaluation step {step}',flush=True)
    build_report(root)
    write_json(output/'status.json',dict(status='complete',gradient_analysis='Run --phase inspect for selected-step gradients and previews'))


def build_report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol=json.loads((root/'protocol.json').read_text())
    output=root/'evaluation'; rows=[]
    for path in sorted(output.glob('*_student.json')):
        step=int(path.name.split('_')[0])
        for role in ['student','teacher']:
            data={r['sample_id']:r for r in json.loads((output/f'{step:06d}_{role}.json').read_text())}
            for split,ids in protocol['probes'].items():
                rows.append(dict(step=step,role=role,split=split,
                                 psnr=float(np.mean([data[n]['psnr'] for n in ids])),
                                 ssim=float(np.mean([data[n]['ssim'] for n in ids]))))
    write_csv(output/'probe_curves.csv',rows,list(rows[0]))
    for metric in ['psnr','ssim']:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,split in zip(axes,protocol['probes']):
            for role in ['student','teacher']:
                chosen=[r for r in rows if r['split']==split and r['role']==role]
                ax.plot([r['step'] for r in chosen],[r[metric] for r in chosen],label=role)
            base=next(r[metric] for r in rows if r['split']==split and r['step']==0)
            ax.axhline(base,linestyle='--',color='gray',label='source')
            ax.set(title=split,xlabel='Adaptation step',ylabel=metric.upper()); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(output/f'{metric}.png',dpi=150); plt.close(fig)
    lines=['# 适配诊断初步报告','', '以下均为固定观察集离线评价，不用于选择适配权重。', '',
           '| 集合 | 模型 | 第0步 PSNR | 最后一步 PSNR | 差值 |','|---|---|---:|---:|---:|']
    for split in protocol['probes']:
        for role in ['student','teacher']:
            seq=[r for r in rows if r['split']==split and r['role']==role]
            lines.append(f"| {split} | {role} | {seq[0]['psnr']:.6f} | {seq[-1]['psnr']:.6f} | {seq[-1]['psnr']-seq[0]['psnr']:.6f} |")
    lines+=['','尚需结合浮点中间图、参数更新和受控对照判断根因；曲线关联不等于因果。',
            '分项梯度与预览由 --phase inspect 生成；程序不会自动更改损失、筛选图片或重训。']
    (root/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def inspect_steps(args):
    """离线副本求分项梯度和同图更新前后差，不触碰主运行状态或参考图。"""
    import torch.nn.functional as F
    from options import parse_args
    root=Path(args.output_dir); protocol=json.loads((root/'protocol.json').read_text())
    if protocol['environment']!=environment(): raise ValueError('Inspection environment differs from diagnostic run')
    folder=root/'inspection'; folder.mkdir(parents=True,exist_ok=False)
    events=[json.loads(line) for line in (root/'run/steps.jsonl').read_text().splitlines()]
    device=u.resolve_device(args.device,0)
    checkpoint=Path(args.experiment_dir)/'source/checkpoints/best.pth'
    opt=parse_args([])
    for k,v in protocol['config'].items(): setattr(opt,k,v)
    opt.device=str(device)
    for step in args.inspect_steps:
        if not 1<=step<=len(events): raise ValueError('Inspection step outside run')
        row=events[step-1]; values={}
        for event in row['events']:
            if event['event'] in ['student_before','student_outputs']:
                values.update(u.load_checkpoint(root/'run'/event['tensor_file']))
        model=u.build_backbone('nu2net',checkpoint,device); u.configure_model(opt,model)
        optimizer=torch.optim.Adam(u.collect_params(model)[0],lr=opt.lr,betas=tuple(opt.betas))
        tta=u.SRTTA(opt,model,optimizer)
        state=u.load_checkpoint(root/'run/snapshots'/f'{step-1:06d}.pth')
        tta.model.load_state_dict(state['student']); tta.model.train()
        image=values['regenerated'].to(device)
        pred=tta.model(image)
        pred_low=tta.wt1(pred.permute(0,2,3,1))[:,0]
        terms={}
        for branch,key,weight in [('s','raw_output',1.),('t','pseudo',opt.teacher_weight)]:
            target=values[key].to(device)
            low=tta.wt1(target.permute(0,2,3,1))[:,0]
            terms[branch+'_pixel']=weight*F.l1_loss(pred,target)
            terms[branch+'_perceptual']=weight*.01*tta.perceptual_loss(pred,target)
            terms[branch+'_low']=weight*.1*F.l1_loss(pred_low,low)
        params=list(tta.candidate_parameters().values()); grads={}
        for name,loss in terms.items():
            tensors=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            grads[name]=torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().cpu() for p,g in zip(params,tensors)])
        result=dict(step=step,sample_id=row['sample_id'],losses={k:float(v.detach()) for k,v in terms.items()},
                    gradient_norms={k:float(g.norm()) for k,g in grads.items()},cosines={})
        for k,a in grads.items():
            result['cosines'][k]={j:(float(torch.dot(a,b)/(a.norm()*b.norm())) if a.norm()>0 and b.norm()>0 else None) for j,b in grads.items()}
        raw=values['raw'].to(device)
        records=read_csv(Path(args.experiment_dir)/'manifests/target_eval.csv')
        sample=next(r for r in records if r['sample_id']==row['sample_id'])
        full_raw=u.read_rgb(sample['raw_path']).unsqueeze(0).to(device)
        crop=row['crop']; top,left,size=crop['top'],crop['left'],crop['patch_size']
        with torch.no_grad():
            before=tta.model(raw)
            full_before=tta.model(full_raw)[...,top:top+size,left:left+size]
            tta.model.load_state_dict(u.load_checkpoint(root/'run/snapshots'/f'{step:06d}.pth')['student'])
            after=tta.model(raw)
            full_after=tta.model(full_raw)[...,top:top+size,left:left+size]
            result['same_crop_update']=dict(mae=float((after-before).abs().mean()),before=tensor_stats(before),after=tensor_stats(after))
            result['crop_vs_full']=dict(before_mae=float((before-full_before).abs().mean()),
                     after_mae=float((after-full_after).abs().mean()),full_region_update=float((full_after-full_before).abs().mean()))
        write_json(folder/f'{step:06d}.json',result)
        save_checkpoint(folder/f'{step:06d}_outputs.pt',dict(before=cpu_tree(before),after=cpu_tree(after)))
        # 显示范围固定，保存预览不改变用于诊断的浮点张量。
        previews=dict(raw=values['raw'],pseudo=values['pseudo'],regenerated=values['regenerated'],
                      before=before,after=after,regenerated_output=values['regenerated_output'])
        for name,tensor in previews.items():
            a=tensor.detach().cpu()[0].clamp(0,1).permute(1,2,0).numpy()
            Image.fromarray(np.rint(a*255).astype(np.uint8)).save(folder/f'{step:06d}_{name}.png')
    write_json(folder/'status.json',dict(status='complete',steps=args.inspect_steps))


def main(argv=None):
    from options import parse_diagnostic_args
    args=parse_diagnostic_args(argv)
    # 在第一次CUDA调用之前配置；仅影响本诊断进程，不改其他训练任务的环境。
    if fixed_protocol(args) and args.phase!='fisher_probe':
        workspace=os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        if workspace not in (':4096:8',':16:8'): raise ValueError('Unsupported CUBLAS_WORKSPACE_CONFIG')
        torch.use_deterministic_algorithms(True,warn_only=False)
    u.seed_everything(23)
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    try:
        return {'audit':static_audit,'accept':acceptance,'run':full_run,'evaluate':offline_evaluate,
            'inspect':inspect_steps,'fisher_probe':fisher_probe,
            'numeric_control':numeric_control,'compare_numeric':compare_numeric}[args.phase](args)
    except Exception as exc:
        # 保留既有成功/失败产物，只追加唯一错误报告，不覆盖旧状态文件。
        stamp=time.time_ns()
        write_json(Path(args.output_dir)/f'failure_{args.phase}_{stamp}.json',
                   dict(status='incomplete',phase=args.phase,error=repr(exc),environment=environment()))
        raise


if __name__=='__main__':
    main()
