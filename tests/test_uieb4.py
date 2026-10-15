"""新数据协议与新G0：组划分、统计、事务恢复、真实RDDM可选集成。"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import prepare_uieb4_experiment as p
import run_uieb4_g0 as r
from test_migration import engine, TinyFeatures, LowFrequency, tiny_model


class ToyEMA(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.ema_model=deepcopy(model)


class ToyGenerator:
    """编排测试替身；真实RDDM数值/状态覆盖由独立集成测试完成。"""
    def __init__(self):
        model=torch.nn.Conv2d(3,3,1)
        self.trainer=SimpleNamespace(model=model,ema=ToyEMA(model),opt0=torch.optim.Adam(model.parameters()),
            step=0,accelerator=SimpleNamespace(scaler=None),save_results=False)

    def train(self,raw,pseudo,names):
        tr=self.trainer;tr.opt0.zero_grad()
        (tr.model(raw)-pseudo.detach()).square().mean().backward();tr.opt0.step();tr.step+=1
        tr.ema.ema_model.load_state_dict(tr.model.state_dict())
        return tr.ema.ema_model(raw).sigmoid().detach()

    def sample(self,raw,names):return self.trainer.ema.ema_model(raw).sigmoid()


def directory():
    root=Path(__file__).resolve().parents[2]/'tmp'/('uieb4_test_'+uuid.uuid4().hex)
    root.mkdir(parents=True)
    return root


class UIEB4Tests(unittest.TestCase):
    def test_runtime_guard_only_allows_line_ending_difference(self):
        root=directory();folder=root/'docs/11_20261015_uieb4_source_g0';folder.mkdir(parents=True)
        file=root/'module.py';file.write_bytes(b'a=1\r\nb=2\r\n')
        import hashlib
        (folder/'RUNTIME_FILES.json').write_text(json.dumps({'module.py':hashlib.sha256(b'a=1\nb=2\n').hexdigest()}))
        with patch.object(p,'ROOT',root):
            p.verify_runtime()
            file.write_bytes(b'a=3\nb=2\n')
            with self.assertRaises(ValueError):p.verify_runtime()

    def test_stratification_preserves_groups_and_seed(self):
        rows=[]
        for c in p.CATEGORIES:
            for i in range(7):
                rows.append(dict(sample_id=c+str(i),scene_id=c+str(i//2),primary_category=c))
        train,val=p.stratify(rows,{c:3 for c in p.CATEGORIES},2022)
        self.assertEqual(len(val),12)
        self.assertFalse({x['scene_id'] for x in train}&{x['scene_id'] for x in val})
        self.assertEqual((train,val),p.stratify(rows,{c:3 for c in p.CATEGORIES},2022))
        with self.assertRaises(ValueError):p.stratify(rows,{c:8 for c in p.CATEGORIES},2022)

    def test_immutable_prevents_overwrite(self):
        f=directory()/'a'
        p.immutable(f,b'original');p.immutable(f,b'original')
        with self.assertRaises(ValueError):p.immutable(f,b'changed')
        self.assertEqual(f.read_bytes(),b'original')

    def test_selection_independent_of_target_scores_and_macro(self):
        rows=[dict(sample_id=str(i),primary_category=p.CATEGORIES[i%4],width=i+32,height=40,
                   psnr=float(100-i)) for i in range(12)]
        selected=r.selection(rows)
        for row in rows:row['psnr']=-999
        self.assertEqual(selected,r.selection(rows));self.assertEqual(len(set(selected)),10)
        records=[dict(sample_id=x['sample_id'],psnr=float(i),ssim=.8) for i,x in enumerate(rows)]
        result=r.aggregate(records,{x['sample_id']:x for x in rows})
        self.assertEqual(result['all']['count'],12)
        self.assertEqual(result['macro_category']['psnr'],5.5)

    def test_resume_pointer_never_points_to_partial_checkpoint(self):
        out=r.s.BudgetOutput(directory(),10_000_000)
        r.commit(out,dict(cursor=0,value=torch.ones(3)),dict(files={}))
        first=r.v.read(out.root/'progress.json')
        original=out.json
        def interrupted(name,value):
            if name=='progress.json':raise RuntimeError('power loss before pointer')
            return original(name,value)
        with patch.object(out,'json',interrupted):
            with self.assertRaises(RuntimeError):r.commit(out,dict(cursor=1,value=torch.zeros(3)),dict(files={}))
        self.assertEqual(r.v.read(out.root/'progress.json'),first)
        self.assertEqual(p.sha(out.root/first['checkpoint']),first['sha256'])
        r.commit(out,dict(cursor=1,value=torch.zeros(3)),dict(files={}))
        self.assertEqual(r.v.read(out.root/'progress.json')['cursor'],1)

    def test_execute_interrupted_resume_complete_and_tamper(self):
        root=directory();rows=[];samples=[]
        for i,c in enumerate(p.CATEGORIES):
            path=root/('%d.png'%i)
            Image.fromarray(np.random.default_rng(i).integers(0,256,(37,41,3),dtype=np.uint8)).save(path)
            rows.append(dict(sample_id=path.name,primary_category=c,width=41,height=37))
            samples.append(r.u.Sample(path.name,path,path))
        exp=root/'experiment';(exp/'manifests').mkdir(parents=True)
        (exp/'manifests/audit.json').write_text('{}')
        checkpoint=exp/'best.pth';checkpoint.write_bytes(b'fixture')
        groups=dict(target_eval=samples,source_val=samples,source_train=[])
        r.u.seed_everything(23);t=engine(compute_fisher=0);g=ToyGenerator()
        data=r.u.TargetRawDataset(samples,32,23);order=[s.sample_id for s in samples]
        r.v.runtime()
        with r.d.numerical_mode(True):
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            initial=r.d.capture(t,g,0,order)
            def build(*args):
                r.d.restore(t,g,initial)
                return t,g,data,order,SimpleNamespace(seed=23)
            args=SimpleNamespace(experiment_dir=str(exp),max_gb=10,device='cpu',phase='run')
            original=r.commit
            def interrupt(out,state,progress):
                original(out,state,progress)
                if state['cursor']==1:raise RuntimeError('committed power loss')
            with patch.object(r,'inputs',return_value=(groups,rows,checkpoint)),\
                 patch.object(r,'selection',return_value=order.copy()),\
                 patch.object(r,'build',side_effect=build),patch.object(r,'acceptance'),\
                 patch.object(r.v,'schedule',return_value=[0,2,4]):
                with patch.object(r,'commit',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'committed power loss'):r.execute(args)
                self.assertEqual(r.v.read(exp/'G0/progress.json')['cursor'],1)
                r.execute(args)
                final=r.signature(t,g,4,order)
                r.d.restore(t,g,initial)
                for i in range(4):r.adapt(t,g,data,order,i)
                self.assertEqual(final,r.signature(t,g,4,order))
                with patch.object(r,'build',side_effect=AssertionError('Completed run rebuilt')):r.execute(args)
                status=r.v.read(exp/'G0/complete.json')
                self.assertEqual(status['steps'],4)
                self.assertEqual(len(r.v.read(exp/'G0/trajectory.json')),4)
                target=exp/'G0/final_y_in'/order[0]
                target.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):r.execute(args)

    @unittest.skipUnless(os.environ.get('DCTTA_TEST_RDDM')=='1','Enable real RDDM integration')
    def test_production_build_reuses_only_matching_new_fisher(self):
        root=directory();samples=[]
        for i in range(2):
            path=root/('%d.png'%i)
            Image.fromarray(np.random.default_rng(i).integers(0,256,(37,41,3),dtype=np.uint8)).save(path)
            samples.append(r.u.Sample(path.name,path,path))
        checkpoint=root/'best.pth';model=tiny_model()
        torch.save(dict(net=model.backbone.state_dict(),model_config=model.model_config,
                        artifact_type='source_training'),checkpoint)
        out=r.s.BudgetOutput(root/'out',10_000_000_000)
        original=r.u.SRTTA
        def cpu_loss(opt,model,optimizer):
            return original(opt,model,optimizer,perceptual_loss=TinyFeatures(),wavelet=LowFrequency())
        r.v.runtime()
        with r.d.numerical_mode(True),patch.object(r.u,'SRTTA',side_effect=cpu_loss):
            t,g,data,order,opt=r.build(SimpleNamespace(device='cpu'),out,{'target_eval':samples},checkpoint)
            expected=r.signature(t,g,0,order)
            del t,g
            t,g,data,order,opt=r.build(SimpleNamespace(device='cpu'),out,{'target_eval':samples},checkpoint)
            self.assertEqual(expected,r.signature(t,g,0,order))
            corrupted=r.u.load_checkpoint(out.root/'fisher.pt')
            corrupted['identity']['source_checkpoint_sha256']='wrong'
            out.checkpoint('fisher.pt',corrupted)
            del t,g
            with self.assertRaisesRegex(ValueError,'Fisher identity mismatch'):
                r.build(SimpleNamespace(device='cpu'),out,{'target_eval':samples},checkpoint)

    @unittest.skipUnless(os.environ.get('DCTTA_TEST_RDDM')=='1','Enable real RDDM integration')
    def test_real_rddm_acceptance_observation_and_resume(self):
        from RDDM.net import ResidualDiffusionModel
        root=directory();raw=root/'raw';raw.mkdir()
        samples=[]
        for i,shape in enumerate(((37,49),(40,48))):
            path=raw/('%d.png'%i)
            Image.fromarray(np.random.default_rng(12+i).integers(0,256,(*shape,3),dtype=np.uint8)).save(path)
            samples.append(r.u.Sample(path.name,path,path))
        data=r.u.TargetRawDataset(samples,40,23)
        order=[s.sample_id for s in samples]
        groups=dict(target_eval=samples,source_val=samples)
        out=r.s.BudgetOutput(root/'out',10_000_000_000)
        r.v.runtime()
        with r.d.numerical_mode(True):
            r.u.seed_everything(23)
            t=engine();t.fisher_identity=t.cache_identity(data)
            t.compute_fisher(r.u.make_raw_loader(data,shuffle=False))
            g=ResidualDiffusionModel(SimpleNamespace(device='cpu',verbose=False,save_rddm_results=False),
                image_size=40,device='cpu',results_folder=root/'rddm')
            g.trainer.ema.step.fill_(99)
            initial=r.d.capture(t,g,0,order)
            out.checkpoint('initial.pth',initial)
            with patch.object(r.d,'acceptance_length',return_value=2):
                r.acceptance(t,g,data,order,SimpleNamespace(seed=23),out,order,groups,{'test':True})
            self.assertEqual(r.signature(t,g,0,order),r.d.digest(initial))
            r.adapt(t,g,data,order,0)
            r.commit(out,r.d.capture(t,g,1,order),dict(files={}))
            r.adapt(t,g,data,order,1)
            expected=r.signature(t,g,2,order)
            pointer=r.v.read(out.root/'progress.json')
            saved=r.u.load_checkpoint(out.root/pointer['checkpoint'])
            r.d.restore(t,g,saved['state'])
            before=r.signature(t,g,1,order)
            files=r.observe(t,g,groups,order,1,order,SimpleNamespace(seed=23),out,{n:20. for n in order})
            self.assertEqual(before,r.signature(t,g,1,order))
            r.v.verify_files(out,files)
            for role in ('teacher_pseudo','x_sd','y_sd','y_in'):
                self.assertEqual(len(list((out.root/role/'snapshots/000001').glob('*.png'))),2)
            r.adapt(t,g,data,order,1)
            self.assertEqual(expected,r.signature(t,g,2,order))


if __name__=='__main__':unittest.main()
