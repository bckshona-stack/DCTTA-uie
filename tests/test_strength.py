"""G4/G5：端点、原梯度路径、容量原子性、真实RDDM验收/中断恢复。"""
import os
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image
import run_strength_uie as s
from test_migration import engine, DeterministicGenerator


def directory():
    path = Path(__file__).resolve().parents[2]/'tmp'/('strength_test_'+uuid.uuid4().hex)
    path.mkdir(parents=True)
    return path


class StrengthTests(unittest.TestCase):
    def test_mix_endpoints_invalid_and_no_mutation(self):
        raw = torch.rand(1,3,8,8,requires_grad=True)
        generated = torch.rand_like(raw)
        original = generated.clone()
        self.assertTrue(torch.equal(s.mix_input(raw,generated,0),raw))
        self.assertTrue(torch.equal(s.mix_input(raw,generated,1),generated))
        self.assertTrue(torch.equal(s.mix_input(raw,generated,.5),.5*raw+.5*generated))
        self.assertFalse(s.mix_input(raw,generated,.5).requires_grad)
        self.assertTrue(torch.equal(generated,original))
        for bad in [generated+2, generated*float('nan'), generated.double(), generated[:,:,:4]]:
            with self.assertRaises(ValueError): s.mix_input(raw,bad,.5)
        with self.assertRaises(ValueError): s.mix_input(raw,generated,2)
        x=torch.zeros(1,3,4,4);x[:,1]=1;x[:,2]=.5
        stats=s.channel_stats(x)
        self.assertEqual(stats['exact_zero'],[1,0,0])
        self.assertEqual(stats['exact_one'],[0,1,0])

    def test_both_losses_share_actual_mixed_student_output(self):
        s.u.seed_everything(23)
        t=engine(compute_fisher=0)
        raw=torch.rand(1,3,32,40)
        generator=DeterministicGenerator()
        proxy=s.MixedGenerator(generator,.5)
        seen=[];components=[]
        original=t.compute_loss2
        def loss(pred,target,*args):
            components.append(pred)
            return original(pred,target,*args)
        t.compute_loss2=loss
        def observer(event,values):
            if event=='student_outputs':seen.append(values)
        t.observer=observer
        result=t.adapt(raw,['a.png'],proxy)
        generated=.8*raw+.2*generator.calls[0][1]
        self.assertTrue(torch.equal(seen[0]['regenerated'],.5*raw+.5*generated))
        self.assertEqual(len(components),2)
        self.assertIs(components[0],components[1])
        self.assertTrue(components[0].requires_grad)
        self.assertFalse(seen[0]['raw_output'].requires_grad)
        self.assertAlmostEqual(result['loss'],result['loss_s']+result['loss_t'],places=5)

    def test_checkpoint_budget_and_path_safety(self):
        root=directory(); out=s.BudgetOutput(root,1_000_000)
        value={'state':torch.rand(32),'cursor':3}
        digest=out.checkpoint('resume.pth',value)
        self.assertEqual(digest,s.v.sha(root/'resume.pth'))
        self.assertTrue(torch.equal(s.u.load_checkpoint(root/'resume.pth')['state'],value['state']))
        out.limit=out.used()+1
        with self.assertRaises(RuntimeError):out.checkpoint('resume.pth',value)
        self.assertEqual(digest,s.v.sha(root/'resume.pth'))
        with self.assertRaises(ValueError):out.checkpoint('../escape.pth',value)
        with self.assertRaises(ValueError):s.verify_index(root,{'../escape':'x'})

    def test_preflight_rejects_nonempty_and_overlapping_output(self):
        root=directory();out=root/'out';out.mkdir();(out/'keep').write_text('keep')
        args=SimpleNamespace(output_dir=str(out),experiment_dir=str(root/'exp'),diagnostic_dir=str(root/'diag'),
                             factorial_dir=str(root/'fac'),max_gb=10)
        with self.assertRaises(FileExistsError):s.preflight(args)
        args.output_dir=args.experiment_dir
        with self.assertRaises(ValueError):s.preflight(args)
        self.assertEqual((out/'keep').read_text(),'keep')

    def test_comparison_uses_all_six_groups_and_paired_sets(self):
        root=directory();out=s.BudgetOutput(root/'out',10_000_000)
        out.json('plan.json',{'fixture':True})
        args=SimpleNamespace(diagnostic_dir=str(root/'diag'),factorial_dir=str(root/'fac'))
        protocol={'order':['t.png'],'probes':{'target_eval':['t.png'],'source_val':['s.png']}}
        def rows(delta=0):
            return [dict(sample_id='t.png',psnr=20.+delta,ssim=.8),dict(sample_id='s.png',psnr=22.,ssim=.9)]
        roots={'G0':root/'diag/evaluation'}
        roots.update({g:root/'fac'/g/'evaluation' for g in ['G1','G2','G3']})
        roots.update({g:out.root/g/'evaluation' for g in s.GROUPS})
        for i,(group,folder) in enumerate(roots.items()):
            s.f.write(folder/'000000_student.json',rows())
            s.f.write(folder/'000001_student.json',rows(i-3))
            if group in s.GROUPS:
                out.json(group+'/status.json',dict(status='complete',plan_sha256=s.v.sha(out.root/'plan.json'),
                    files={str((folder/'000001_student.json').relative_to(out.root)):s.v.sha(folder/'000001_student.json')}))
        with patch.object(s,'preflight',return_value=(out,protocol,{},[],[0,1])):s.compare(args)
        result=s.v.read(out.root/'comparison/summary.json')
        self.assertEqual(set(result),set(roots))
        self.assertEqual(result['G0']['target']['worse_over_1db'],1)
        self.assertEqual(result['G5']['target']['mean_delta'],2.)
        self.assertEqual(result['G5']['source_val']['mean_delta'],0.)
        self.assertTrue((out.root/'comparison/G0_G5_probe_psnr.png').is_file())

    @unittest.skipUnless(os.environ.get('DCTTA_TEST_RDDM')=='1','Enable real RDDM integration')
    def test_real_acceptance_workflow_resume_and_tamper(self):
        if shutil.disk_usage(Path(__file__).resolve().parents[2]).free < 4*1024**3:
            self.skipTest('Need 4 GiB for retained real-RDDM states')
        from RDDM.net import ResidualDiffusionModel
        root=directory();raw=root/'raw';raw.mkdir();diag=root/'diag';fac=root/'fac'
        for i,shape in enumerate([(37,49),(40,48)]):
            Image.fromarray(np.random.default_rng(i+8).integers(0,256,(*shape,3),dtype=np.uint8)).save(raw/('%d.png'%i))
        samples=[SimpleNamespace(sample_id=p.name,raw_path=p,reference_path=p) for p in sorted(raw.glob('*.png'))]
        data=s.u.TargetRawDataset(samples,32,23);order=[x.sample_id for x in samples]
        groups={'target_eval':samples,'source_val':[], 'source_train':[]}
        chosen=[dict(sample_id=x.sample_id,source_psnr=20.,width=49 if i==0 else 48,height=37 if i==0 else 40) for i,x in enumerate(samples)]
        protocol={'order':order,'probes':{'target_eval':order}}
        out=s.BudgetOutput(root/'output',10_000_000_000)
        out.json('plan.json',{'fixture':True})
        args=SimpleNamespace(experiment_dir=str(root/'exp'),diagnostic_dir=str(diag),factorial_dir=str(fac),
                             output_dir=str(out.root),device='cpu',max_gb=10)
        s.v.runtime();s.u.seed_everything(23)
        with s.d.numerical_mode(True):
            t=engine(); t.fisher_identity=t.cache_identity(data)
            t.compute_fisher(s.u.make_raw_loader(data,shuffle=False))
            g=ResidualDiffusionModel(SimpleNamespace(device='cpu',verbose=False,save_rddm_results=False),
                                     image_size=32,device='cpu',results_folder=root/'rddm')
            g.trainer.ema.step.fill_(99)
            initial=s.d.capture(t,g,0,order); update=t.update_teacher
            def fixture_build(*args):
                s.d.restore(t,g,initial);t.update_teacher=update
                return t,g,data,order,SimpleNamespace(seed=23),initial
            def fixture_preflight(args):return out,protocol,groups,chosen,[0,1,2]
            # 旧端点用原适配代理生成，不使用新增MixedGenerator。
            for group,fixed,use_raw in [('G0',False,False),('G1',True,False),('G2',False,True),('G3',True,True)]:
                s.d.restore(t,g,initial);t.update_teacher=(lambda:None) if fixed else update
                losses=t.adapt(data[0][1].unsqueeze(0),[order[0]],s.f.GeneratorInput(g,use_raw))
                if group=='G0':
                    s.f.write(diag/'run/trajectory.json',[dict(losses=losses,state_sha256=s.d.digest(s.d.capture(t,g,1,order)))])
                else:
                    s.f.write(fac/group/'trajectory.json',[dict(losses=losses),dict(losses={})])
                    s.f.save_checkpoint(fac/group/'snapshots/000001.pth',dict(student=s.d.cpu_tree(t.model.state_dict()),teacher=s.d.cpu_tree(t.model_teacher.state_dict())))
            with patch.object(s,'preflight',fixture_preflight),patch.object(s,'build',fixture_build),patch.object(s.d,'acceptance_length',return_value=2):
                s.acceptance(args)
                self.assertTrue(s.v.read(out.root/'acceptance.json')['passed'])
                with patch.object(s,'build',side_effect=AssertionError('Repeated acceptance')):s.acceptance(args)
                # 独立连续基准；生产运行含每步诊断和磁盘断点，末态须完全一致。
                for group in s.RUN_ORDER:
                    s.d.restore(t,g,initial);t.update_teacher=(lambda:None) if s.GROUPS[group]['fixed_teacher'] else update
                    expected=[s.adapt_one(t,g,data,order,i,.5) for i in range(2)]
                    original_json=s.BudgetOutput.json
                    def interrupted(self,name,value):
                        answer=original_json(self,name,value)
                        if name==group+'/progress.json' and value['committed_step']==1:
                            raise RuntimeError('committed interruption')
                        return answer
                    with patch.object(s.BudgetOutput,'json',interrupted):
                        with self.assertRaisesRegex(RuntimeError,'committed interruption'):s.run_group(args,group)
                    s.run_group(args,group)
                    actual=s.v.read(out.root/group/'trajectory.json')
                    self.assertEqual(actual,expected)
                    status=s.v.read(out.root/group/'status.json')
                    self.assertEqual(status['images'],32)
                    with patch.object(s,'build',side_effect=AssertionError('Repeated completed group')):s.run_group(args,group)
                # 模拟未提交输出损坏不涉及本测试：已提交文件必须拒绝。
                name=next(iter(s.v.read(out.root/'G4/status.json')['files']))
                (out.root/name).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Committed artifact changed'):s.run_group(args,'G4')


if __name__=='__main__':unittest.main()
