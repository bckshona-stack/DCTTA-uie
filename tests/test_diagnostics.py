"""诊断观测不改变更新轨迹；完整状态恢复包含 RDDM、优化器、EMA 和 RNG。"""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from test_migration import engine, test_directory
from diagnose_uie import capture, restore, digest, run_segment, tensor_stats
from uie_tta import TargetRawDataset, select_samples, make_raw_loader, load_checkpoint


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        if self._testMethodName in ['test_real_observer_and_full_replay','test_static_build_evaluation_and_inspection']:
            import shutil
            if shutil.disk_usage(Path(__file__).resolve().parents[2]).free<5*1024**3:
                self.skipTest('Need 5 GiB free for retained real-RDDM diagnostic test artifacts')

    def test_stats_do_not_normalize_or_mutate(self):
        x=torch.tensor([[[[-1.,2.]],[[.3,.4]],[[.5,.6]]]])
        before=x.clone(); stats=tensor_stats(x)
        self.assertEqual(stats['min'],-1.)
        self.assertEqual(stats['max'],2.)
        self.assertTrue(torch.equal(x,before))

    def test_nonfinite_gradient_and_disk_shortage_stop(self):
        from unittest.mock import patch
        from diagnose_uie import parameter_summary, require_space, state_bytes
        layer=torch.nn.Linear(1,1)
        layer.weight.grad=torch.full_like(layer.weight,float('nan'))
        with self.assertRaises(FloatingPointError): parameter_summary(layer)
        self.assertEqual(state_bytes({'a':torch.zeros(4),'b':[np.zeros(2,dtype=np.float64)]}),32)
        with patch('diagnose_uie.shutil.disk_usage',return_value=SimpleNamespace(free=119)):
            with self.assertRaises(OSError): require_space('.',100)

    def test_fisher_comparison_does_not_accept_small_mask_changes(self):
        from diagnose_uie import compare_fishers
        old={'bias':[torch.tensor([1e-22,2e-22]),torch.tensor([0.,1.])]}
        new={'bias':[torch.tensor([2e-22,1e-22]),torch.tensor([1.,0.])]}
        report=compare_fishers(old,new)
        self.assertFalse(report['exact_match'])
        self.assertEqual(report['mask_changed'],2)
        self.assertLess(report['max_absolute_error'],1e-20)
        self.assertTrue(compare_fishers(old,old)['exact_match'])

    def test_formal_acceptance_requires_22_images_and_ema_boundary(self):
        from diagnose_uie import acceptance_length
        ids=[str(i) for i in range(22)]
        data=SimpleNamespace(samples=[SimpleNamespace(sample_id=n) for n in ids],
                             crops=[{'patch_size':152}]+[{'patch_size':320}]*21)
        opt=SimpleNamespace(patch_size=320)
        self.assertEqual(acceptance_length(22,data,ids,opt,SimpleNamespace(update_after_step=100),5),22)
        with self.assertRaisesRegex(ValueError,'at least 22'):
            acceptance_length(21,data,ids,opt,SimpleNamespace(update_after_step=100),5)
        with self.assertRaisesRegex(ValueError,'warmup'):
            acceptance_length(22,data,ids,opt,SimpleNamespace(update_after_step=200),5)

    def test_fisher_probe_measures_without_rddm_or_acceptance(self):
        import json
        from unittest.mock import patch
        from test_source_training import SourceTrainingTests
        from test_migration import TinyFeatures, DeterministicGenerator
        from train_source_uie import train_source, save_checkpoint
        from adapt_target_uie import main as target_main
        from diagnose_uie import fisher_probe, build
        fixture=SourceTrainingTests(); fixture.setUp()
        opt,_=fixture.prepare(['--epochs','1']); train_source(opt)
        root=fixture.root/'probe_output'; root.mkdir()
        args=SimpleNamespace(experiment_dir=opt.experiment_dir,output_dir=str(root),device='cpu')
        with patch('utils.loss_utils.PerceptualLoss',TinyFeatures):
            with patch('RDDM.net.ResidualDiffusionModel',return_value=DeterministicGenerator()):
                target_main(['--experiment_dir',opt.experiment_dir,'--device','cpu','--patch_size','32'])
            old_path=Path(opt.experiment_dir)/'dctta/fisher.pt'
            old=load_checkpoint(old_path)
            next(iter(old['fishers'].values()))[0][0]+=1e-4
            save_checkpoint(old_path,old)
            before=torch.are_deterministic_algorithms_enabled()
            with patch('RDDM.net.ResidualDiffusionModel',side_effect=AssertionError('Probe initialized RDDM')):
                fisher_probe(args)
                with self.assertRaisesRegex(ValueError,'Recomputed Fisher differs'):
                    build(opt.experiment_dir,'cpu',root/'strict_gate')
            report=json.loads((root/'fisher_probe/report.json').read_text())
            self.assertTrue(report['repeats']['exact_match'])
            self.assertFalse(report['original_to_repeat'][0]['exact_match'])
            self.assertEqual(report['status'],'measured_not_accepted')
            self.assertEqual(report['strict_single_image']['status'],'completed')
            self.assertEqual(torch.are_deterministic_algorithms_enabled(),before)
            self.assertFalse((root/'acceptance/acceptance.json').exists())

    @unittest.skipUnless(os.environ.get('DCTTA_TEST_RDDM')=='1','Real RDDM diagnostic acceptance')
    def test_real_observer_and_full_replay(self):
        from RDDM.net import ResidualDiffusionModel
        with test_directory() as directory:
            root=Path(directory); raw=root/'raw'; raw.mkdir()
            for i,side in enumerate([32,40]):
                Image.fromarray(np.random.default_rng(i).integers(0,256,(side,side+8,3),dtype=np.uint8)).save(raw/f'{i}.png')
            data=TargetRawDataset(select_samples(raw),40,23); order=[s.sample_id for s in data.samples]
            t=engine(); t.fisher_identity=t.cache_identity(data)
            t.compute_fisher(make_raw_loader(data,shuffle=False))
            g=ResidualDiffusionModel(SimpleNamespace(device='cpu',verbose=False,save_rddm_results=False),
                                     image_size=40,device='cpu',results_folder=root/'rddm')
            # 提前设到预热边界，验证计数、有效更新分支以及buffer恢复。
            g.trainer.ema.step.fill_(99)
            initial=capture(t,g,0,order)
            off=run_segment(t,g,data,order,root/'off',0,2)
            restore(t,g,initial)
            self.assertEqual(digest(initial),digest(capture(t,g,0,order)))
            on=run_segment(t,g,data,order,root/'on',0,2,True,True)
            self.assertEqual(off,on)
            middle=load_checkpoint(root/'on/replay_states/000001.pth')
            restore(t,g,middle)
            replay=run_segment(t,g,data,order,root/'replay',1,2)
            self.assertEqual(replay,off[1:])
            import json
            events=json.loads((root/'on/steps.jsonl').read_text().splitlines()[0])['events']
            for name,count in [('loss_components',2),('rddm_input',5),('rddm_after',5),('sample_prediction',5)]:
                self.assertEqual(sum(e['event']==name for e in events),count)

    @unittest.skipUnless(os.environ.get('DCTTA_TEST_RDDM')=='1','Real diagnostic workflow integration')
    def test_static_build_evaluation_and_inspection(self):
        import json
        from unittest.mock import patch
        from test_source_training import SourceTrainingTests
        from test_migration import TinyFeatures
        from train_source_uie import train_source, save_checkpoint, write_json
        from adapt_target_uie import main as target_main
        import diagnose_uie as d
        from uie_tta import sha256_file
        from copy import deepcopy
        fixture=SourceTrainingTests(); fixture.setUp()
        opt,_=fixture.prepare(['--epochs','1']); train_source(opt)
        root=fixture.root/'diagnostic'; root.mkdir()
        args=SimpleNamespace(experiment_dir=opt.experiment_dir,output_dir=str(root),device='cpu',
                             severity_scores=None,inspect_steps=[1],acceptance_steps=22,protocol='fixed_fisher_strict')
        with patch('utils.loss_utils.PerceptualLoss',TinyFeatures):
            target_main(['--experiment_dir',opt.experiment_dir,'--device','cpu','--run_all','--patch_size','32'])
            exp=Path(opt.experiment_dir); old=load_checkpoint(exp/'dctta/fisher.pt')
            contract=dict(files={n:sha256_file(exp/n) for n in ['dctta/fisher.pt','dctta/run.json','source/checkpoints/best.pth']},
                          legacy_implementation=old['identity']['implementation'],current_implementation={})
            with patch.object(d,'FIXED_FISHER_CONTRACT',contract), d.numerical_mode(True):
                d.static_audit(args)
                self.assertTrue(json.loads((root/'static_audit/status.json').read_text())['reload_passed'])
                # 只缩小合成样本数；控制、重复、观测、恢复、默认对照均用真实RDDM执行。
                with patch.object(d,'acceptance_length',return_value=2), patch('uie_tta.SRTTA.compute_fisher',side_effect=AssertionError('Fixed mode recomputed Fisher')):
                    d.acceptance(args)
                    self.assertTrue(json.loads((root/'acceptance/acceptance.json').read_text())['passed'])
                    with self.assertRaises(FileNotFoundError): d.full_run(args)
                    d.numeric_control(args); d.compare_numeric(args); d.full_run(args)
                d.offline_evaluate(args); d.inspect_steps(args)
                self.assertEqual(sha256_file(root/'run/fisher.pt'),contract['files']['dctta/fisher.pt'])
                self.assertEqual(sha256_file(exp/'dctta/fisher.pt'),contract['files']['dctta/fisher.pt'])
                # 即使文件仍可加载，未经批准的替换和目标身份变更都必须被拒绝。
                t=engine(); t.fisher_identity=deepcopy(old['identity']); t.fisher_identity['fisher_ratio']=.5
                with self.assertRaisesRegex(ValueError,'identity mismatch'): d.load_fixed_fisher(t,exp,root/'bad_identity')
                bad=deepcopy(contract); bad['files']['dctta/fisher.pt']='0'*64
                with patch.object(d,'FIXED_FISHER_CONTRACT',bad):
                    with self.assertRaisesRegex(ValueError,'artifact changed'): d.verify_fixed_files(exp)
            self.assertTrue((root/'DIAGNOSTIC_REPORT.md').is_file())
            self.assertTrue((root/'evaluation/psnr.png').is_file())
            self.assertTrue((root/'numeric_comparison/psnr.png').is_file())
            self.assertEqual(json.loads((root/'inspection/000001.json').read_text())['step'],1)


if __name__=='__main__': unittest.main()
