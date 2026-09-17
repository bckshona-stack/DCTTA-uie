"""整图诊断的容量、选图、PSNR 与真实 RDDM 轨迹隔离验证。"""
import os
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import diagnose_visual_uie as v


def retained_directory():
    root=Path(__file__).resolve().parents[2]/'tmp'/('visual_test_'+uuid.uuid4().hex)
    root.mkdir(parents=True)
    return root


class VisualTests(unittest.TestCase):
    def test_selection_keeps_improvements_and_unique_sizes(self):
        rows=[dict(sample_id='%02d.png'%i,reason='',width=32+i,height=40,
                   source_psnr=20.,g0_psnr=20+i-10,delta_psnr=i-10.) for i in range(12)]
        selected=v.select_ten(rows)
        self.assertEqual(len({r['sample_id'] for r in selected}),10)
        self.assertEqual([r['sample_id'] for r in selected[:3]],['00.png','01.png','02.png'])
        self.assertEqual(selected[6]['sample_id'],'11.png')
        self.assertIn('最大改善',selected[6]['reason'])
        self.assertIn('改善样本不足',selected[7]['reason'])
        root=retained_directory(); (root/'pick.csv').write_bytes(v.csv_bytes(selected))
        self.assertEqual(v.selection(root/'pick.csv',rows),selected)
        (root/'bad.csv').write_bytes(v.csv_bytes(selected[:-1]+[selected[0]]))
        with self.assertRaises(ValueError):v.selection(root/'bad.csv',rows)

    def test_schedule_and_budget_atomic_replacement(self):
        self.assertEqual(v.schedule(133),[0,14,27,40,54,67,80,94,107,120,133])
        root=retained_directory(); output=v.Output(root,12)
        output.put('a',b'123456')
        with self.assertRaises(RuntimeError):output.put('b',b'1234567')
        self.assertEqual((root/'a').read_bytes(),b'123456')
        self.assertFalse((root/'b').exists())
        output.put('a',b'654321')
        self.assertEqual((root/'a').read_bytes(),b'654321')
        with self.assertRaises(ValueError):output.put('../escape',b'x')
        with self.assertRaises(ValueError):v.independent_output(root,[root/'nested'])
        with self.assertRaises(ValueError):v.independent_output(root/'nested',[root])

    def test_prepare_refuses_unrelated_output(self):
        root=retained_directory()
        out=root/'output'; out.mkdir(); (out/'keep.txt').write_text('keep')
        args=SimpleNamespace(output_dir=str(out),experiment_dir=str(root/'exp'),
            diagnostic_dir=str(root/'diag'),max_gb=10,selection_csv=None)
        with self.assertRaises(FileExistsError):v.prepare(args)
        self.assertEqual((out/'keep.txt').read_text(),'keep')

    def test_png_and_psnr_match_legacy(self):
        v.runtime(); torch=v.torch
        from utils.val_utils import compute_psnr_ssim
        root=retained_directory(); output=v.Output(root,1_000_000)
        pred=torch.linspace(-.1,1.1,3*37*49).reshape(1,3,37,49)
        ref=torch.full_like(pred,.4)
        self.assertEqual(v.psnr(pred,ref),float(compute_psnr_ssim(pred,ref)[0]))
        digest,q=v.export_image(output,'sample.png',pred)
        self.assertEqual(digest,v.sha(root/'sample.png'))
        from PIL import Image
        with Image.open(root/'sample.png') as image:self.assertEqual(image.size,(49,37))
        self.assertEqual(v.psnr(q,q),float('inf'))
        with self.assertRaises(ValueError):v.psnr(pred*float('nan'),ref)
        self.assertEqual(v.sample_seed(23,'x'),v.sample_seed(23,'x'))
        self.assertNotEqual(v.sample_seed(23,'x'),v.sample_seed(23,'y'))

    def test_query_chunks_match_dense_and_keep_all_keys(self):
        v.runtime(); torch=v.torch
        from RDDM.src.residual_denoising_diffusion_pytorch import Attention
        layer=Attention(16,heads=4,dim_head=8).eval()
        image=torch.randn(1,16,17,23)
        generator=SimpleNamespace(trainer=SimpleNamespace(ema=SimpleNamespace(ema_model=layer)))
        before=v.d.digest(layer.state_dict()); shapes=[]
        original_einsum=torch.einsum
        def measured(equation,*args):
            if equation=='b h d i, b h d j -> b h i j':
                shapes.append((args[0].shape[-1],args[1].shape[-1]))
            return original_einsum(equation,*args)
        with torch.no_grad():
            expected=layer(image)
            with patch.object(torch,'einsum',measured), v.bounded_sampling_attention(generator,chunk_size=32,threshold=0):
                actual=layer(image)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        self.assertTrue(shapes)
        self.assertTrue(all(query<=32 and keys==17*23 for query,keys in shapes))
        self.assertNotIn('forward',layer.__dict__)
        self.assertEqual(before,v.d.digest(layer.state_dict()))
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with v.bounded_sampling_attention(generator,threshold=0):
                raise RuntimeError('injected')
        self.assertNotIn('forward',layer.__dict__)
        with self.assertRaisesRegex(RuntimeError,'inference only'):
            v.query_chunk_attention(layer,image)

    @unittest.skipUnless(os.environ.get('DCTTA_TEST_RDDM')=='1','Set DCTTA_TEST_RDDM=1 for real RDDM')
    def test_real_rddm_full_frame_does_not_change_updates(self):
        v.runtime(); torch=v.torch; d=v.d; u=v.u
        original_bounded=v.bounded_sampling_attention
        forced=patch.object(v,'bounded_sampling_attention',lambda generator: original_bounded(generator,chunk_size=7,threshold=0))
        forced.start(); self.addCleanup(forced.stop)
        from RDDM.net import ResidualDiffusionModel
        from test_migration import engine
        root=retained_directory()
        rawdir=root/'raw'; rawdir.mkdir()
        from PIL import Image
        import numpy as np
        Image.fromarray(np.random.default_rng(8).integers(0,256,(37,49,3),dtype=np.uint8)).save(rawdir/'a.png')
        samples=u.select_samples(rawdir)
        data=u.TargetRawDataset(samples,32,23); order=['a.png']
        t=engine();t.fisher_identity=t.cache_identity(data)
        t.compute_fisher(u.make_raw_loader(data,shuffle=False))
        g=ResidualDiffusionModel(SimpleNamespace(device='cpu',verbose=False,save_rddm_results=False),
            image_size=32,device='cpu',results_folder=root/'rddm')
        g.trainer.ema.step.fill_(99)
        output=v.Output(root/'output',10_000_000)
        sample=SimpleNamespace(sample_id='a.png',raw_path=rawdir/'a.png',reference_path=rawdir/'a.png')
        _,crop=data[0]
        initial=d.capture(t,g,0,order)
        loss=t.adapt(crop.unsqueeze(0),order,g)
        expected=d.digest(d.capture(t,g,1,order))
        d.restore(t,g,initial)
        with v.isolated_diagnostic(t,g,0,order):
            x,metrics,files=v.observe(t,g,sample,20.,output,'online/000001/before',v.sample_seed(23,'a.png'))
        self.assertEqual(x.shape,(1,3,37,49))
        self.assertEqual(d.digest(initial),d.digest(d.capture(t,g,0,order)))
        self.assertEqual(loss,t.adapt(crop.unsqueeze(0),order,g))
        self.assertEqual(expected,d.digest(d.capture(t,g,1,order)))
        with v.isolated_diagnostic(t,g,1,order):
            reused,after,af=v.observe(t,g,sample,20.,output,'online/000001/after',23,
                fixed_x=x,prior={r['role']:r['psnr'] for r in metrics})
        self.assertTrue(torch.equal(reused,x))
        self.assertEqual(len(files)+len(af),5)
        self.assertEqual({r['x_sd_tensor_sha256'] for r in metrics+after},{d.digest(x)})
        self.assertEqual(expected,d.digest(d.capture(t,g,1,order)))
        # 同状态、同固定种子，整图采样完全可复现。
        for trial in range(2):
            with v.isolated_diagnostic(t,g,1,order):
                again,_,_=v.observe(t,g,sample,20.,output,'snapshots/%06d'%trial,123)
            if trial==0:first=again
            else:self.assertTrue(torch.equal(first,again))
        self.assertEqual(expected,d.digest(d.capture(t,g,1,order)))
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with v.isolated_diagnostic(t,g,1,order):
                torch.rand(7)
                g.trainer.ema.step.add_(1)
                raise RuntimeError('injected failure')
        self.assertEqual(expected,d.digest(d.capture(t,g,1,order)))


if __name__=='__main__':unittest.main()
