"""物理参数拟合：数学边界、单图重置、梯度隔离、真实教师与续跑。"""
import csv
import io
import json
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import physical_degradation as p
import pilot_physical_uie as pilot


class PhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pilot.configure_runtime('cpu')
        torch.set_num_threads(1)

    def test_bilinear_equivalence_and_derivatives(self):
        for h, w, grid in [(17, 23, 4), (13, 9, 16), (31, 32, 7)]:
            model = p.PhysicalFit(p.FitConfig(grid_size=grid), h, w)
            x = torch.rand(1, 3, grid, grid, requires_grad=True)
            ours = (model.height_weights @ x) @ model.width_weights.T
            expected = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            self.assertTrue(torch.allclose(ours, expected, atol=1e-6, rtol=1e-6))
            weights = torch.rand_like(ours)
            a = torch.autograd.grad((ours*weights).sum(), x, retain_graph=True)[0]
            b = torch.autograd.grad((expected*weights).sum(), x)[0]
            self.assertTrue(torch.allclose(a, b, atol=2e-5, rtol=2e-6))

    def test_bounds_strength_and_finite_gradient(self):
        model = p.PhysicalFit(p.FitConfig(grid_size=4), 24, 32)
        raw = torch.rand(1, 3, 24, 32)
        self.assertTrue(torch.equal(model.render(raw, 0), raw))
        initial = model.parameter_hash()
        full = model.render(raw, 1)
        half = model.render(raw, .5)
        self.assertTrue(torch.allclose(half, .5*(raw+full), atol=1e-7))
        t, b, _ = model.fields()
        self.assertTrue(bool((t >= .1-1e-7).all() and (t <= 1).all()))
        self.assertTrue(bool((b >= 0).all() and (b <= 1).all()))
        self.assertTrue(bool((full >= 0).all() and (full <= 1).all()))
        model.objective(raw, raw*.8+.05)[0].backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in model.parameters()))
        self.assertEqual(initial, model.parameter_hash())
        for gamma in (-1, 1.1, float('nan')):
            with self.assertRaises(ValueError): model.render(raw, gamma)

    def test_synthetic_fit_reduces_error_without_input_or_rng_gradients(self):
        torch.manual_seed(123)
        clean = torch.rand(1, 3, 24, 32, requires_grad=True)
        background = torch.tensor([.15, .4, .7]).reshape(1, 3, 1, 1)
        raw = (clean.detach()*.7+background*.3).requires_grad_()
        rng = torch.get_rng_state().clone()
        for grid in (4, 32):
            result = list(p.fit_steps(raw, clean, p.FitConfig(grid_size=grid), 100))
            self.assertLess(result[-1][1]['reconstruction_l1'], result[0][1]['reconstruction_l1']*.2)
        self.assertIsNone(raw.grad); self.assertIsNone(clean.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_per_image_reset_and_order_independence(self):
        raw, pseudo = torch.rand(1, 3, 20, 24), torch.rand(1, 3, 20, 24)
        config = p.FitConfig(grid_size=4)
        first = list(p.fit_steps(raw, pseudo, config, 8))[-1][0]
        list(p.fit_steps(raw*.5, pseudo*.5, config, 12))
        second = list(p.fit_steps(raw, pseudo, config, 8))[-1][0]
        self.assertEqual(first.parameter_hash(), second.parameter_hash())
        self.assertEqual(first.initial_hash, p.PhysicalFit(config, 40, 60).initial_hash)
        self.assertEqual(first.initial_hash, second.initial_hash)

    def test_invalid_configuration_and_images(self):
        for kwargs in [dict(t_min=0), dict(t_min=.95), dict(grid_size=1), dict(lr=float('nan')),
                       dict(tv_weight=-1), dict(seed=-1)]:
            with self.assertRaises(ValueError): p.FitConfig(**kwargs)
        raw = torch.rand(1, 3, 12, 16)
        for value in (raw.double(), raw+1, raw*float('nan'), raw.expand(2, -1, -1, -1)):
            with self.assertRaises(ValueError): list(p.fit_steps(value, value, p.FitConfig(), 1))

    def fixture(self):
        root = Path(__file__).resolve().parents[2]/'tmp'/('physical_test_'+uuid.uuid4().hex)
        root.mkdir(parents=True)
        from uie_backbones.nu2net import NU2Net
        cfg = dict(basic_channel=4, is_residual=True, tail='norm')
        source = root/'source.pth'
        torch.save(dict(net=NU2Net(**cfg).state_dict(), model_config=cfg), source)
        samples = []
        for i in range(2):
            name = '%d.png'%i
            value = torch.rand(1, 3, 40+i*8, 48+i*8)
            path = root/name
            path.write_bytes(pilot.image_bytes(value))
            # reference不存在，成功运行证明预实验不需要它。
            samples.append(pilot.u.Sample(name, path, root/'NONEXISTENT_REFERENCE.png'))
        data = pilot.u.TargetRawDataset(samples, 32, 23)
        contract = dict(fit=p.FitConfig(grid_size=4).identity(), points=[0, 2, 4], scopes=['patch', 'full'],
                        samples=data.identity(), expected_images=2*2*(2+6*3))
        return root, source, data, contract

    def test_real_teacher_resume_matches_continuous_and_tamper_rejected(self):
        root, source, data, contract = self.fixture()
        out = pilot.v.Output(root/'resumed', 100_000_000)
        out.json('manifest.json', contract)
        args = SimpleNamespace()
        prepared = (out, contract, data, source, torch.device('cpu'))
        actual = pilot.fit_unit
        calls = []
        def interrupted(*a, **kw):
            calls.append(1)
            if len(calls) == 2: raise RuntimeError('interrupted')
            return actual(*a, **kw)
        with patch.object(pilot, 'prepare', return_value=prepared), patch.object(pilot, 'fit_unit', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'): pilot.run(args)
        self.assertEqual(pilot.v.read(out.root/'progress.json')['cursor'], 1)
        with patch.object(pilot, 'prepare', return_value=prepared):
            pilot.run(args)
            with patch.object(pilot, 'fit_unit', side_effect=AssertionError('must skip')): pilot.run(args)
        second = pilot.v.Output(root/'continuous', 100_000_000)
        second.json('manifest.json', contract)
        with patch.object(pilot, 'prepare', return_value=(second, contract, data, source, torch.device('cpu'))):
            pilot.run(args)
        self.assertEqual((out.root/'metrics.csv').read_bytes(), (second.root/'metrics.csv').read_bytes())
        status = pilot.v.read(out.root/'status.json')
        self.assertEqual(status['images'], 80)
        self.assertEqual(status['metric_rows'], 12)
        self.assertEqual(sum(x.is_file() for x in (out.root/'images').rglob('*.png')), 80)
        for name, digest in status['files'].items():
            if not name.endswith('timing.json') and name != 'progress.json':
                self.assertEqual(digest, pilot.v.sha(second.root/name))
        artifact = next((out.root/'images').rglob('parameters.pt'))
        artifact.write_bytes(b'corrupted')
        with patch.object(pilot, 'prepare', return_value=prepared):
            with self.assertRaisesRegex(ValueError, 'Committed output changed'): pilot.run(args)

    def test_complete_preflight_without_reference(self):
        root, source, _, _ = self.fixture()
        exp, diag, visual = root/'exp', root/'diag', root/'visual'
        (exp/'source/checkpoints').mkdir(parents=True)
        (exp/'source/checkpoints/best.pth').write_bytes(source.read_bytes())
        (exp/'manifests').mkdir(); diag.mkdir(); visual.mkdir()
        selected, raw_rows, samples = [], [], []
        for i in range(10):
            name = 'selected_%d.png'%i
            path = root/name
            path.write_bytes(pilot.image_bytes(torch.rand(1, 3, 40, 48)))
            selected.append(dict(sample_id=name, reason='fixed'))
            raw_rows.append(dict(sample_id=name, raw_path=str(path), raw_sha256=pilot.v.sha(path),
                                 reference_path='NONEXISTENT_REFERENCE.png'))
            samples.append(pilot.u.Sample(name, path, None))
        target_csv = exp/'manifests/target_eval.csv'
        target_csv.write_bytes(pilot.v.csv_bytes(raw_rows))
        data = pilot.u.TargetRawDataset(samples, 32, 23)
        (exp/'manifests/audit.json').write_text(json.dumps(dict(files={'target_eval.csv':pilot.v.sha(target_csv)})))
        selected_csv = visual/'selected_samples.csv'
        selected_csv.write_bytes(pilot.v.csv_bytes(selected))
        (visual/'run_manifest.json').write_text(json.dumps(dict(selected=selected,
            selection_csv_sha256=pilot.v.sha(selected_csv))))
        (visual/'status.json').write_text(json.dumps(dict(status='complete')))
        implementations = {name:pilot.v.sha(Path(pilot.__file__).parent/name) for name in
                           ('uie_tta.py', 'options.py', 'uie_backbones/nu2net.py')}
        protocol = dict(diagnostic_protocol='fixed_fisher_strict',
            source_sha256=pilot.v.sha(exp/'source/checkpoints/best.pth'),
            config=dict(patch_size=32, seed=23), target=data.identity(), order=[s.sample_id for s in samples],
            environment=dict(implementation=implementations))
        (diag/'protocol.json').write_text(json.dumps(protocol))
        args = SimpleNamespace(experiment_dir=exp, diagnostic_dir=diag, visual_dir=visual,
            output_dir=root/'output', raw_dir=None, device='cpu', grid_size=4, t_min=.1,
            fit_lr=.05, tv_weight=.01, fit_seed=23, steps='0,1', scopes='patch,full', max_gb=1)
        prepared = pilot.prepare(args)
        self.assertEqual(prepared[1]['expected_images'], 280)
        self.assertTrue(all(s.reference_path is None for s in prepared[2].samples))
        args.fit_lr = .06
        with self.assertRaisesRegex(ValueError, 'contract changed'): pilot.prepare(args)


if __name__ == '__main__': unittest.main()
