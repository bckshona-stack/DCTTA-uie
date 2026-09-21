"""真实NU2Net小图验证：历史权重推理、指标一致性、断点续跑和篡改拒绝。"""
import copy
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import export_g0_yin as e


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e.runtime()
        e.torch.set_num_threads(1)

    def fixture(self):
        root = Path(__file__).resolve().parents[2]/'tmp'/('yin_test_'+uuid.uuid4().hex)
        root.mkdir(parents=True)
        diag, visual = root/'diag', root/'visual'
        (diag/'run/snapshots').mkdir(parents=True)
        (diag/'evaluation').mkdir()
        visual.mkdir()
        from uie_backbones.nu2net import NU2Net
        config = dict(basic_channel=4, is_residual=True, tail='norm')
        source = root/'source.pth'
        e.torch.save(dict(net=NU2Net(**config).state_dict(), model_config=config), source)
        model = e.u.build_backbone('nu2net', source, 'cpu')
        samples = {}
        chosen = []
        for i in range(2):
            name = '%d.png' % i
            raw = e.torch.rand(1, 3, 25+i, 33+i)
            out = e.v.Output(root/'data', 100_000_000)
            e.v.export_image(out, name, raw)
            e.v.export_image(out, 'ref_'+name, raw*.7+.1)
            samples[name] = SimpleNamespace(sample_id=name, raw_path=out.root/name, reference_path=out.root/('ref_'+name))
            chosen.append(dict(sample_id=name, source_psnr=20.))
        historical = []
        weights = {}
        for step in range(3):
            student = copy.deepcopy(model.state_dict())
            with e.torch.no_grad():
                next(model.parameters()).add_(.001)
            teacher = copy.deepcopy(model.state_dict())
            path = diag/'run/snapshots'/('%06d.pth' % step)
            e.torch.save(dict(student=student, teacher=teacher), path)
            weights[str(step)] = e.v.sha(path)
            with e.torch.no_grad():
                for sample in samples.values():
                    raw = e.u.read_rgb(sample.raw_path).unsqueeze(0)
                    ref = e.u.read_rgb(sample.reference_path).unsqueeze(0)
                    pred = e.u.eight_view_mean(model, raw)
                    if step in (0, 2):
                        historical.append(dict(step=step, sample_id=sample.sample_id,
                            kind='snapshot', role='teacher_pseudo', psnr=e.v.psnr(pred, ref)))
                model.load_state_dict(student)
                e.v.Output(diag, 100_000_000).json('evaluation/%06d_student.json' % step,
                    e.d.score(model, list(samples.values())))
                model.load_state_dict(teacher)
        (visual/'metrics.csv').write_bytes(e.v.csv_bytes(historical))
        contract = dict(selected=chosen, stages=[0, 2], steps=3, snapshots=weights, expected_images=4)
        args = SimpleNamespace(diagnostic_dir=diag, visual_dir=visual, device='cpu')
        return root, args, contract, samples, source

    def test_real_model_resume_integrity(self):
        root, args, contract, samples, source = self.fixture()
        out = e.v.Output(root/'output', 100_000_000)
        out.json('manifest.json', contract)
        prepare_result = (out, contract, samples, source)
        original = e.measure
        def interrupt(*a, **kw):
            if a[4] == 1:
                raise RuntimeError('simulated interruption')
            return original(*a, **kw)
        with patch.object(e, 'prepare', return_value=prepare_result), patch.object(e, 'measure', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                e.execute(args)
        self.assertEqual(e.v.read(out.root/'progress.json')['next_step'], 1)
        with patch.object(e, 'prepare', return_value=prepare_result):
            e.execute(args)
            with patch.object(e, 'measure', side_effect=AssertionError('must skip')):
                e.execute(args)
        rows = e.csv_read(out.root/'metrics.csv')
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(list((out.root/'student_raw').rglob('*.png'))), 4)
        model = e.u.build_backbone('nu2net', source, 'cpu')
        for row in rows:
            state = e.u.load_checkpoint(args.diagnostic_dir/'run/snapshots'/('%06d.pth' % int(row['step'])))
            sample = samples[row['sample_id']]
            raw, ref = [e.u.read_rgb(p).unsqueeze(0) for p in (sample.raw_path, sample.reference_path)]
            with e.torch.no_grad():
                model.load_state_dict(state['teacher'])
                self.assertAlmostEqual(float(row['teacher_raw_psnr']), e.v.psnr(model(raw), ref), places=6)
            if row['path']:
                png = e.u.read_rgb(out.root/row['path']).unsqueeze(0)
                self.assertAlmostEqual(float(row['student_raw_png_psnr']), e.v.psnr(png, ref), places=6)
        second = e.v.Output(root/'uninterrupted', 100_000_000)
        second.json('manifest.json', contract)
        with patch.object(e, 'prepare', return_value=(second, contract, samples, source)):
            e.execute(args)
        self.assertEqual((out.root/'metrics.csv').read_bytes(), (second.root/'metrics.csv').read_bytes())
        self.assertEqual(e.v.read(out.root/'status.json')['files'], e.v.read(second.root/'status.json')['files'])
        (out.root/rows[0]['path']).write_bytes(b'corrupted')
        with patch.object(e, 'prepare', return_value=prepare_result):
            with self.assertRaisesRegex(ValueError, 'Committed output changed'):
                e.execute(args)

    def test_historical_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            e.verify_reference(22., 21., 'test')
        with self.assertRaises(ValueError):
            e.verify_reference(float('nan'), 21., 'test')
        e.verify_reference(21.+1e-7, 21., 'test')


if __name__ == '__main__':
    unittest.main()
