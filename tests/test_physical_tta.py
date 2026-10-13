"""正式物理TTA：原损失梯度、固定教师、Fisher、观测隔离和逐步续跑。"""
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import run_physical_uie as r
from test_migration import tiny_model, options, TinyFeatures, LowFrequency


class FormalPhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r.pilot.configure_runtime('cpu'); r.v.runtime(); torch.set_num_threads(1)

    def fixture(self):
        root = Path(__file__).resolve().parents[2]/'tmp'/('physical_formal_'+uuid.uuid4().hex)
        root.mkdir(parents=True)
        r.u.seed_everything(23)
        samples = []
        for i in range(3):
            path = root/('%d.png'%i)
            path.write_bytes(r.pilot.image_bytes(torch.rand(1, 3, 40+i*8, 48+i*8)))
            samples.append(r.u.Sample(path.name, path, path))
        data = r.u.TargetRawDataset([r.u.Sample(x.sample_id, x.raw_path, None) for x in samples], 32, 23)
        order = [x.sample_id for x in samples]
        opt = options(compute_fisher=1, lr=2e-4, betas=(.9, .999)); model = tiny_model(); r.u.configure_model(opt, model)
        t = r.FixedPhysicalTTA(opt, model, torch.optim.Adam(r.u.collect_params(model)[0], lr=opt.lr, betas=opt.betas),
                              perceptual_loss=TinyFeatures(), wavelet=LowFrequency())
        t.fisher_identity = t.cache_identity(data); t.compute_fisher(r.u.make_raw_loader(data, shuffle=False))
        t.fisher_origin = dict(policy='fixture')
        initial = r.capture(t, 0, order)
        protocol = dict(order=order, probes={'target_eval':order})
        chosen = [dict(sample_id=x.sample_id, height=40+i*8, width=48+i*8, source_psnr=20.) for i,x in enumerate(samples)]
        groups = dict(target_eval=samples, source_val=[])
        out = r.s.BudgetOutput(root/'output', 100_000_000); out.json('plan.json', dict(fixture=True))
        args = SimpleNamespace(factorial_dir=str(root/'factorial'))
        def build(*a):
            r.restore(t, initial)
            return t, data, initial
        prepared = out, protocol, groups, chosen, [0,1,2,3]
        return root, args, prepared, build, t, data, initial

    def test_common_fit_and_actual_gradient_path(self):
        _, _, _, _, t, data, initial = self.fixture(); raw = data[0][1].unsqueeze(0)
        gen = r.PhysicalGenerator(.5); seen = {}; predictions = []
        original = t.compute_loss2
        def loss(pred, target, *a):
            predictions.append(pred)
            return original(pred, target, *a)
        t.compute_loss2 = loss
        def observer(event, values):
            if event == 'student_outputs': seen.update(values)
        t.observer = observer; teacher_hash = r.d.digest(t.model_teacher.state_dict())
        result = t.adapt(raw, [data.samples[0].sample_id], gen)
        self.assertEqual(len(predictions), 2); self.assertIs(predictions[0], predictions[1])
        self.assertTrue(predictions[0].requires_grad); self.assertFalse(seen['raw_output'].requires_grad)
        self.assertFalse(seen['regenerated'].requires_grad)
        self.assertEqual(gen.steps, 200); self.assertEqual(len(gen.trace), 201)
        self.assertEqual(teacher_hash, r.d.digest(t.model_teacher.state_dict()))
        self.assertTrue(all(x.grad is None for x in t.model_teacher.parameters()))
        for n, parameter in t.candidate_parameters().items():
            mask = t.fishers[n][1].bool()
            self.assertTrue(torch.equal(parameter.detach()[mask], t.model_state[n][mask]))
        self.assertAlmostEqual(result['loss'], result['loss_s']+result['loss_t'], places=5)
        full = r.PhysicalGenerator(1.); full.train(raw, gen.pseudo.to(raw.device), ['same'])
        self.assertEqual(gen.payload['parameter_sha256'], full.payload['parameter_sha256'])
        self.assertTrue(torch.allclose(gen.x_sd, .5*(raw+full.x_sd), atol=1e-7))
        # 200步拟合后的参数用于raw，不用物理重建I_hat替代x_sd。
        model = r.load_fit(gen.payload, 'cpu')
        self.assertTrue(torch.equal(gen.x_sd, model.render(raw, .5).detach()))
        self.assertFalse(torch.allclose(gen.x_sd, model.objective(raw, gen.pseudo)[3]))

    def test_gamma_zero_matches_unmodified_fixed_teacher_raw_path(self):
        _, _, prepared, _, t, data, initial = self.fixture(); order = prepared[1]['order']
        a, _ = r.adapt_one(t, data, order, 0, 0., measure=False)
        expected = r.capture(t, 1, order)
        r.restore(t, initial)
        class Identity:
            def train(self, raw, pseudo, names): return raw.detach()
        b = t.adapt(data[0][1].unsqueeze(0), [order[0]], Identity())
        self.assertEqual(a['losses'], b)
        self.assertEqual(r.d.digest(expected), r.d.digest(r.capture(t, 1, order)))

    def test_observation_cache_and_restoration_exact(self):
        _, _, prepared, _, t, data, initial = self.fixture()
        out, protocol, groups, chosen, stages = prepared
        r.adapt_one(t, data, protocol['order'], 0, .5)
        before = r.d.digest(r.capture(t, 1, protocol['order']))
        files, metrics, _ = r.observe(t, groups, chosen, 1, protocol['order'], out, 'G6', protocol, stages)
        self.assertEqual(before, r.d.digest(r.capture(t, 1, protocol['order'])))
        self.assertEqual({x['role'] for x in metrics}, {'teacher_pseudo','x_sd','y_sd','y_in'})
        self.assertEqual(len(metrics), 12)
        sample = groups['target_eval'][0]
        with patch.object(r.p, 'fit_steps', side_effect=AssertionError('cached fit must skip')):
            r.full_cached(t, sample, out)
        packet = next((out.root/'full_cache').glob('*.pth'))
        saved = r.u.load_checkpoint(packet); saved['parameters']['a'].add_(1); torch.save(saved, packet)
        with self.assertRaisesRegex(ValueError, 'parameter identity'):
            r.full_cached(t, sample, out)

    def test_resume_and_observed_run_match_continuous(self):
        _, args, prepared, build, t, data, initial = self.fixture()
        out, protocol, groups, chosen, stages = prepared
        out.json('acceptance.json', dict(passed=True, plan_sha256=r.v.sha(out.root/'plan.json'), files={}))
        expected = {}
        for group, gamma in r.GROUPS.items():
            r.restore(t, initial)
            for i in range(3): r.adapt_one(t, data, protocol['order'], i, gamma, measure=False)
            expected[group] = r.d.digest(r.capture(t, 3, protocol['order']))
        with patch.object(r, 'preflight', return_value=prepared), patch.object(r, 'build', side_effect=build):
            for group in r.GROUPS:
                original = r.s.BudgetOutput.json
                def interrupt(self, name, value):
                    result = original(self, name, value)
                    if name == group+'/progress.json' and value['committed_step'] == 1:
                        raise RuntimeError('committed interruption')
                    return result
                with patch.object(r.s.BudgetOutput, 'json', interrupt):
                    with self.assertRaisesRegex(RuntimeError, 'committed interruption'): r.run_group(args, group)
                r.run_group(args, group)
                state = r.u.load_checkpoint(out.root/group/'resume.pth')['state']
                self.assertEqual(expected[group], r.d.digest(state))
                status = r.v.read(out.root/group/'status.json')
                self.assertEqual(status['images'], 60)
                self.assertEqual(len(list((out.root/group).rglob('*.png'))), 60)
                with patch.object(r, 'build', side_effect=AssertionError('completed skip')): r.run_group(args, group)
            a = r.v.read(out.root/'G6/trajectory.json'); b = r.v.read(out.root/'G7/trajectory.json')
            self.assertEqual([x['inputs']['parameter_sha256'] for x in a], [x['inputs']['parameter_sha256'] for x in b])
            artifact = out.root/'G6/training/000001/x_sd.png'; artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Committed artifact'): r.run_group(args, 'G6')

    def test_device_acceptance_workflow_with_real_tiny_student(self):
        root, args, prepared, build, t, data, initial = self.fixture()
        row, _ = r.adapt_one(t, data, prepared[1]['order'], 0, 0., measure=False)
        r.f.write(root/'factorial/G3/trajectory.json', [dict(losses=row['losses'])])
        r.f.save_checkpoint(root/'factorial/G3/snapshots/000001.pth', dict(student=r.d.cpu_tree(t.model.state_dict())))
        with patch.object(r, 'preflight', return_value=prepared), patch.object(r, 'build', side_effect=build):
            r.acceptance(args)
            report = r.v.read(prepared[0].root/'acceptance.json')
            self.assertTrue(report['passed']); self.assertTrue(report['checks']['G3_gamma_zero'])
            with patch.object(r, 'build', side_effect=AssertionError('accepted skip')): r.acceptance(args)

    def test_build_restores_real_student_without_constructing_rddm(self):
        root, args, prepared, _, t, data, initial = self.fixture()
        args.experiment_dir = root/'exp'; args.diagnostic_dir = root/'diag'; args.device='cpu'
        initial['fisher_origin'] = dict(policy='fixed_first_round')
        cfg = dict(basic_channel=4, is_residual=True, tail='norm')
        # 存真实NU2Net源权重；build用原restore路径恢复学生、Adam和Fisher。
        state = {n.removeprefix('backbone.'):x for n,x in initial['student']['weights'].items()}
        r.f.save_checkpoint(root/'exp/source/checkpoints/best.pth', dict(net=state, model_config=cfg))
        r.f.save_checkpoint(root/'diag/run/replay_states/000000.pth', initial)
        r.f.save_checkpoint(root/'exp/dctta/fisher.pt', dict(fishers=initial['fisher']))
        protocol = dict(prepared[1], config=dict(vars(t.opt), patch_size=32), target=data.identity())
        with patch.object(r.d, 'inputs', return_value=(prepared[2], {}, None)), patch.object(r.d, 'build', side_effect=AssertionError('must not build RDDM')):
            restored, actual_data, actual_initial = r.build(args, protocol, TinyFeatures(), LowFrequency())
        self.assertEqual(r.d.digest(initial), r.d.digest(actual_initial))
        self.assertEqual(actual_data.identity(), data.identity())
        self.assertEqual(r.d.digest(restored.model_teacher.state_dict()), r.d.digest(restored.model_state))

    def test_output_safety_rejects_before_reading_old_inputs(self):
        root = Path(__file__).resolve().parents[2]/'tmp'/('physical_safety_'+uuid.uuid4().hex)
        root.mkdir(); output = root/'out'; output.mkdir(); (output/'keep.txt').write_text('keep')
        args = SimpleNamespace(device='cpu',output_dir=output,
            **{name:root/name for name in ['experiment_dir','diagnostic_dir','factorial_dir','strength_dir','visual_dir','pilot_dir']})
        with self.assertRaisesRegex(ValueError, 'initially be empty'): r.preflight(args)
        self.assertEqual((output/'keep.txt').read_text(), 'keep')
        args.output_dir = args.experiment_dir
        with self.assertRaisesRegex(ValueError, 'separate sibling'): r.preflight(args)

    def test_comparison_and_cross_group_fit_identity(self):
        root, args, prepared, _, _, _, _ = self.fixture()
        out, protocol, groups, chosen, stages = prepared
        args.diagnostic_dir = root/'diag'; args.strength_dir = root/'strength'
        def records(delta): return [dict(sample_id=n, psnr=20.+delta, ssim=.8) for n in protocol['order']]+[dict(sample_id='source.png', psnr=22.,ssim=.9)]
        roots = {'G0':root/'diag/evaluation'}
        roots.update({g:root/'factorial'/g/'evaluation' for g in ('G1','G2','G3')})
        roots.update({g:root/'strength'/g/'evaluation' for g in ('G4','G5')})
        roots.update({g:out.root/g/'evaluation' for g in r.GROUPS})
        for i,(g, folder) in enumerate(roots.items()):
            r.f.write(folder/'000003_student.json', records(i-3))
            if g == 'G0': r.f.write(folder/'000000_student.json', records(0))
            if g in r.GROUPS:
                out.json(g+'/trajectory.json', [dict(sample_id=n, inputs=dict(parameter_sha256=n, initial_sha256='same')) for n in protocol['order']])
                out.json(g+'/status.json', dict(status='complete', plan_sha256=r.v.sha(out.root/'plan.json'),files={}))
        with patch.object(r, 'preflight',return_value=prepared):
            r.compare(args)
            self.assertEqual(set(r.v.read(out.root/'comparison/summary.json')), set(roots))
            self.assertEqual(r.v.read(out.root/'comparison/summary.json')['G7']['target']['mean_delta'], 4.)
            value = r.v.read(out.root/'G7/trajectory.json'); value[0]['inputs']['parameter_sha256']='changed'
            out.json('G7/trajectory.json', value)
            with self.assertRaisesRegex(ValueError, 'common physical fit'): r.compare(args)


if __name__ == '__main__': unittest.main()
