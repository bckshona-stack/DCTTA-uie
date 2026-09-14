"""两阶段实验的数据边界、源训练和恢复测试；使用独立小型合成数据，不代表 UIEB 效果。"""
import json
import os
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from options import parse_source_args
from train_source_uie import (prepare_manifests, train_source, SourcePairs, scan_pairs,
                             checked_manifest, read_csv, write_csv, load_checkpoint)
from uie_tta import build_backbone, seed_everything


class SourceTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.root = Path(__file__).resolve().parents[2] / 'tmp' / 'source_checks' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        for domain, count in [('source', 5), ('target', 2)]:
            for part in ('raw', 'ref'):
                (self.root / domain / part).mkdir(parents=True)
            for i in range(count):
                arr = np.random.default_rng(i + (100 if domain == 'target' else 0)).integers(0, 256, (40, 48, 3), dtype=np.uint8)
                for part in ('raw', 'ref'):
                    Image.fromarray(arr).save(self.root / domain / part / f'{domain}_{i}.png')
        self.args = ['--experiment_dir', str(self.root / 'experiment'), '--source_raw', str(self.root / 'source/raw'),
                     '--source_ref', str(self.root / 'source/ref'), '--target_raw', str(self.root / 'target/raw'),
                     '--target_ref', str(self.root / 'target/ref'), '--expected_source', '5', '--expected_target', '2',
                     '--val_count', '1', '--device', 'cpu', '--num_workers', '0', '--patch_size', '32',
                     '--basic_channel', '4', '--batch_size', '2', '--epochs', '3']

    def prepare(self, extra=None):
        opt = parse_source_args(self.args + (extra or []))
        audit = prepare_manifests(opt)
        return opt, audit

    def test_paired_augmentation_and_manifest_integrity(self):
        opt, audit = self.prepare()
        folder = Path(opt.experiment_dir) / 'manifests'
        samples = checked_manifest(folder, 'source_train.csv', audit)
        self.assertEqual(len(samples), 4)
        dataset = SourcePairs(samples, 32)
        seed_everything(5)
        for _ in range(4):
            raw, ref = dataset[0]
            self.assertTrue(torch.equal(raw, ref))
            self.assertEqual(tuple(raw.shape), (3, 32, 32))
        with self.assertRaises(FileExistsError):
            prepare_manifests(opt)
        with (folder / 'source_train.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'Manifest changed'):
            checked_manifest(folder, 'source_train.csv', audit)

    def test_cross_domain_leakage_rejected(self):
        for part in ('raw', 'ref'):
            shutil.copyfile(self.root / 'source' / part / 'source_0.png', self.root / 'target' / part / 'target_0.png')
        with self.assertRaisesRegex(ValueError, 'leakage'):
            self.prepare()

    def test_conflicting_references_report_all_paths_and_pixel_difference(self):
        for index in (1, 2):
            shutil.copyfile(self.root / 'source/raw/source_0.png',
                            self.root / 'source/raw' / f'source_{index}.png')
        with self.assertRaisesRegex(ValueError, 'source_0.png'):
            self.prepare()
        folder = self.root / 'experiment/manifests'
        report = json.loads((folder / 'exact_duplicate_report.json').read_text(encoding='utf-8'))
        self.assertEqual(len(report['pairs']), 2)
        for row in report['pairs']:
            self.assertEqual(row['issue'], 'conflicting_references')
            self.assertTrue(Path(row['a_raw']).is_file())
            self.assertTrue(Path(row['b_reference']).is_file())
            self.assertGreater(row['reference_mae_0_255'], 0)
        self.assertFalse((folder / 'audit.json').exists())
        self.assertFalse((folder / 'source_train.csv').exists())

    @unittest.skipUnless((Path(__file__).resolve().parents[2] / 'tmp/uieb_conflict_check/raw/582_img_.png').exists(),
                         'Optional user-provided original UIEB conflict fixtures')
    def test_approved_real_uieb_exclusions_keep_originals_and_check_identity(self):
        fixture = Path(__file__).resolve().parents[2] / 'tmp/uieb_conflict_check'
        for name in ('582_img_.png', '648_img_.png'):
            for part in ('raw', 'ref'):
                shutil.copyfile(fixture / part / name, self.root / 'source' / part / name)
        opt, audit = self.prepare(['--expected_source', '7'])
        self.assertEqual(audit['counts'], dict(source_train=4, source_val=1, target_eval=2))
        self.assertEqual(len(audit['approved_source_exclusions']), 2)
        folder = Path(opt.experiment_dir) / 'manifests'
        ids = {r['sample_id'] for name in ('source_train.csv', 'source_val.csv') for r in read_csv(folder / name)}
        self.assertFalse(ids & {'582_img_.png', '648_img_.png'})
        self.assertEqual(len(list((self.root / 'source/raw').glob('*.png'))), 7)
        exclusion = json.loads((folder / 'source_exclusions.json').read_text(encoding='utf-8'))
        self.assertEqual(exclusion['retained_count'], 5)
        self.assertFalse(exclusion['originals_modified'])
        with Image.open(self.root / 'source/ref/582_img_.png') as image:
            changed = np.array(image.convert('RGB'), copy=True)
        changed[0, 0, 0] ^= 1
        Image.fromarray(changed).save(self.root / 'source/ref/582_img_.png')
        # 使用另一实验目录，不覆盖已经固定的清单。
        with self.assertRaisesRegex(ValueError, 'Approved exclusion content mismatch'):
            self.prepare(['--expected_source', '7', '--experiment_dir', str(self.root / 'changed')])

    def test_near_duplicates_require_review_and_keep_scene_together(self):
        base = np.asarray(Image.open(self.root / 'source/raw/source_0.png')).copy()
        base[0, 0, 0] ^= 1
        for part in ('raw', 'ref'):
            Image.fromarray(base).save(self.root / 'source' / part / 'source_1.png')
        with self.assertRaisesRegex(ValueError, 'Review candidate'):
            self.prepare()
        folder = self.root / 'experiment/manifests'
        rows = read_csv(folder / 'near_duplicate_review.csv')
        for row in rows:
            row['decision'] = 'same'
        write_csv(folder / 'near_duplicate_review.csv', rows, list(rows[0]))
        self.prepare()
        a = {r['sample_id'] for r in read_csv(folder / 'source_train.csv')}
        b = {r['sample_id'] for r in read_csv(folder / 'source_val.csv')}
        self.assertTrue({'source_0.png', 'source_1.png'} <= a or {'source_0.png', 'source_1.png'} <= b)

    def test_resume_matches_continuous_training_and_reload(self):
        opt, _ = self.prepare()
        original_experiment = Path(opt.experiment_dir)
        resumed_experiment = self.root / 'resumed'
        shutil.copytree(original_experiment / 'manifests', resumed_experiment / 'manifests')
        # 源训练若尝试打开任何目标图片，立即失败。
        original_open = Image.open
        def guarded_open(path, *args, **kwargs):
            if 'target' in str(path):
                raise AssertionError('Target image read during source training')
            return original_open(path, *args, **kwargs)
        with patch('PIL.Image.open', guarded_open):
            full = train_source(opt)
            interrupted = parse_source_args(self.args + ['--experiment_dir', str(resumed_experiment), '--stop_after_epoch', '1'])
            train_source(interrupted)
            resume_path = resumed_experiment / 'source/checkpoints/last.pth'
            resumed = parse_source_args(self.args + ['--experiment_dir', str(resumed_experiment), '--resume', str(resume_path)])
            history = train_source(resumed)
        left = load_checkpoint(original_experiment / 'source/checkpoints/last.pth')
        right = load_checkpoint(resume_path)
        self.assertEqual(left['scheduler'], right['scheduler'])
        self.assertTrue(all(torch.equal(left['net'][k], right['net'][k]) for k in left['net']))
        for a, b in zip(full, history):
            for key in ('train_l1', 'val_l1', 'val_psnr', 'val_ssim', 'lr'):
                self.assertEqual(a[key], b[key])
        self.assertEqual(right['global_step'], 6)
        self.assertAlmostEqual(right['history'][-1]['lr'], 0)
        model = build_backbone('nu2net', resume_path, torch.device('cpu'))
        self.assertEqual(model(torch.rand(1, 3, 40, 48)).shape, (1, 3, 40, 48))
        for name in ('l1', 'psnr', 'ssim', 'learning_rate'):
            with Image.open(resumed_experiment / 'source/curves' / (name + '.png')) as img:
                self.assertGreater(img.width, 500)
        altered = parse_source_args(self.args + ['--resume', str(original_experiment / 'source/checkpoints/last.pth'), '--lr', '0.002'])
        with self.assertRaisesRegex(ValueError, 'Resume configuration'):
            train_source(altered)

    def test_checkpoint_commit_interruption_can_recover_best(self):
        opt, _ = self.prepare()
        import train_source_uie as source
        original_save = source.save_checkpoint
        def interrupted_save(path, payload):
            if Path(path).name == 'best.pth':
                raise RuntimeError('simulated save interruption')
            original_save(path, payload)
        opt.stop_after_epoch = 1
        with patch('train_source_uie.save_checkpoint', interrupted_save):
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                train_source(opt)
        opt.resume = str(Path(opt.experiment_dir) / 'source/checkpoints/last.pth')
        history = train_source(opt)
        self.assertEqual(len(history), 1)
        self.assertTrue((Path(opt.experiment_dir) / 'source/checkpoints/best.pth').is_file())

    @unittest.skipUnless(os.environ.get('DCTTA_TEST_RDDM') == '1', 'Opt-in real RDDM experiment integration')
    def test_three_way_experiment_from_trained_source(self):
        from adapt_target_uie import main as target_main, summarize
        from test_migration import TinyFeatures
        opt, _ = self.prepare(['--epochs', '1'])
        train_source(opt)
        with patch('utils.loss_utils.PerceptualLoss', TinyFeatures):
            means = target_main(['--experiment_dir', opt.experiment_dir, '--device', 'cpu',
                                 '--run_all', '--patch_size', '32'])
        folder = Path(opt.experiment_dir)
        self.assertIn('dctta_minus_source_psnr', means)
        self.assertEqual(len(read_csv(folder / 'comparison/per_image.csv')), 2)
        report = json.loads((folder / 'dctta/run.json').read_text(encoding='utf-8'))
        self.assertEqual(report['adaptation_steps'], 2)
        with self.assertRaises(FileExistsError):
            target_main(['--experiment_dir', opt.experiment_dir, '--device', 'cpu', '--run_all', '--patch_size', '32'])
        identity_path = folder / 'dctta/experiment_identity.json'
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
        identity['target_manifest_sha256'] = 'mismatch'
        identity_path.write_text(json.dumps(identity), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            summarize(folder)


if __name__ == '__main__':
    unittest.main()
