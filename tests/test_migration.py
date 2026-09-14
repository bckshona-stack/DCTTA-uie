"""Run with python -m unittest discover -s tests -v from movetoUIE."""

import ast
from copy import deepcopy
import hashlib
import importlib
import json
import logging
from pathlib import Path
import random
import sys
from contextlib import contextmanager
import uuid
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import RandomHorizontalFlip, RandomRotation

from uie_tta import build_backbone, configure_model, collect_params
from options import parse_args
from uie_tta import SRTTA
from uie_tta import (Sample, select_samples, TargetRawDataset, EvaluationDataset,
                        make_raw_loader, crop16)
from uie_tta import isolated_rng, seed_everything
from uie_tta import eight_view_mean, transform_tensor, TEACHER_OPS, FISHER_OPS
from uie_backbones.nu2net import NU2Net, normalize_img
from uie_tta import ImageRestorationAdapter


ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)


@contextmanager
def test_directory():
    # Retain test artifacts for inspection; never recursively delete a directory.
    path = ROOT / "tmp" / "uie_checks" / uuid.uuid4().hex
    path.mkdir(parents=True)
    yield str(path)


def options(**kwargs):
    values = dict(backbone="nu2net", teacher_weight=5.0, ema_decay=0.45,
                  fisher_ratio=0.6, compute_fisher=1, iterations=1, seed=23, verbose=False)
    return SimpleNamespace(**(values | kwargs))


class TinyFeatures(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(F.avg_pool2d(pred, 2), F.avg_pool2d(target.detach(), 2))


class LowFrequency(nn.Module):
    def forward(self, image):
        return F.avg_pool2d(image.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).unsqueeze(1)


class DeterministicGenerator:
    def __init__(self):
        self.calls = []

    def train(self, image, pseudo, names):
        self.calls.append((image.detach().clone(), pseudo.detach().clone(), tuple(names)))
        assert not pseudo.requires_grad
        return (0.8 * image + 0.2 * pseudo).detach()


def tiny_model():
    return ImageRestorationAdapter(NU2Net(basic_channel=4), "nu2net",
                                   {"basic_channel": 4, "is_residual": True, "tail": "norm"})


def engine(model=None, **kwargs):
    opt = options(**kwargs)
    model = model if model is not None else tiny_model()
    configure_model(opt, model)
    optimizer = torch.optim.Adam(collect_params(model)[0], lr=2e-4, betas=(0.9, 0.999))
    return SRTTA(opt, model, optimizer, perceptual_loss=TinyFeatures(), wavelet=LowFrequency())


def root_tta_class():
    tree = ast.parse((ROOT / "tta.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SRTTA"]
    # CPU device adaptation only; the root file is never modified.
    class CpuCalls(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
                node.func.attr = "to"
                node.args = [ast.Constant("cpu")]
            return node
    module = ast.fix_missing_locations(CpuCalls().visit(ast.Module(body=nodes, type_ignores=[])))
    scope = {"torch": torch, "optim": torch.optim, "nn": nn,
             "RandomHorizontalFlip": RandomHorizontalFlip, "RandomRotation": RandomRotation,
             "PerceptualLoss": TinyFeatures, "logger": logging.getLogger("reference"),
             "tqdm": lambda data, **kw: data,
             "utils_tta": SimpleNamespace(transform_tensor=transform_tensor,
                 augment_transform=lambda x: ([transform_tensor(x, op) for op in FISHER_OPS], list(FISHER_OPS)))}
    exec(compile(module, "root_tta_reference", "exec"), scope)
    return scope["SRTTA"]


class MigrationTests(unittest.TestCase):
    def setUp(self):
        seed_everything(17)

    def test_nu2net_source_is_unchanged(self):
        record = json.loads((ROOT / "movetoUIE/third_party/UnderwaterRanker/SOURCE.json").read_text())
        source = ROOT / "movetoUIE/uie_backbones/nu2net.py"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), record["nu2net_sha256"])

    def test_original_tail_exhibits_batch_coupling(self):
        a = torch.linspace(0.2, 0.8, 48).reshape(1, 3, 4, 4)
        b = a + 2
        self.assertFalse(torch.allclose(normalize_img(a), normalize_img(torch.cat([a, b]))[:1]))

    def test_adapter_shape_and_gradient(self):
        model = tiny_model()
        for height, width in ((32, 32), (33, 47)):
            image = torch.rand(1, 3, height, width, requires_grad=True)
            result = model(image)
            self.assertEqual(result.shape, image.shape)
            self.assertTrue(torch.isfinite(result).all())
            result.square().mean().backward()
            self.assertTrue(torch.isfinite(image.grad).all())
            self.assertGreater(image.grad.abs().sum().item(), 0)

    def test_teacher_transforms_match_root_rotation_order(self):
        image = torch.rand(1, 3, 32, 32)
        flip = RandomHorizontalFlip(p=1)
        old = [RandomRotation((angle, angle))(flip(image)) for angle in (0, 90, 180, 270)]
        old += [RandomRotation((angle, angle))(image) for angle in (0, 90, 180, 270)]
        for op, expected in zip(TEACHER_OPS, old):
            torch.testing.assert_close(transform_tensor(image, op), expected, rtol=0, atol=0)
        rectangular = torch.rand(1, 3, 32, 48)
        for op in TEACHER_OPS:
            torch.testing.assert_close(transform_tensor(transform_tensor(rectangular, op), op, True), rectangular)

    def test_teacher_always_uses_single_view(self):
        seen = []
        model = tiny_model().eval()
        hook = model.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[0]))
        image = torch.rand(1, 3, 32, 48)
        result = eight_view_mean(model, image)
        hook.remove()
        self.assertEqual(seen, [1] * 8)
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.shape, image.shape)
        with self.assertRaises(ValueError):
            eight_view_mean(model, image.expand(2, -1, -1, -1))

    def test_strict_checkpoint_and_no_shared_storage(self):
        model = tiny_model()
        with test_directory() as directory:
            file = Path(directory) / "source.pth"
            torch.save({"net": model.backbone.state_dict(), "model_config": model.model_config}, file)
            loaded = build_backbone("nu2net", file, torch.device("cpu"))
            image = torch.rand(1, 3, 32, 32)
            torch.testing.assert_close(model(image), loaded(image))
            tta = engine(loaded)
            for student, teacher in zip(tta.model.parameters(), tta.model_teacher.parameters()):
                self.assertNotEqual(student.data_ptr(), teacher.data_ptr())
                self.assertFalse(teacher.requires_grad)
                torch.testing.assert_close(student, teacher)
            torch.save({"net": {}, "model_config": model.model_config}, file)
            with self.assertRaises(RuntimeError):
                build_backbone("nu2net", file, torch.device("cpu"))
            torch.save({"net": model.backbone.state_dict()}, file)
            with self.assertRaises(ValueError):
                build_backbone("nu2net", file, torch.device("cpu"))

    def test_nu2net_candidates_all_convolutions(self):
        model = tiny_model()
        configure_model(options(), model)
        expected = {id(p) for layer in model.modules() if isinstance(layer, nn.Conv2d)
                    for p in layer.parameters(recurse=False)}
        actual, names = collect_params(model)
        self.assertEqual({id(p) for p in actual}, expected)
        self.assertEqual(len(actual), len(set(names)))
        self.assertTrue(all(not list(layer.parameters()) for layer in model.modules() if isinstance(layer, nn.InstanceNorm2d)))

    def test_fisher_matches_root_algorithm(self):
        tta = engine()
        RootTTA = root_tta_class()
        old = object.__new__(RootTTA)
        old.model = deepcopy(tta.model)
        old.device = torch.device("cpu")
        old.opt = tta.opt
        old.fishers = {}
        old.compute_loss = lambda a, b: torch.sqrt((a - b).square() + 1e-3).mean()
        image = torch.rand(1, 3, 32, 32)
        tta.compute_fisher([(["a"], image)])
        old.compute_fisher([(["a"], image, torch.zeros_like(image))])
        for name in tta.fishers:
            torch.testing.assert_close(tta.fishers[name][0], old.fishers[name][0], rtol=1e-5, atol=1e-10)
            torch.testing.assert_close(tta.fishers[name][1], old.fishers[name][1], rtol=0, atol=0)

    def test_update_loss_gradients_restoration_and_ema_match_root(self):
        tta = engine()
        image = torch.rand(1, 3, 32, 32)
        tta.compute_fisher([(["a"], image)])
        RootTTA = root_tta_class()
        old = object.__new__(RootTTA)
        old.model = deepcopy(tta.model)
        old.model_teacher = deepcopy(tta.model_teacher)
        old.model_state = deepcopy(tta.model_state)
        old.fishers = deepcopy(tta.fishers)
        old.optimizer = torch.optim.Adam(collect_params(old.model)[0], lr=2e-4)
        old.opt = tta.opt
        old.device = torch.device("cpu")
        old.airnet = 0
        old.wt1 = LowFrequency()
        old.G_lossfn_weight, old.P_lossfn_weight, old.RL_lossfn_weight = 1, 0.01, 0.1
        first, second = DeterministicGenerator(), DeterministicGenerator()
        for step in range(2):
            image = torch.rand(1, 3, 32, 32)
            loss_old = old.test_time_adaptation(None, image, ["a"], first, step, None)
            losses = tta.adapt(image, ["a"], second)
            self.assertAlmostEqual(loss_old.item(), losses["loss"], places=5)
            for (name, p), q in zip(tta.model.named_parameters(), old.model.parameters()):
                torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-6)
                torch.testing.assert_close(p.grad, q.grad, rtol=3e-3, atol=1e-6)
                if p.requires_grad:
                    mask = tta.fishers[name][1].bool()
                    torch.testing.assert_close(p[mask], tta.model_state[name][mask], rtol=0, atol=0)
            with torch.no_grad():
                for teacher, student in zip(old.model_teacher.parameters(), old.model.parameters()):
                    teacher.mul_(0.45).add_(student, alpha=0.55)
            for a, b in zip(tta.model_teacher.parameters(), old.model_teacher.parameters()):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
        self.assertEqual(len(second.calls), 2)
        self.assertEqual(tta.steps, 2)
        self.assertTrue(torch.equal(tta._teacher_eight_view(image)[1], torch.ones_like(image)))

    def test_teacher_weight_zero_is_valid_ablation(self):
        tta = engine(teacher_weight=0, compute_fisher=0)
        losses = tta.adapt(torch.rand(1, 3, 32, 32), ["a"], DeterministicGenerator())
        self.assertEqual(losses["loss_t"], 0)

    def test_cache_rejects_empty_partial_bad_values_and_wrong_identity(self):
        tta = engine()
        tta.fisher_identity = {"model": "test", "ratio": 0.6}
        tta.compute_fisher([(["a"], torch.rand(1, 3, 32, 32))])
        with test_directory() as directory:
            file = Path(directory) / "fisher.pt"
            tta.save_fisher(file)
            payload = torch.load(file, weights_only=False)
            tta.load_fisher(file)
            bad = []
            empty = deepcopy(payload); empty["fishers"] = {}; bad.append(empty)
            partial = deepcopy(payload); partial["fishers"].pop(next(iter(partial["fishers"]))); bad.append(partial)
            wrong = deepcopy(payload); wrong["identity"]["ratio"] = 0.4; bad.append(wrong)
            values = deepcopy(payload); next(iter(values["fishers"].values()))[1].fill_(float("nan")); bad.append(values)
            ratio = deepcopy(payload); next(iter(ratio["fishers"].values()))[1].zero_(); bad.append(ratio)
            for item in bad:
                torch.save(item, file)
                with self.assertRaises(ValueError):
                    tta.load_fisher(file)

    def test_raw_dataset_never_reads_reference_and_limit_is_shared(self):
        with test_directory() as directory:
            root = Path(directory)
            Image.fromarray(np.random.randint(0, 255, (48, 64, 3), np.uint8)).save(root / "a.png")
            Image.new("RGB", (48, 48)).save(root / "b.png")
            samples = select_samples(root, limit=1)
            samples = [Sample(s.sample_id, s.raw_path, root / "missing_reference.png") for s in samples]
            raw = TargetRawDataset(samples, 32, seed=12)
            first = raw[0]
            raw[0]
            torch.testing.assert_close(first[1], raw[0][1], rtol=0, atol=0)
            self.assertEqual(len(raw), 1)
            self.assertEqual(first[1].shape, (3, 32, 32))
            with self.assertRaises(FileNotFoundError):
                EvaluationDataset(samples, "pad8")
            self.assertNotIn("reference", str(raw.identity()))

    def test_adaptive_crops_preserve_pixels_and_full_image_evaluation(self):
        with test_directory() as directory:
            root = Path(directory)
            for name, h, w in [('small', 159, 201), ('medium', 270, 301), ('large', 350, 401)]:
                pixels = np.random.default_rng(23).integers(0, 256, (h, w, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(root / (name + '.png'))
            samples = select_samples(root)
            data = TargetRawDataset(samples, 320, 23)
            expected = {'small.png': 152, 'medium.png': 264, 'large.png': 320}
            for i, s in enumerate(samples):
                name, cropped = data[i]
                record = data.identity()['samples'][i]
                self.assertEqual(cropped.shape, (3, expected[name], expected[name]))
                original = torch.from_numpy(np.array(Image.open(s.raw_path))).permute(2, 0, 1).float() / 255
                t, l, size = record['top'], record['left'], record['patch_size']
                torch.testing.assert_close(cropped, original[:, t:t+size, l:l+size], rtol=0, atol=0)
                reverse = TargetRawDataset(list(reversed(samples)), 320, 23)
                torch.testing.assert_close(cropped, reverse[len(samples)-1-i][1], rtol=0, atol=0)
            full = EvaluationDataset(samples, 'pad8')
            adapter = ImageRestorationAdapter(nn.Identity(), 'nu2net', {}, 'pad8')
            for item in full:
                raw = item['raw'].unsqueeze(0)
                torch.testing.assert_close(adapter(raw), raw, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                TargetRawDataset(samples, 320, adaptive=False)

    def test_cache_does_not_change_rng_or_loader_order(self):
        with test_directory() as directory:
            root = Path(directory)
            for i in range(3):
                Image.new("RGB", (48, 48), (i * 50, 80, 120)).save(root / f"{i}.png")
            data = TargetRawDataset(select_samples(root), 32, 23)
            tta = engine()
            tta.fisher_identity = tta.cache_identity(data)
            before = torch.get_rng_state().clone()
            python_before = random.getstate()
            loader = make_raw_loader(data, shuffle=False)
            tta.compute_fisher(loader)
            self.assertTrue(torch.equal(torch.get_rng_state(), before))
            self.assertEqual(random.getstate(), python_before)
            file = root / "cache.pt"
            tta.save_fisher(file)
            order1 = [names[0] for names, _ in make_raw_loader(data, shuffle=True)]
            tta.load_fisher(file)
            order2 = [names[0] for names, _ in make_raw_loader(data, shuffle=True)]
            self.assertEqual(order1, order2)
            # Target content and ratio changes invalidate the cache.
            previous = deepcopy(tta.fisher_identity)
            Image.new("RGB", (48, 48), (1, 2, 3)).save(root / "0.png")
            self.assertNotEqual(previous, tta.cache_identity(data))

    def test_imports_do_not_parse_arguments(self):
        with patch.object(sys, "argv", ["foreign", "--unrelated"]):
            importlib.import_module("options")
            importlib.import_module("tta")
        args = parse_args([])
        self.assertEqual((args.lr, args.ema_decay, args.teacher_weight, args.fisher_ratio), (2e-4, 0.45, 5, 0.6))

    def test_manifest_pairing_and_output_collisions(self):
        with test_directory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 40)).save(root / "input.png")
            Image.new("RGB", (32, 40)).save(root / "truth.png")
            manifest = root / "list.csv"
            manifest.write_text("sample_id,raw_path,reference_path\nscene,input.png,truth.png\n", encoding="utf-8")
            samples = select_samples(manifest=manifest)
            self.assertEqual(samples[0].reference_path, root / "truth.png")
            item = EvaluationDataset(samples, "pad8")[0]
            self.assertEqual(item["raw"].shape, item["reference"].shape)
            manifest.write_text("sample_id,raw_path\na.jpg,input.png\na.jpg.png,truth.png\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                select_samples(manifest=manifest)

    def test_default_width_nu2net_forward_backward(self):
        model = NU2Net(basic_channel=64)
        image = torch.rand(1, 3, 256, 256)
        output = model(image)
        self.assertEqual(output.shape, image.shape)
        F.l1_loss(output, image).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_vgg19_batching_preserves_loss_and_gradient(self):
        from torchvision.models.vgg import make_layers, cfgs
        from utils.loss_utils import PerceptualLoss
        # Real VGG19 feature architecture, fixed random weights; no network or
        # substitute loss. Pretrained weight download is deliberately not used.
        features = make_layers(cfgs["E"])
        with patch("torchvision.models.vgg19", return_value=SimpleNamespace(features=features)):
            loss_module = PerceptualLoss().eval().requires_grad_(False)
        pred = torch.rand(1, 3, 32, 32, requires_grad=True)
        target = torch.rand(1, 3, 32, 32, requires_grad=True)
        batched_loss = loss_module(pred, target)
        batched_grad, target_grad = torch.autograd.grad(batched_loss, (pred, target), allow_unused=True)
        self.assertIsNone(target_grad)
        pred_features = loss_module.vgg(pred)
        target_features = loss_module.vgg(target.detach())
        separate_loss = sum(weight * F.l1_loss(a, b)
                            for weight, a, b in zip(loss_module.weights, pred_features, target_features))
        separate_grad = torch.autograd.grad(separate_loss, pred)[0]
        torch.testing.assert_close(batched_loss, separate_loss, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(batched_grad, separate_grad, rtol=1e-4, atol=1e-8)

    def test_promptir_candidate_policy_and_structure(self):
        from einops import rearrange
        import numbers
        tree = ast.parse((ROOT / "net/model.py").read_text(encoding="utf-8"))
        adapted = ast.parse((ROOT / "movetoUIE/net/model.py").read_text(encoding="utf-8"))
        wanted = {"to_3d", "to_4d", "BiasFree_LayerNorm", "WithBias_LayerNorm", "LayerNorm",
                  "FeedForward", "Attention", "Downsample", "Upsample", "TransformerBlock",
                  "OverlapPatchEmbed", "PromptGenBlock", "PromptIR"}
        original = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted}
        migrated = {n.name: n for n in adapted.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted}
        for name in wanted:
            self.assertEqual(ast.dump(original[name]), ast.dump(migrated[name]))
        scope = {"torch": torch, "nn": nn, "F": F, "numbers": numbers, "rearrange": rearrange}
        exec(compile(ast.Module(body=list(original.values()), type_ignores=[]), "promptir_reference", "exec"), scope)
        model = scope["PromptIR"](decoder=True)
        configure_model(options(backbone="promptir"), model)
        params, names = collect_params(model)
        self.assertEqual(len(params), 329)
        self.assertFalse(any("prompt_param" in name or "temperature" in name for name in names))
        checkpoint_path = ROOT / "movetoUIE/pretrain/model.ckpt"
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("state_dict", checkpoint)
            state = {name.removeprefix("net."): value for name, value in state.items()}
            model.load_state_dict(state, strict=True)

    def test_raw_only_baseline_and_adapted_checkpoint_guard(self):
        from uie_tta import main
        with test_directory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            Image.new("RGB", (40, 32), (50, 100, 150)).save(root / "raw/one.png")
            model = tiny_model()
            checkpoint = {"net": model.backbone.state_dict(), "model_config": model.model_config}
            weights = root / "source.pth"
            torch.save(checkpoint, weights)
            args = ["--model_path", str(weights), "--lq_dir", str(root / "raw"),
                    "--device", "cpu", "--result_dir", str(root / "baseline")]
            # Baseline must not construct/import adaptation-only dependencies.
            import builtins
            original_import = builtins.__import__
            def guarded_import(name, *args, **kwargs):
                if name.startswith("RDDM") or name == "utils.loss_utils":
                    raise AssertionError(f"Baseline imported adaptation dependency: {name}")
                return original_import(name, *args, **kwargs)
            with patch("builtins.__import__", guarded_import):
                result = main(args + ["--mode", "baseline"])
            self.assertEqual(result["status"], "complete")
            self.assertNotIn("psnr", result["evaluation_records"][0])
            self.assertIn("not computed", (root / "baseline/metrics.txt").read_text())
            checkpoint["artifact_type"] = "adapted_inference_only"
            torch.save(checkpoint, weights)
            with self.assertRaises(ValueError):
                main(args + ["--mode", "adapt"])


if __name__ == "__main__":
    unittest.main()
