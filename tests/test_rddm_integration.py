"""Opt-in CPU smoke test with the real persistent RDDM (no pretrained VGG)."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


@unittest.skipUnless(os.environ.get("DCTTA_TEST_RDDM") == "1", "Set DCTTA_TEST_RDDM=1 to run real diffusion smoke test")
class RDDMIntegrationTests(unittest.TestCase):
    def test_five_steps_and_cross_image_state(self):
        from RDDM.net import ResidualDiffusionModel
        torch.set_num_threads(2)
        torch.manual_seed(23)
        root = Path(__file__).resolve().parents[2]
        opt = SimpleNamespace(device="cpu", verbose=False, save_rddm_results=False)
        generator = ResidualDiffusionModel(opt, image_size=32, device="cpu",
                                           results_folder=root / "tmp/rddm_test")
        parameter = next(generator.trainer.model.parameters())
        original = parameter.detach().clone()
        image = torch.rand(1, 3, 32, 32)
        pseudo = (0.7 * image + 0.1).detach()
        first = generator.train(image, pseudo, ["first.png"])
        intermediate = parameter.detach().clone()
        opt_id = id(generator.trainer.opt0)
        first_ema_step = generator.trainer.ema.step.item()
        # 同一个 RDDM 依次处理不同尺寸，不能重置优化器或 EMA。
        image2 = torch.rand(1, 3, 40, 40)
        second = generator.train(image2, (0.7 * image2 + 0.1).detach(), ["second.png"])
        sampled = generator.sample(image2, ["second.png"])
        self.assertEqual(generator.trainer.step, 5)
        self.assertEqual(first_ema_step, 5)
        self.assertEqual(generator.trainer.ema.step.item(), 10)
        self.assertEqual(id(generator.trainer.opt0), opt_id)
        self.assertFalse(torch.equal(original, intermediate))
        self.assertFalse(torch.equal(intermediate, parameter))
        self.assertEqual(first.shape, image.shape)
        self.assertEqual(second.shape, image2.shape)
        self.assertEqual(sampled.shape, image2.shape)
        self.assertTrue(torch.isfinite(sampled).all())
        self.assertTrue(torch.isfinite(first).all() and torch.isfinite(second).all())


if __name__ == "__main__":
    unittest.main()
