"""Persistent online RDDM; preserve the original training/sampling algorithm."""
from pathlib import Path

import torch

from .src.residual_denoising_diffusion_pytorch import ResidualDiffusion, Trainer, UnetRes


class ResidualDiffusionModel:
    def __init__(self, opt, image_size=256, train_batch_size=1, num_samples=1,
                 sum_scale=0.01, num_unet=1, objective="pred_res", test_res_or_noise="res",
                 debug=False, device=None, results_folder=None):
        self.device = torch.device(device or getattr(opt, "device", None) or
                                   ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.show_progress = bool(getattr(opt, "verbose", False))
        self.save_results = bool(getattr(opt, "save_rddm_results", False))
        self.train_num_steps = 5
        self.sampling_timesteps = 5
        self.model = UnetRes(dim=64, dim_mults=(1, 2, 4, 8), num_unet=num_unet,
                             condition=True, input_condition=False,
                             objective=objective, test_res_or_noise=test_res_or_noise)
        self.diffusion = ResidualDiffusion(
            self.model, image_size=image_size, timesteps=1000,
            sampling_timesteps=self.sampling_timesteps, objective=objective,
            loss_type="l2", condition=True, sum_scale=sum_scale,
            input_condition=False, input_condition_mask=False,
            test_res_or_noise=test_res_or_noise)
        self.diffusion.show_progress = self.show_progress
        self.trainer = Trainer(
            self.diffusion, train_batch_size=train_batch_size, num_samples=num_samples,
            train_lr=2e-4, train_num_steps=self.train_num_steps,
            gradient_accumulate_every=2, ema_decay=0.995, amp=False,
            convert_image_to="RGB", condition=True, save_and_sample_every=5,
            equalizeHist=False, crop_patch=False, generation=True, num_unet=num_unet,
            show_progress=self.show_progress, save_results=self.save_results,
            device=self.device,
            results_folder=Path(results_folder) if results_folder else Path(__file__).resolve().parents[1] / "results" / "rddm")

    def train(self, input, target, name):
        if input.device != self.device or target.device != self.device:
            raise RuntimeError("RDDM and backbone must use the same explicit device")
        if input.shape != target.shape or input.shape[0] != 1:
            raise ValueError("RDDM requires matching single-image input and pseudo-label")
        if any(size % 8 for size in input.shape[-2:]) or min(input.shape[-2:]) < 32:
            raise ValueError("RDDM spatial dimensions must be >=32 and divisible by 8")
        # Trainer resets only the local five-step counter; weights, RAdam
        # moments and EMA persist across calls. Inputs stay in [0,1] here;
        # the diffusion implementation performs its own internal conversion.
        return self.trainer.train(input, target, name)

    def sample(self, input, name):
        return self.trainer.test(input, name, last=True, FID=True)
