"""UIE 版 DCTTA 的集中实现。

阅读顺序：main() 查看整个目标数据集的处理过程；SRTTA.adapt() 查看单次更新；
SRTTA.compute_fisher() 和 fisher_restoration() 查看重要参数的选择与保护。

本文件集中设备、骨干接口、数据、八视图、适应和评价，参数定义位于 options.py。
NU²Net 原网络仍在 uie_backbones/nu2net.py，RDDM 扩散网络仍在 RDDM/，
感知及小波损失仍复用 utils/loss_utils.py，不复制或更改这些网络的数学实现。
"""

from contextlib import contextmanager
import hashlib
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
import csv
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
import json

from options import parse_args


# ============================================================================
# 一、设备、随机状态与权重读取
# ============================================================================

def resolve_device(device=None, cuda=0):
    selected = torch.device(device or (f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"))
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("Only cpu and cuda devices are supported")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = selected.index if selected.index is not None else cuda
        selected = torch.device("cuda", index)
        torch.cuda.set_device(selected)
    return selected


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextmanager
def isolated_rng(seed=None):
    """Auxiliary work must not change later RDDM noise or adaptation order."""
    # VGG 初始化和 Fisher 计算完成后恢复随机状态，避免缓存是否命中改变后续适应。
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            if seed is not None:
                seed_everything(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path):
    # These are local, user-supplied training checkpoints, including metadata.
    return torch.load(path, map_location="cpu", weights_only=False)


# ============================================================================
# 二、八视图增强、逆变换与教师伪标签
# ============================================================================

FISHER_OPS = ("", "v", "h", "t", "vh", "vt", "ht", "vht")
# Same order as the original flip/90-degree-rotation teacher on square patches.
TEACHER_OPS = ("v", "t", "h", "vht", "", "vt", "vh", "ht")


def transform_tensor(image, op, undo=False):
    if undo and "t" in op:
        image = image.transpose(-2, -1).contiguous()
    if "v" in op:
        image = image.flip(-1).contiguous()
    if "h" in op:
        image = image.flip(-2).contiguous()
    if not undo and "t" in op:
        image = image.transpose(-2, -1).contiguous()
    return image


@torch.no_grad()
def eight_view_mean(model, image, observer=None):
    # NU²Net 的 tail 会检查整个 batch 是否越界。八个视图不能合批，否则可能改变输出。
    # 每个输出先逆变换回原坐标，再求均值；整个教师伪标签过程不参与反向传播。
    # NU2Net's normalization tail has a batch-wide branch condition. Do not cat
    # different views (or different images) into a single network forward.
    if image.shape[0] != 1:
        raise ValueError("Eight-view inference requires batch_size=1")
    outputs = [transform_tensor(model(transform_tensor(image, op)), op, undo=True)
               for op in TEACHER_OPS]
    if observer is not None:
        observer('teacher_views', dict(zip(TEACHER_OPS, outputs)))
    total = outputs[0]
    for output in outputs[1:]:
        total = total + output
    return total / 8.0


# ============================================================================
# 三、骨干尺寸接口：保留 NU²Net 原始网络和输出归一化
# ============================================================================

class ImageRestorationAdapter(nn.Module):
    def __init__(self, backbone, name, model_config, spatial_policy="pad8"):
        super().__init__()
        self.backbone = backbone
        self.backbone_name = name
        self.model_config = dict(model_config)
        self.spatial_policy = spatial_policy

    def forward(self, image):
        # 这里只补齐空间尺寸并裁回输出，不改 NU²Net 残差、tail 或学生计算图。
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected BCHW RGB input")
        height, width = image.shape[-2:]
        if min(height, width) < 16:
            raise ValueError("Restoration inputs must be at least 16x16")
        if self.spatial_policy == "pad8":
            bottom, right = (-height) % 8, (-width) % 8
            if bottom or right:
                mode = "reflect" if bottom < height and right < width else "replicate"
                image = F.pad(image, (0, right, 0, bottom), mode=mode)
        elif self.spatial_policy == "crop16":
            # Legacy PromptIR evaluation protocol; also crop reference in data.
            if height % 16 or width % 16:
                raise ValueError("crop16 preprocessing must happen in the dataset")
        else:
            raise ValueError(f"Unknown spatial policy: {self.spatial_policy}")
        restored = self.backbone(image)
        return restored[..., :height, :width]


# ============================================================================
# 四、骨干构建、严格权重加载与可更新参数选择
# ============================================================================

def strip_uniform_prefix(state, prefix):
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix):]: value for key, value in state.items()}
    return state


def build_backbone(name, checkpoint_path, device):
    # 先向裸模型严格加载源权重，再包装尺寸接口；不继承源训练优化器状态。
    checkpoint = load_checkpoint(checkpoint_path)
    if name == "nu2net":
        from uie_backbones.nu2net import NU2Net
        required = {"basic_channel", "is_residual", "tail"}
        if not isinstance(checkpoint, dict) or not {"net", "model_config"} <= checkpoint.keys():
            raise ValueError("NU2Net checkpoint must contain net and model_config (see handoff)")
        config = checkpoint["model_config"]
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError(f"NU2Net model_config must contain exactly {sorted(required)}")
        if config["tail"] != "norm" or config["is_residual"] is not True:
            raise ValueError("This migration uses the original residual NU2Net with tail='norm'")
        if not isinstance(config["basic_channel"], int) or config["basic_channel"] < 4:
            raise ValueError("Invalid NU2Net basic_channel")
        model = NU2Net(**config)
        state = checkpoint["net"]
        spatial_policy = "pad8"
    elif name == "promptir":
        # NU2Net users never import this mixed legacy module.
        from net.model import PromptIR
        config = {"decoder": True}
        model = PromptIR(**config)
        state = checkpoint.get("state_dict", checkpoint)
        state = strip_uniform_prefix(state, "module.")
        state = strip_uniform_prefix(state, "net.")
        spatial_policy = "crop16"
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    state = strip_uniform_prefix(state, "module.")
    model.load_state_dict(state, strict=True)
    adapter = ImageRestorationAdapter(model, name, config, spatial_policy).to(device).eval()
    adapter.source_sha256 = sha256_file(checkpoint_path)
    adapter.source_path = str(Path(checkpoint_path).resolve())
    adapter.checkpoint_kind = checkpoint.get("artifact_type", "source")
    return adapter


def configure_model(opt, model, airnet=False):
    # 先定义骨干的候选更新集合，Fisher 再在每个候选张量内部选择保护元素。
    # NU²Net 更新全部卷积参数；PromptIR 保留原实际优化器的329个参数张量。
    if airnet:
        raise ValueError("AirNet requires an explicit backbone adapter; it is not a NU2Net alias")
    bare = getattr(model, "backbone", model)
    name = getattr(model, "backbone_name", getattr(opt, "backbone", "promptir"))
    bare.requires_grad_(False)
    if name == "nu2net":
        # InstanceNorm in upstream NU2Net has affine=False. All convolution
        # weights and biases are eligible; TIPS protects elements within them.
        for module in bare.modules():
            if isinstance(module, nn.Conv2d):
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad_(True)
    elif name == "promptir":
        # Match the root repository's *actual optimizer*, not the broader
        # requires_grad flags that previously included prompt_param/temperature.
        for key, parameter in bare.named_parameters():
            prefix = key.split(".")[0]
            parameter.requires_grad_(
                any(token in prefix for token in ("prompt", "encoder", "decoder"))
                and key.rsplit(".", 1)[-1] in {"weight", "bias"}
            )
    else:
        raise ValueError(f"No parameter selection policy for {name}")
    model.train()
    params, _ = collect_params(model)
    if not params:
        raise ValueError("Backbone has no adaptation parameters")
    logging.getLogger(__name__).info("TTA parameters: %d tensors, %d elements", len(params),
                                     sum(parameter.numel() for parameter in params))
    return model


def collect_params(model):
    selected = [(name, parameter) for name, parameter in model.named_parameters()
                if parameter.requires_grad]
    return [parameter for _, parameter in selected], [name for name, _ in selected]


# ============================================================================
# 五、目标数据：适应只读取 raw，reference 仅用于评价
# ============================================================================

@dataclass(frozen=True)
class Sample:
    sample_id: str
    raw_path: Path
    reference_path: Path = None


def select_samples(lq_dir=None, gt_dir=None, manifest=None, data_root=None, limit=0):
    # 只在这里选一次目标子集；Fisher、逐图适应、最终评价共用同一组 sample_id。
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    samples = []
    if manifest:
        manifest = Path(manifest).resolve()
        base = Path(data_root).resolve() if data_root else manifest.parent
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"sample_id", "raw_path"} <= set(reader.fieldnames or []):
                raise ValueError("Manifest needs sample_id and raw_path columns")
            for row in reader:
                ref = row.get("reference_path", "").strip()
                samples.append(Sample(row["sample_id"], (base / row["raw_path"]).resolve(),
                                      (base / ref).resolve() if ref else None))
    else:
        if not lq_dir:
            raise ValueError("Provide --lq_dir or --manifest")
        directory = Path(lq_dir).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        reference_dir = Path(gt_dir).resolve() if gt_dir else None
        for raw in sorted(directory.iterdir()):
            if raw.is_file() and raw.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                samples.append(Sample(raw.name, raw, reference_dir / raw.name if reference_dir else None))
    if not samples:
        raise ValueError("Target dataset is empty")
    seen, raw_seen, output_seen = set(), set(), set()
    for sample in samples:
        if (not sample.sample_id or sample.sample_id in {".", ".."}
                or any(char in sample.sample_id for char in '/\\:\x00')):
            raise ValueError(f"sample_id must be a safe filename: {sample.sample_id!r}")
        if sample.sample_id.casefold() in seen or sample.raw_path in raw_seen:
            raise ValueError(f"Duplicate sample or raw path: {sample.sample_id}")
        output_name = sample.sample_id if sample.sample_id.lower().endswith(".png") else sample.sample_id + ".png"
        if output_name.casefold() in output_seen:
            raise ValueError(f"Colliding output filename: {sample.sample_id}")
        output_seen.add(output_name.casefold())
        seen.add(sample.sample_id.casefold())
        raw_seen.add(sample.raw_path)
        if not sample.raw_path.is_file():
            raise FileNotFoundError(sample.raw_path)
    # One selection shared by Fisher, adaptation, baseline and final evaluation.
    return samples[:limit] if limit else samples


def read_rgb(path):
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def crop16(image):
    height, width = image.shape[-2:]
    if min(height, width) < 16:
        raise ValueError("Images must be at least 16x16")
    # Match legacy utils.image_utils.crop_img (symmetric boundary crop).
    dh, dw = height % 16, width % 16
    return image[..., dh // 2:height - (dh - dh // 2), dw // 2:width - (dw - dw // 2)]


class TargetRawDataset(Dataset):
    # 目标适应数据集没有读取 reference 的路径，避免参考图进入任何适应损失。
    def __init__(self, samples, patch_size=320, seed=23, adaptive=True):
        self.samples = samples
        self.patch_size = patch_size
        self.seed = seed
        self.adaptive = adaptive
        self.crops = []
        if patch_size < 32 or patch_size % 8:
            raise ValueError("RDDM/NU2Net patches must be >=32 and divisible by 8")
        for sample in samples:
            with Image.open(sample.raw_path) as image:
                width, height = image.size
                # UIE 仅调整裁剪边长，不缩放或填充；大图保留原有 patch 上限。
                size = (min(patch_size, height, width) // 8) * 8 if adaptive else patch_size
                if size < 32 or min(height, width) < size:
                    raise ValueError(f"{sample.sample_id}: image {image.size} cannot support patch {size} (minimum 32)")
                digest = hashlib.sha256(f"{seed}:{sample.sample_id}".encode()).digest()
                rng = random.Random(int.from_bytes(digest[:8], "big"))
                self.crops.append(dict(height=height, width=width, patch_size=size,
                                       top=rng.randint(0, height-size), left=rng.randint(0, width-size)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = read_rgb(sample.raw_path)
        height, width = image.shape[-2:]
        crop = self.crops[index]
        if (height, width) != (crop['height'], crop['width']):
            raise ValueError(f"Image dimensions changed after preparation: {sample.sample_id}")
        # Fisher 与适配复用同一确定性裁剪，不受读取顺序或 worker 数量影响。
        top, left, size = crop['top'], crop['left'], crop['patch_size']
        return sample.sample_id, image[:, top:top+size, left:left+size]

    def identity(self):
        return {
            "patch_size": self.patch_size, "crop_seed": self.seed,
            "crop_policy": "per_sample_sha256_adaptive_crop_v2" if self.adaptive else "per_sample_sha256_fixed_crop_v2",
            "color_range": "RGB_0_1",
            "samples": [{"id": s.sample_id, "raw_sha256": sha256_file(s.raw_path), **crop}
                        for s, crop in zip(self.samples, self.crops)],
        }


class EvaluationDataset(Dataset):
    def __init__(self, samples, spatial_policy):
        self.samples = samples
        self.spatial_policy = spatial_policy
        references = [sample.reference_path is not None for sample in samples]
        if any(references) and not all(references):
            raise ValueError("Evaluation references must be provided for all selected images or none")
        self.has_reference = all(references)
        # Check pairing before adaptation starts, but never pass reference to TTA.
        for sample in samples:
            if sample.reference_path is not None:
                if not sample.reference_path.is_file():
                    raise FileNotFoundError(sample.reference_path)
                with Image.open(sample.raw_path) as raw, Image.open(sample.reference_path) as reference:
                    if raw.size != reference.size:
                        raise ValueError(f"Raw/reference size mismatch: {sample.sample_id}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        raw = read_rgb(sample.raw_path)
        result = {"sample_id": sample.sample_id, "raw": raw}
        if sample.reference_path:
            result["reference"] = read_rgb(sample.reference_path)
        if self.spatial_policy == "crop16":
            for name in ("raw", "reference"):
                if name in result:
                    result[name] = crop16(result[name])
        return result


def make_raw_loader(dataset, num_workers=0, shuffle=True, seed=23, pin_memory=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, generator=generator,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=False)


# ============================================================================
# 六、SRTTA：一致性损失、学生更新、Fisher 保护与教师 EMA
# ============================================================================

ALGORITHM_VERSION = "dctta-code-reference-uie-v1"


def compute_loss(pred, target, eps=1e-3):
    return torch.sqrt((pred - target).square() + eps).mean()


class SRTTA:
    """测试时适应器：保留原仓库损失、Fisher 恢复和 EMA 更新顺序。"""
    def __init__(self, opt, model, optimizer, logger=None, perceptual_loss=None, wavelet=None):
        self.opt = opt
        self.model = model
        self.optimizer = optimizer
        self.logger = logger or logging.getLogger(__name__)
        self.device = next(model.parameters()).device
        self.E_decay = getattr(opt, "ema_decay", 0.45)
        self.G_lossfn_weight = 1.0
        self.P_lossfn_weight = 0.01
        self.RL_lossfn_weight = 0.1
        self.model_state = deepcopy(model.state_dict())
        # 保存不可变源参数供 TIPS 恢复；teacher 深复制，避免与 student 共享存储。
        self.model_teacher = deepcopy(model).requires_grad_(False).eval()
        # Loss construction must not affect diffusion initialization/noise.
        with isolated_rng():
            if perceptual_loss is None or wavelet is None:
                from utils.loss_utils import PerceptualLoss, WaveletTransform
            self.perceptual_loss = perceptual_loss if perceptual_loss is not None else PerceptualLoss()
            self.wt1 = wavelet if wavelet is not None else WaveletTransform(1, True)
        self.perceptual_loss.to(self.device).requires_grad_(False).eval()
        self.wt1.to(self.device).requires_grad_(False)
        self.fishers = {}
        self.fisher_identity = None
        self.steps = 0

    def candidate_parameters(self):
        return {name: p for name, p in self.model.named_parameters() if p.requires_grad}

    def _teacher_eight_view(self, image):
        pseudo = eight_view_mean(self.model_teacher, image, getattr(self, 'observer', None))
        # 原实现对重复的确定性八视图均值求方差，等效 confidence=1。
        # 此处不启用“八个视图间方差加权”，因为那会改变像素损失的定义。
        # Original deterministic repeated-ensemble variance is zero. A variance
        # across individual views is a different objective, not an optimization.
        return pseudo, torch.ones_like(pseudo)

    def compute_loss2(self, pred, target, pred_low, target_low, confidence):
        # 每一类一致性损失均沿用原仓库：像素 L1 + 0.01×VGG感知 L1 + 0.1×低频 L1。
        pixel = F.l1_loss(confidence * pred, confidence * target)
        perceptual = self.perceptual_loss(pred, target)
        low = F.l1_loss(pred_low, target_low)
        observer = getattr(self, 'observer', None)
        if observer is not None:
            observer('loss_components', dict(pixel=pixel, perceptual=perceptual, low=low,
                     weights=[self.G_lossfn_weight, self.P_lossfn_weight, self.RL_lossfn_weight]))
        return (self.G_lossfn_weight * pixel + self.P_lossfn_weight * perceptual
                + self.RL_lossfn_weight * low)

    def adapt(self, image, names, degradation_model):
        """单次完整更新：teacher → RDDM → student → Fisher恢复 → teacher EMA。"""
        if image.shape[0] != 1:
            raise ValueError("The reference online protocol requires batch_size=1")
        image = image.to(self.device)
        self.model.train()
        self.model_teacher.eval()
        pseudo, confidence = self._teacher_eight_view(image)
        observer = getattr(self, 'observer', None)
        if observer is not None:
            observer('student_before', dict(model=self.model, teacher=self.model_teacher,
                     optimizer=self.optimizer, raw=image, pseudo=pseudo, confidence=confidence))
        # 1. 用当前 raw 与 teacher 伪标签训练已有 RDDM；随后仍以原 raw 为采样条件。
        #    RDDM 对象由 main 在图像循环外创建，权重、优化器和 EMA 跨图累计。
        # Keep the original direction: train(input=x_in, target=pseudo), then
        # sample conditioned on x_in. Do not reinterpret the argument order.
        second_lq = degradation_model.train(image, pseudo, names).detach()
        if second_lq.device != self.device or second_lq.shape != image.shape:
            raise RuntimeError("RDDM output shape/device differs from restoration input")
        if not torch.isfinite(second_lq).all():
            raise FloatingPointError("RDDM produced non-finite values")
        with torch.no_grad():
            original_output = self.model(image)
        # 2. 同一个学生分别处理原图和再退化图；原图输出停止梯度，更新从本分支传回。
        adapted_output = self.model(second_lq)
        if observer is not None:
            observer('student_outputs', dict(raw_output=original_output,
                     regenerated=second_lq, regenerated_output=adapted_output))
        pred_low = self.wt1(adapted_output.permute(0, 2, 3, 1))[:, 0]
        input_low = self.wt1(original_output.permute(0, 2, 3, 1))[:, 0]
        pseudo_low = self.wt1(pseudo.permute(0, 2, 3, 1))[:, 0]
        loss_s = self.compute_loss2(adapted_output, original_output, pred_low, input_low, confidence)
        # 3. loss_s 保持两路恢复一致；loss_t 将再退化图恢复结果约束到教师伪标签。
        loss_t = adapted_output.new_zeros(())
        if self.opt.teacher_weight > 0:
            loss_t = self.opt.teacher_weight * self.compute_loss2(
                adapted_output, pseudo, pred_low, pseudo_low, confidence)
        loss = loss_s + loss_t
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DCTTA loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in self.candidate_parameters().values()):
            raise FloatingPointError("Non-finite DCTTA gradients")
        self.optimizer.step()
        if observer is not None:
            observer('student_adam', dict(model=self.model, optimizer=self.optimizer))
        # 4. 必须先完成 Adam，再恢复重要元素，最后更新教师；不能交换这三个步骤。
        if self.opt.compute_fisher:
            self.fisher_restoration()
        self.update_teacher()
        if observer is not None:
            observer('student_restored', dict(model=self.model, teacher=self.model_teacher))
        self.steps += 1
        losses = {"loss_s": loss_s.item(), "loss_t": loss_t.item(), "loss": loss.item()}
        if getattr(self.opt, "verbose", False):
            self.logger.info("DCTTA step %d: %s", self.steps, losses)
        return losses

    def __call__(self, train_loader, input_img, img_name, degeneration_model):
        # Compatibility with the old call site. No unused restoration forward.
        losses = None
        for _ in range(self.opt.iterations):
            losses = self.adapt(input_img, img_name, degeneration_model)
        return losses

    @torch.no_grad()
    def update_teacher(self):
        # teacher = 0.45×旧teacher + 0.55×已执行Fisher恢复的student（默认配置）。
        students = dict(self.model.named_parameters())
        for name, teacher in self.model_teacher.named_parameters():
            teacher.mul_(self.E_decay).add_(students[name], alpha=1 - self.E_decay)
        # Supported backbones have no mutable running-statistic buffers. Never
        # silently introduce a new EMA policy for another architecture.
        teachers = dict(self.model_teacher.named_buffers())
        for name, buffer in self.model.named_buffers():
            if not torch.equal(teachers[name], buffer):
                raise RuntimeError(f"Mutable buffer requires an explicit backbone policy: {name}")

    @torch.no_grad()
    def fisher_restoration(self):
        # 使用源权重覆盖 mask=1 的元素。仅清零梯度不足以抵消 Adam 已有动量。
        candidates = self.candidate_parameters()
        if set(self.fishers) != set(candidates):
            raise ValueError("Fisher masks must cover all adaptation parameters")
        for name, parameter in candidates.items():
            mask = self.fishers[name][1].bool()
            # Post-Adam restoration preserves selected source values even when
            # optimizer moments are nonzero. Retain the original Adam history.
            parameter.copy_(torch.where(mask, self.model_state[name], parameter))

    def cache_identity(self, raw_dataset):
        candidates = self.candidate_parameters()
        state_digest = hashlib.sha256()
        for name, tensor in sorted(self.model_state.items()):
            state_digest.update(name.encode())
            state_digest.update(str(tensor.dtype).encode())
            state_digest.update(str(tuple(tensor.shape)).encode())
            state_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        # 缓存校验绑定实际实现文件；旧分散模块现在只是兼容入口。
        source_files = [Path(__file__), Path(__file__).with_name("options.py")]
        backbone_dir = Path(__file__).resolve().parent / "uie_backbones"
        if getattr(self.model, "backbone_name", None) == "nu2net":
            source_files.append(backbone_dir / "nu2net.py")
        else:
            source_files.append(Path(__file__).resolve().parent / "net" / "model.py")
        return {
            "algorithm": ALGORITHM_VERSION,
            "implementation": {path.name: sha256_file(path) for path in source_files},
            "torch_version": str(torch.__version__), "device_type": self.device.type,
            "source_checkpoint_sha256": getattr(self.model, "source_sha256", None),
            "source_state_sha256": state_digest.hexdigest(),
            "backbone": getattr(self.model, "backbone_name", type(self.model).__name__),
            "model_config": getattr(self.model, "model_config", {}),
            "spatial_policy": getattr(self.model, "spatial_policy", "identity"),
            "fisher_ratio": self.opt.fisher_ratio,
            "parameters": {name: {"shape": list(p.shape), "dtype": str(p.dtype)}
                           for name, p in candidates.items()},
            "target": raw_dataset.identity(),
        }

    def compute_fisher(self, loader):
        """只使用目标 raw，按增强一致性梯度平方估计重要性，不执行优化器更新。"""
        if not 0 <= self.opt.fisher_ratio <= 1:
            raise ValueError("fisher_ratio must lie in [0, 1]")
        parameters = self.candidate_parameters()
        if not parameters or len(loader) == 0:
            raise ValueError("Fisher requires parameters and nonempty target data")
        sums = {name: torch.zeros_like(p) for name, p in parameters.items()}
        was_training = self.model.training
        self.model.train()
        try:
            # Isolate all random state; use a separate non-shuffled raw loader.
            with isolated_rng(getattr(self.opt, "seed", 23)):
                for _, image in tqdm(loader, desc="Fisher", disable=not getattr(self.opt, "verbose", False)):
                    image = image.to(self.device)
                    if image.shape[0] != 1:
                        raise ValueError("Fisher reference protocol requires batch_size=1")
                    self.model.zero_grad(set_to_none=True)
                    outputs = []
                    for op in reversed(FISHER_OPS):
                        # 七个增强视图不求梯度；最后的原图分支保留梯度，与原实现相同。
                        if op:
                            with torch.no_grad():
                                output = transform_tensor(self.model(transform_tensor(image, op)), op, undo=True)
                        else:
                            output = self.model(image)
                        outputs.append(output)
                    pseudo = torch.cat(outputs).mean(0, keepdim=True).detach()
                    compute_loss(outputs[-1], pseudo).backward()
                    for name, parameter in parameters.items():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise ValueError(f"Missing or non-finite Fisher gradient: {name}")
                        sums[name].add_(parameter.grad.detach().square())
            fishers = {}
            for name, parameter in parameters.items():
                fisher = sums[name].flatten()
                count = int(fisher.numel() * self.opt.fisher_ratio)
                # 沿用原代码的“每个张量分别 topk”，不改成跨整个网络的全局阈值。
                mask = torch.zeros_like(fisher)
                mask[torch.topk(fisher, k=count).indices] = 1
                fishers[name] = [fisher, mask.view_as(parameter)]
            self.fishers = fishers
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)
        return self.fishers

    def save_fisher(self, path):
        if self.fisher_identity is None:
            raise ValueError("Set Fisher cache identity before saving")
        self._validate_fishers(self.fishers)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"identity": self.fisher_identity,
                    "fishers": {name: [f.cpu(), mask.cpu()] for name, (f, mask) in self.fishers.items()}}, path)

    def _validate_fishers(self, fishers):
        parameters = self.candidate_parameters()
        if not isinstance(fishers, dict) or not fishers or set(fishers) != set(parameters):
            raise ValueError("Empty or incomplete Fisher cache")
        for name, parameter in parameters.items():
            values = fishers[name]
            if not isinstance(values, (tuple, list)) or len(values) != 2:
                raise ValueError(f"Invalid Fisher entry: {name}")
            fisher, mask = values
            if not torch.is_tensor(fisher) or not torch.is_tensor(mask):
                raise ValueError(f"Fisher and mask must be tensors: {name}")
            if fisher.shape != (parameter.numel(),) or mask.shape != parameter.shape:
                raise ValueError(f"Fisher shape mismatch: {name}")
            if (not torch.isfinite(fisher).all() or (fisher < 0).any()
                    or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all()):
                raise ValueError(f"Invalid Fisher values: {name}")
            if int(mask.sum().item()) != int(parameter.numel() * self.opt.fisher_ratio):
                raise ValueError(f"Fisher mask retention ratio mismatch: {name}")

    def load_fisher(self, path):
        # 权重、数据、参数集合、比例或实现身份不一致时明确失败，不静默复用错误掩码。
        payload = load_checkpoint(path)
        if self.fisher_identity is None or not isinstance(payload, dict) or payload.get("identity") != self.fisher_identity:
            raise ValueError("Fisher identity mismatch; recompute for this model, target and configuration")
        fishers = payload.get("fishers")
        self._validate_fishers(fishers)
        self.fishers = {name: [f.to(self.device), mask.to(self.device)]
                        for name, (f, mask) in fishers.items()}
        return self.fishers


# ============================================================================
# 七、统一评价与图像保存
# ============================================================================

def evaluate(model, dataset, output_dir, dataset_name, metrics_filename="metrics.txt", num_workers=0, ensemble=False):
    # 只有评价阶段读取 reference。指标在浮点输出上计算，再保存量化后的 PNG。
    device = next(model.parameters()).device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(metrics_filename).name != metrics_filename:
        raise ValueError("metrics_filename must be a filename, not a path")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                        generator=torch.Generator().manual_seed(0))
    records = []
    model.eval()
    if dataset.has_reference:
        from utils.val_utils import compute_psnr_ssim, write_metrics_report
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", unit="img"):
            raw = batch["raw"].to(device)
            result = eight_view_mean(model, raw) if ensemble else model(raw)
            if result.shape != raw.shape or not torch.isfinite(result).all():
                raise RuntimeError("Invalid restored image")
            sample_id = batch["sample_id"][0]
            record = {"image": sample_id}
            if dataset.has_reference:
                psnr, ssim = compute_psnr_ssim(result, batch["reference"].to(device))
                record.update(psnr=float(psnr), ssim=float(ssim))
            filename = sample_id if sample_id.lower().endswith(".png") else sample_id + ".png"
            record["output_file"] = filename
            array = result[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            Image.fromarray((array * 255).astype(np.uint8)).save(output_dir / filename)
            records.append(record)
    report_path = output_dir / metrics_filename
    if dataset.has_reference:
        write_metrics_report(records, report_path, dataset_name)
    else:
        report_path.write_text(f"Dataset: {dataset_name}\nImages evaluated: {len(records)}\n"
                               "Reference: unavailable; PSNR/SSIM not computed\n\n"
                               + "\n".join(record["image"] for record in records) + "\n", encoding="utf-8")
    return records


# ============================================================================
# 八、主流程：初始化 → Fisher → 逐图适应 → 最终评价
# ============================================================================

ROOT = Path(__file__).resolve().parent


def main(argv=None, default_backbone="nu2net", default_mode="adapt"):
    """入口共用主流程：baseline/x8 直接评价；adapt 先处理整个目标集再评价。"""
    opt = parse_args(argv, default_backbone, default_mode)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("dctta")
    device = resolve_device(opt.device, opt.cuda)
    opt.device = str(device)
    seed_everything(opt.seed)
    if opt.backbone == "promptir":
        opt.model_path = opt.model_path or str(ROOT / "pretrain" / "model.ckpt")
        if not opt.manifest and not opt.lq_dir:
            opt.lq_dir = str(ROOT / "testdata" / "Rain100H" / "LQ")
            opt.gt_dir = opt.gt_dir or str(ROOT / "testdata" / "Rain100H" / "GT")
    elif not opt.model_path:
        raise ValueError("NU2Net requires --model_path pointing to a trained source checkpoint")
    if opt.mode == "adapt" and opt.patch_size % (16 if opt.backbone == "promptir" else 8):
        raise ValueError("patch_size is incompatible with the backbone preprocessing protocol")
    samples = select_samples(opt.lq_dir, opt.gt_dir, opt.manifest, opt.data_root, opt.limit)
    with isolated_rng(opt.seed):
        model = build_backbone(opt.backbone, opt.model_path, device)
    if model.checkpoint_kind == "adapted_inference_only" and opt.mode != "evaluate":
        raise ValueError("Adapted weights are for --mode evaluate; start each experiment from source weights")
    dataset_name = opt.dataset_name or (Path(opt.manifest).stem if opt.manifest else Path(opt.lq_dir).resolve().parent.name)
    output = Path(opt.result_dir) if opt.result_dir else ROOT / "results" / f"{dataset_name}_{opt.backbone}_{opt.mode}"
    output.mkdir(parents=True, exist_ok=True)
    evaluation_data = EvaluationDataset(samples, model.spatial_policy)
    logger.info("Setup: backbone=%s, mode=%s, device=%s, images=%d", opt.backbone, opt.mode, device, len(samples))
    run_info = {
        "status": "running", "configuration": vars(opt), "dataset": dataset_name,
        "source_checkpoint_sha256": model.source_sha256, "model_config": model.model_config,
        "spatial_policy": model.spatial_policy, "metrics": "RGB float clip[0,1], data_range=1, skimage defaults",
        "selected_samples": [{"id": s.sample_id, "raw": str(s.raw_path),
                              "reference": str(s.reference_path) if s.reference_path else None} for s in samples],
    }
    info_path = output / "run.json"
    info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    if opt.mode == "adapt":
        from RDDM.net import ResidualDiffusionModel
        raw_data = TargetRawDataset(samples, opt.patch_size, opt.seed, adaptive=opt.backbone == "nu2net")
        configure_model(opt, model)
        parameters, names = collect_params(model)
        optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=tuple(opt.betas))
        tta = SRTTA(opt, model, optimizer, logger=logger)
        tta.fisher_identity = tta.cache_identity(raw_data)
        run_info.update(algorithm=ALGORITHM_VERSION, candidate_parameters=names,
                        target_identity=tta.fisher_identity["target"])
        identity_hash = hashlib.sha256(json.dumps(tta.fisher_identity, sort_keys=True).encode()).hexdigest()[:20]
        fisher_path = Path(opt.fisher_cache_path) if opt.fisher_cache_path else (
            ROOT / "pretrain" / f"fisher_{opt.backbone}_{identity_hash}.pt")
        if opt.compute_fisher:
            if fisher_path.exists() and not opt.recompute_fisher:
                # A stale cache is an error, never silently ignored or accepted.
                tta.load_fisher(fisher_path)
                logger.info("Fisher: loaded %s", fisher_path)
                run_info["fisher"] = "loaded"
            else:
                loader = make_raw_loader(raw_data, opt.num_workers, shuffle=False, seed=opt.seed)
                tta.compute_fisher(loader)
                tta.save_fisher(fisher_path)
                logger.info("Fisher: computed and saved %s", fisher_path)
                run_info["fisher"] = "computed"
            run_info["fisher_cache"] = str(fisher_path.resolve())
        else:
            logger.info("Fisher: disabled (ablation, not full DCTTA)")
            run_info["fisher"] = "disabled_ablation"
        # Create ONCE per target run. Parameters, optimizer and RDDM EMA persist
        # across every image. VGG/Fisher work cannot change this initialization.
        with isolated_rng(opt.seed + 1):
            # 只初始化一次退化生成器；不要移入下方逐图循环，否则会丢失跨图在线学习。
            generator = ResidualDiffusionModel(opt, image_size=opt.patch_size, device=device,
                                               results_folder=output / "rddm")
        loader = make_raw_loader(raw_data, opt.num_workers, shuffle=bool(opt.shuffle), seed=opt.seed,
                                 pin_memory=device.type == "cuda")
        run_info["adaptation_order"] = []
        with (output / "adaptation.jsonl").open("w", encoding="utf-8") as stream:
            for names, image in tqdm(loader, desc="TTA", unit="img"):
                run_info["adaptation_order"].extend(names)
                for iteration in range(opt.iterations):
                    losses = tta.adapt(image, names, generator)
                    stream.write(json.dumps({"image": names[0], "iteration": iteration + 1, **losses}) + "\n")
                    stream.flush()
        # Deployment artifact, not a resumable online run. Resuming adaptation
        # would also require RDDM, teacher, optimizer and all RNG states.
        state = {name: p.detach().cpu() for name, p in model.backbone.state_dict().items()}
        # 保存最终学生用于部署/评价；完整断点恢复还需 RDDM、教师、优化器和随机状态。
        checkpoint = {"model_config": model.model_config, "backbone": opt.backbone,
                      "artifact_type": "adapted_inference_only", "source_sha256": model.source_sha256,
                      "algorithm": ALGORITHM_VERSION, "adaptation_steps": tta.steps}
        checkpoint["net" if opt.backbone == "nu2net" else "state_dict"] = state
        torch.save(checkpoint, output / "adapted_model.pth")
        run_info["adaptation_steps"] = tta.steps
        info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Adaptation complete; evaluating final student")
    records = evaluate(model, evaluation_data, output, dataset_name, opt.metrics_filename,
                       opt.num_workers, ensemble=opt.mode == "x8")
    run_info.update(status="complete", evaluation_records=records)
    info_path.write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Done: %s", output.resolve())
    return run_info
