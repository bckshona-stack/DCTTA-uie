"""直接拟合单图有效衰减/背景光；不读取reference，不持有教师或学生。"""
from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn


@dataclass(frozen=True)
class FitConfig:
    grid_size: int = 32
    t_min: float = .1
    init_t: float = .9
    init_noise: float = .05
    tv_weight: float = .01
    lr: float = .05
    seed: int = 23

    def __post_init__(self):
        if type(self.grid_size) is not int or not 2 <= self.grid_size <= 128:
            raise ValueError('grid_size must be an integer in [2,128]')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0,2**32)')
        if not all(math.isfinite(x) for x in
                   (self.t_min, self.init_t, self.init_noise, self.tv_weight, self.lr)):
            raise ValueError('Non-finite fit configuration')
        if not 0 < self.t_min < self.init_t < 1:
            raise ValueError('Require 0 < t_min < init_t < 1')
        if self.init_noise < 0 or self.tv_weight < 0 or not 0 < self.lr <= 1:
            raise ValueError('Invalid noise/TV/learning rate')

    def identity(self):
        return asdict(self)


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(tuple(value.shape)).encode()+
                          str(value.dtype).encode()+value.numpy().tobytes()).hexdigest()


def check_image(value):
    if (value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3 or
            value.dtype != torch.float32 or min(value.shape[-2:]) < 2):
        raise ValueError('Expected one float32 BCHW RGB image')
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError('Image must be finite and in [0,1]')


def interpolation_weights(size, grid):
    """align_corners=False双线性插值的一维权重，在CPU构造。"""
    position = ((torch.arange(size, dtype=torch.float32)+.5)*grid/size-.5).clamp(0, grid-1)
    left = position.floor().long()
    right = (left+1).clamp(max=grid-1)
    fraction = position-left.float()
    weights = torch.zeros(size, grid, dtype=torch.float32)
    rows = torch.arange(size)
    weights[rows, left] += 1-fraction
    weights[rows, right] += fraction
    return weights


class PhysicalFit(nn.Module):
    """每次实例化均为共同初态；插值反向用matmul避免CUDA原子归约。"""
    def __init__(self, config, height, width, device='cpu'):
        super().__init__()
        self.config = config
        self.tau_max = -math.log(config.t_min)
        fraction = -math.log(config.init_t)/self.tau_max
        center = math.log(fraction/(1-fraction))
        # 独立CPU随机流；不改变学生/RDDM/全局随机数状态。
        generator = torch.Generator(device='cpu').manual_seed(config.seed)
        a = center+config.init_noise*torch.randn(1, 3, config.grid_size, config.grid_size, generator=generator)
        b = config.init_noise*torch.randn(1, 3, 1, 1, generator=generator)
        self.a = nn.Parameter(a.to(device))
        self.b = nn.Parameter(b.to(device))
        self.register_buffer('height_weights', interpolation_weights(height, config.grid_size).to(device))
        self.register_buffer('width_weights', interpolation_weights(width, config.grid_size).to(device))
        self.initial_hash = self.parameter_hash()

    def parameter_hash(self):
        return hashlib.sha256((tensor_hash(self.a)+tensor_hash(self.b)).encode()).hexdigest()

    def fields(self):
        grid = self.tau_max*self.a.sigmoid()
        tau = (self.height_weights @ grid) @ self.width_weights.T
        return (-tau).exp(), self.b.sigmoid(), grid

    def objective(self, raw, pseudo):
        t, b, grid = self.fields()
        reconstruction = pseudo*t+(1-t)*b
        pixel = (reconstruction-raw).abs().mean()
        tv = ((grid[..., 1:, :]-grid[..., :-1, :]).abs().mean()+
              (grid[..., :, 1:]-grid[..., :, :-1]).abs().mean())/2
        return pixel+self.config.tv_weight*tv, pixel, tv, reconstruction

    def render(self, image, gamma):
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError('gamma must be in [0,1]')
        if image.shape[-2:] != (self.height_weights.shape[0], self.width_weights.shape[0]):
            raise ValueError('Fit/image dimensions differ')
        t, b, _ = self.fields()
        effective = 1-gamma*(1-t)
        return image*effective+(1-effective)*b

    def parameter_payload(self):
        return dict(config=self.config.identity(), a=self.a.detach().cpu().clone(),
                    b=self.b.detach().cpu().clone(), parameter_sha256=self.parameter_hash(),
                    initial_sha256=self.initial_hash, height=self.height_weights.shape[0],
                    width=self.width_weights.shape[0])


def fit_steps(raw, pseudo, config, max_steps):
    """逐步返回post-update状态；新实例/Adam确保逐图完全重置。"""
    check_image(raw); check_image(pseudo)
    if raw.shape != pseudo.shape or raw.device != pseudo.device:
        raise ValueError('raw/pseudo shape and device must match')
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError('max_steps must be a nonnegative integer')
    raw, pseudo = raw.detach(), pseudo.detach()
    model = PhysicalFit(config, *raw.shape[-2:], device=raw.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .999))
    for step in range(max_steps+1):
        gradient_norm = 0.
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = model.objective(raw, pseudo)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite physical fit loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite physical fit gradients')
            gradient_norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters()))
            optimizer.step()
        with torch.no_grad():
            total, pixel, tv, _ = model.objective(raw, pseudo)
            if not torch.isfinite(total):
                raise FloatingPointError('Non-finite post-update objective')
            row = dict(step=step, total=float(total), reconstruction_l1=float(pixel),
                       tv=float(tv), gradient_l2_before_update=gradient_norm)
        yield model, row


def channel_stats(image):
    """统计真实float输入；不将PNG量化边界冒充真实截断比例。"""
    check_image(image)
    values = image.detach().cpu().reshape(3, -1)
    result = {}
    for c, name in enumerate('rgb'):
        x = values[c]
        result.update({name+'_mean': float(x.mean()), name+'_std': float(x.std(unbiased=False)),
                       name+'_zero': float((x == 0).float().mean()),
                       name+'_one': float((x == 1).float().mean()),
                       name+'_near_zero': float((x < 1/255).float().mean()),
                       name+'_near_one': float((x > 254/255).float().mean())})
        for q, tag in [(.01, 'p01'), (.5, 'p50'), (.99, 'p99')]:
            result[name+'_'+tag] = float(torch.quantile(x, q))
    return result
