"""源域 NU2Net-L1：数据清单、场景检查、训练、完整图片验证和轮次级续训。

原图始终只读；目标 reference 仅在准备阶段核对配对，不进入源训练和模型选择。
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF, InterpolationMode
from tqdm import tqdm

from options import parse_source_args
from uie_tta import (ImageRestorationAdapter, EvaluationDataset, select_samples,
                     sha256_file, seed_everything, resolve_device, load_checkpoint)
from uie_backbones.nu2net import NU2Net
from utils.val_utils import compute_psnr_ssim


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def image_info(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        size = rgb.size
        pixels = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
        small = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        dhash = int(''.join('1' if bit else '0' for bit in bits), 2)
    return size, pixels, dhash


def scan_pairs(raw_dir, ref_dir, domain, expected):
    """完整文件名优先；扩展名不同只允许唯一 stem 匹配，绝不按排序位置配对。"""
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    def images(directory):
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f'Directory not found (check exact folder hierarchy): {directory}')
        if 'score_sorted' in directory.parts:
            raise ValueError('Use original filenames, not score_sorted copies')
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    raws, refs = images(raw_dir), images(ref_dir)
    if len(raws) != expected or len(refs) != expected:
        raise ValueError(f'{domain}: expected {expected} raw/ref pairs; found {len(raws)}/{len(refs)}')
    rows, used, names = [], set(), set()
    for raw in raws:
        candidates = [p for p in refs if p.name.casefold() == raw.name.casefold()]
        if not candidates:
            candidates = [p for p in refs if p.stem.casefold() == raw.stem.casefold()]
        if len(candidates) != 1 or candidates[0] in used or raw.stem.casefold() in names:
            raise ValueError(f'Ambiguous/reused image pair: {raw}')
        ref = candidates[0]
        used.add(ref)
        names.add(raw.stem.casefold())
        size, pixel_hash, dhash = image_info(raw)
        ref_size, ref_hash, _ = image_info(ref)
        if size != ref_size or min(size) < 16:
            raise ValueError(f'Invalid raw/reference dimensions: {raw}: {size}/{ref_size}')
        rows.append(dict(domain=domain, sample_id=raw.name, scene_id=f'{domain}:{raw.name}',
                         raw_path=str(raw), reference_path=str(ref), width=size[0], height=size[1],
                         raw_sha256=sha256_file(raw), reference_sha256=sha256_file(ref),
                         pixel_hash=pixel_hash, ref_pixel_hash=ref_hash, dhash=dhash))
    return rows


def prepare_manifests(opt):
    folder = Path(opt.experiment_dir).resolve() / 'manifests'
    # 已固定的清单不可被再次随机划分覆盖。另一次实验应使用新 experiment_dir。
    if (folder / 'audit.json').exists():
        raise FileExistsError(f'Manifests already finalized: {folder}; use them or select a new experiment_dir')
    source = scan_pairs(opt.source_raw, opt.source_ref, 'source', opt.expected_source)
    target = scan_pairs(opt.target_raw, opt.target_ref, 'target', opt.expected_target)
    folder.mkdir(parents=True, exist_ok=True)
    # 2026-09-08 用户确认排除两份“同一 raw、不同 ref”的原始样本；只改清单，不删原图。
    # 绑定已核查的 RGB 像素哈希，避免将来同名的其他内容被意外排除。
    approved = {
        '582_img_.png': '29bbf9bf4ed818cb838365e92e9c460a0229c9ed1bbf4f97da586a1368d85cf9',
        '648_img_.png': 'd0d6655aeaa2327dd0c3638084c06f98069cbdc1c2d96f72da923dbe4d87dc2a',
    }
    raw_hash = '8ca135dcaec7893d54b4c5b9bc17ff605780aba50f62ed2bb6fcb525fed2f7a3'
    exclusions = []
    requested = set(opt.exclude_source_ids)
    if requested - approved.keys():
        raise ValueError('Unknown source exclusion; review and explicitly record its identity before excluding')
    for row in source:
        if row['sample_id'] in requested:
            if row['pixel_hash'] != raw_hash or row['ref_pixel_hash'] != approved[row['sample_id']]:
                raise ValueError(f'Approved exclusion content mismatch: {row["sample_id"]}; inspect before excluding')
            exclusions.append(dict(sample_id=row['sample_id'], raw_path=row['raw_path'],
                                   reference_path=row['reference_path'], raw_pixel_hash=row['pixel_hash'],
                                   reference_pixel_hash=row['ref_pixel_hash'],
                                   reason='User approved 2026-09-08: identical raw with conflicting original references'))
    write_json(folder / 'source_exclusions.json', dict(excluded=exclusions,
               requested_but_absent=sorted(requested - {r['sample_id'] for r in source}),
               input_count=len(source), retained_count=len(source) - len(exclusions), originals_modified=False))
    source = [r for r in source if r['sample_id'] not in requested]
    rows = source + target
    keys = {(r['domain'], r['sample_id']): r for r in rows}
    if opt.scene_groups:
        seen = set()
        for group in read_csv(opt.scene_groups):
            key = (group['domain'], group['sample_id'])
            if key not in keys or key in seen or not group['scene_id'].strip():
                raise ValueError(f'Invalid scene group: {group}')
            seen.add(key)
            keys[key]['scene_id'] = 'manual:' + group['scene_id'].strip()
    # 先完整列出精确 raw 重复及 reference 冲突，避免只报一句错误而无法定位。
    # 像素哈希在统一 RGB 解码后计算，不把文件名、压缩格式或元信息当作图像内容。
    duplicate_checks = []
    representatives = {}
    for row in rows:
        match = representatives.setdefault(row['pixel_hash'], row)
        if match is row:
            continue
        conflict = row['ref_pixel_hash'] != match['ref_pixel_hash']
        issue = ('source_target_leakage' if row['domain'] != match['domain'] else
                 'duplicate_target' if row['domain'] == 'target' else
                 'conflicting_references' if conflict else 'identical_source_pair')
        record = dict(issue=issue, a_domain=match['domain'], a_id=match['sample_id'],
                      a_raw=match['raw_path'], a_reference=match['reference_path'],
                      b_domain=row['domain'], b_id=row['sample_id'], b_raw=row['raw_path'],
                      b_reference=row['reference_path'], raw_pixel_hash=row['pixel_hash'],
                      a_reference_pixel_hash=match['ref_pixel_hash'],
                      b_reference_pixel_hash=row['ref_pixel_hash'],
                      reference_mae_0_255=0.0, reference_max_diff=0)
        if conflict:
            with Image.open(match['reference_path']) as image:
                a = np.asarray(image.convert('RGB'), dtype=np.int16)
            with Image.open(row['reference_path']) as image:
                b = np.asarray(image.convert('RGB'), dtype=np.int16)
            difference = np.abs(a - b)
            record.update(reference_mae_0_255=float(difference.mean()),
                          reference_max_diff=int(difference.max()))
        duplicate_checks.append(record)
    report_path = folder / 'exact_duplicate_report.json'
    write_json(report_path, dict(pairs=duplicate_checks, approved_source_exclusions=exclusions,
               note='RGB pixel comparison. Different references do not alone prove incorrect pairing; inspect originals.'))
    blocking = [r for r in duplicate_checks if r['issue'] != 'identical_source_pair']
    if blocking:
        details = '\n'.join(f'{r["issue"]}: {r["a_id"]} <-> {r["b_id"]}\n'
                            f'  reference A: {r["a_reference"]}\n'
                            f'  reference B: {r["b_reference"]}' for r in blocking)
        raise ValueError(f'Exact duplicate/reference conflict or leakage detected:\n{details}\n'
                         f'Full report: {report_path}\nNo final manifests written; inspect these pairs first.')
    duplicates = []
    kept = []
    for row in rows:
        match = next((r for r in kept if r['pixel_hash'] == row['pixel_hash']), None)
        if match:
            if row['domain'] != match['domain'] or row['domain'] == 'target':
                raise ValueError(f'Duplicate target or source/target leakage: {row["raw_path"]}')
            if row['ref_pixel_hash'] != match['ref_pixel_hash']:
                raise ValueError('Identical raw has conflicting references')
            duplicates.append(dict(excluded=row['raw_path'], retained=match['raw_path']))
        else:
            kept.append(row)
    source = [r for r in kept if r['domain'] == 'source']
    target = [r for r in kept if r['domain'] == 'target']
    for a in source:
        for b in target:
            if (Path(a['raw_path']).stem.casefold() == Path(b['raw_path']).stem.casefold()
                    or a['ref_pixel_hash'] == b['ref_pixel_hash'] or a['scene_id'] == b['scene_id']):
                raise ValueError(f'Source/target overlap: {a["sample_id"]}, {b["sample_id"]}')
    # dHash 只筛候选，不能自动认定相同场景；人工写 same/different 后再准备。
    review_file = folder / 'near_duplicate_review.csv'
    previous = {}
    if review_file.exists():
        for r in read_csv(review_file):
            previous[(r['a_domain'], r['a_id'], r['b_domain'], r['b_id'], r['a_hash'], r['b_hash'])] = r['decision']
    parent = {id(r): id(r) for r in kept}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    def union(a, b):
        parent[find(id(b))] = find(id(a))
    candidates, pending = [], False
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            same_manual = a['scene_id'] == b['scene_id']
            if same_manual:
                union(a, b)
            distance = bin(a['dhash'] ^ b['dhash']).count('1')
            if distance > 4:
                continue
            key = (a['domain'], a['sample_id'], b['domain'], b['sample_id'], a['pixel_hash'], b['pixel_hash'])
            decision = 'same' if same_manual else previous.get(key, '')
            if decision not in {'', 'same', 'different'}:
                raise ValueError('Review decision must be same or different')
            candidates.append(dict(zip(('a_domain', 'a_id', 'b_domain', 'b_id', 'a_hash', 'b_hash'), key),
                                   distance=distance, decision=decision))
            pending |= not decision
            if decision == 'same':
                if a['domain'] != b['domain']:
                    raise ValueError('Confirmed same-scene source/target overlap; correct the dataset before splitting')
                union(a, b)
    fields = ['a_domain', 'a_id', 'b_domain', 'b_id', 'a_hash', 'b_hash', 'distance', 'decision']
    write_csv(review_file, candidates, fields)
    write_csv(folder / 'image_inventory.csv', rows,
              ['domain', 'sample_id', 'raw_path', 'reference_path', 'width', 'height', 'raw_sha256', 'reference_sha256'])
    statistics = {domain: {'count': len(items), 'min_short_side': min(min(r['width'], r['height']) for r in items),
                   'below_patch': {str(p): sum(min(r['width'], r['height']) < p for r in items) for p in (256, 320, 384, 512)}}
                  for domain, items in [('source', source), ('target', target)]}
    write_json(folder / 'preparation_report.json', dict(sizes=statistics, approved_source_exclusions=exclusions,
               duplicates_excluded_from_manifest=duplicates,
               near_candidates=len(candidates), pending_review=pending,
               note='dHash candidates are not exhaustive; supply --scene_groups for known related scenes'))
    if pending:
        raise ValueError(f'Review candidate pairs in {review_file}: fill decision=same/different and rerun --prepare_only')
    groups = {}
    for r in source:
        groups.setdefault(find(id(r)), []).append(r)
    group_list = list(groups.values())
    random.Random(opt.seed).shuffle(group_list)
    # 子集和选择尽量接近指定验证数量；整个场景分组不可拆散。
    possible = {0: []}
    for index, group in enumerate(group_list):
        for count, chosen in list(possible.items()):
            possible.setdefault(count + len(group), chosen + [index])
    counts = [n for n in possible if 0 < n < len(source)]
    if not counts:
        raise ValueError('Not enough independent scenes for source train/validation')
    count = min(counts, key=lambda n: (abs(n - opt.val_count), n))
    selected = set(possible[count])
    train, val = [], []
    for i, group in enumerate(group_list):
        for r in group:
            r['scene_id'] = f'source_group_{i:04d}'
        (val if i in selected else train).extend(group)
    manifest_fields = ['sample_id', 'raw_path', 'reference_path', 'scene_id', 'raw_sha256', 'reference_sha256']
    for name, items in [('source_train', train), ('source_val', val), ('target_eval', target)]:
        write_csv(folder / f'{name}.csv', items, manifest_fields)
    audit = dict(seed=opt.seed, approved_source_exclusions=exclusions,
                 counts=dict(source_train=len(train), source_val=len(val), target_eval=len(target)),
                 sizes=statistics, files={name: sha256_file(folder / name) for name in
                 ('source_train.csv', 'source_val.csv', 'target_eval.csv', 'near_duplicate_review.csv', 'source_exclusions.json')})
    write_json(folder / 'audit.json', audit)
    print(json.dumps(audit['counts'], ensure_ascii=False))
    return audit


def checked_manifest(folder, name, audit):
    path = Path(folder) / name
    if sha256_file(path) != audit['files'][name]:
        raise ValueError(f'Manifest changed after preparation: {path}')
    rows = read_csv(path)
    for row in rows:
        for field in ('raw', 'reference'):
            if sha256_file(row[field + '_path']) != row[field + '_sha256']:
                raise ValueError(f'Image changed after preparation: {row[field + "_path"]}')
    return select_samples(manifest=path)


class SourcePairs(Dataset):
    """按原 NU2Net 的 PIL bicubic RandomResizedCrop + 同步 H/V flip。"""
    def __init__(self, samples, patch_size):
        self.samples, self.patch_size = samples, patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.raw_path) as image:
            raw = image.convert('RGB')
        with Image.open(sample.reference_path) as image:
            ref = image.convert('RGB')
        i, j, h, w = transforms.RandomResizedCrop.get_params(raw, (0.08, 1.0), (3 / 4, 4 / 3))
        raw, ref = [TF.to_tensor(TF.resized_crop(im, i, j, h, w, [self.patch_size] * 2,
                    InterpolationMode.BICUBIC)) for im in (raw, ref)]
        if np.random.rand() < 0.5:
            raw, ref = raw.flip([2]), ref.flip([2])
        if np.random.rand() < 0.5:
            raw, ref = raw.flip([1]), ref.flip([1])
        return raw, ref


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def validate(model, samples, device):
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    for batch in EvaluationDataset(samples, 'pad8'):
        raw = batch['raw'].unsqueeze(0).to(device)
        ref = batch['reference'].unsqueeze(0).to(device)
        pred = model(raw)
        if pred.shape != ref.shape or not torch.isfinite(pred).all():
            raise FloatingPointError('Invalid source validation prediction')
        psnr, ssim = compute_psnr_ssim(pred, ref)
        totals += [torch.nn.functional.l1_loss(pred, ref).item(), float(psnr), float(ssim)]
    return dict(zip(('val_l1', 'val_psnr', 'val_ssim'), (totals / len(samples)).tolist()))


def random_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_random_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, payload):
    # 原子替换单个明确路径：中断保存不会破坏上一完整轮次的 checkpoint。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_curves(folder, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for filename, keys in [('l1', ['train_l1', 'val_l1']), ('psnr', ['val_psnr']),
                           ('ssim', ['val_ssim']), ('learning_rate', ['lr'])]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for key in keys:
            ax.plot([r['epoch'] for r in history], [r[key] for r in history], label=key,
                    marker='.', markersize=5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(filename)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / (filename + '.png'), dpi=140)
        plt.close(fig)


def train_source(opt):
    import matplotlib  # 在长训练前检查曲线依赖，避免训练结束才失败。
    folder = Path(opt.experiment_dir).resolve()
    manifests = folder / 'manifests'
    audit = json.loads((manifests / 'audit.json').read_text(encoding='utf-8'))
    train_samples = checked_manifest(manifests, 'source_train.csv', audit)
    val_samples = checked_manifest(manifests, 'source_val.csv', audit)
    # 源训练不读取目标 raw/ref；目标数据在独立准备和评价流程核验。
    source_dir = folder / 'source'
    if not opt.resume and any((source_dir / 'checkpoints' / name).exists() for name in ('last.pth', 'best.pth')):
        raise FileExistsError('Source checkpoints exist; use --resume or a new experiment_dir')
    seed_everything(opt.seed)
    device = resolve_device(opt.device)
    config = dict(basic_channel=opt.basic_channel, is_residual=True, tail='norm')
    model = ImageRestorationAdapter(NU2Net(**config), 'nu2net', config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    steps_per_epoch = math.ceil(len(train_samples) / opt.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs * steps_per_epoch)
    # 保留最后不足 batch 的样本。每轮 generator 独立定种子，使 worker 的增强可在轮次边界恢复。
    contract = dict(epochs=opt.epochs, batch_size=opt.batch_size, num_workers=opt.num_workers,
                    patch_size=opt.patch_size, lr=opt.lr, seed=opt.seed, model_config=config,
                    steps_per_epoch=steps_per_epoch, device_type=device.type,
                    torch_version=str(torch.__version__),
                    data={n: audit['files'][n] for n in ('source_train.csv', 'source_val.csv')},
                    implementation={p.name: sha256_file(p) for p in
                                    (Path(__file__), Path(__file__).with_name('uie_tta.py'),
                                     Path(__file__).with_name('uie_backbones') / 'nu2net.py')})
    start, best, best_epoch, history = 0, -float('inf'), 0, []
    if opt.resume:
        payload = load_checkpoint(opt.resume)
        if payload.get('training_contract') != contract or payload.get('artifact_type') != 'source_training':
            raise ValueError('Resume configuration/data/code differs from the saved source training run')
        model.backbone.load_state_dict(payload['net'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        start, best, best_epoch = payload['epoch'], payload['best_source_psnr'], payload['best_epoch']
        history = payload['history']
        restore_random_state(payload['rng'])
        # last 先提交、best 后提交。如果恰好在两次提交间中断，当前 last 可重建该轮 best。
        best_path = source_dir / 'checkpoints' / 'best.pth'
        existing_best = load_checkpoint(best_path) if best_path.exists() else None
        if start == best_epoch and (existing_best is None or existing_best['epoch'] < best_epoch):
            save_checkpoint(best_path, payload)
            existing_best = payload
        if existing_best is None:
            raise FileNotFoundError('Keep best.pth alongside last.pth when resuming')
        if existing_best['training_contract'] != contract or existing_best['epoch'] != best_epoch:
            raise ValueError('best.pth does not match the resume checkpoint')
    write_json(source_dir / 'logs' / 'config.json', dict(arguments=vars(opt), contract=contract,
               source_train_ids=[s.sample_id for s in train_samples], source_val_ids=[s.sample_id for s in val_samples]))
    dataset = SourcePairs(train_samples, opt.patch_size)
    end = opt.stop_after_epoch or opt.epochs
    if start > end:
        raise ValueError('Requested stop epoch precedes resume checkpoint')
    if history:
        state = random_state()
        write_csv(source_dir / 'logs' / 'epochs.csv', history, list(history[0]))
        save_curves(source_dir / 'curves', history)
        restore_random_state(state)
    for epoch in range(start, end):
        tick = time.perf_counter()
        generator = torch.Generator().manual_seed(opt.seed + epoch)
        loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False,
                            num_workers=opt.num_workers, worker_init_fn=seed_worker, generator=generator,
                            pin_memory=device.type == 'cuda')
        model.train()
        total, count = 0.0, 0
        for raw, ref in tqdm(loader, desc=f'Source {epoch + 1}/{opt.epochs}'):
            raw, ref = raw.to(device), ref.to(device)
            pred = model(raw)
            loss = torch.nn.functional.l1_loss(pred, ref)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite source L1 loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite source gradients')
            optimizer.step()
            scheduler.step()
            total += loss.item() * raw.shape[0]
            count += raw.shape[0]
        metrics = validate(model, val_samples, device)
        if not all(np.isfinite(metrics[k]) for k in ('val_l1', 'val_ssim')) or math.isnan(metrics['val_psnr']):
            raise FloatingPointError('Invalid validation metrics')
        improved = metrics['val_psnr'] > best
        if improved:
            best, best_epoch = metrics['val_psnr'], epoch + 1
        history.append(dict(epoch=epoch + 1, train_l1=total / count, **metrics,
                            lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter() - tick,
                            best_epoch=best_epoch, best_psnr=best))
        payload = dict(net={k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
                       model_config=config, artifact_type='source_training', epoch=epoch + 1,
                       global_step=(epoch + 1) * steps_per_epoch, optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), best_source_psnr=best, best_epoch=best_epoch,
                       history=history, rng=random_state(), training_contract=contract,
                       source_train_ids=[s.sample_id for s in train_samples],
                       source_val_ids=[s.sample_id for s in val_samples])
        save_checkpoint(source_dir / 'checkpoints' / 'last.pth', payload)
        if improved:
            save_checkpoint(source_dir / 'checkpoints' / 'best.pth', payload)
        # 曲线绘制不应影响下轮裁剪随机状态。
        state = random_state()
        write_csv(source_dir / 'logs' / 'epochs.csv', history, list(history[0]))
        save_curves(source_dir / 'curves', history)
        restore_random_state(state)
        print(json.dumps(history[-1], ensure_ascii=False))
    return history


def main(argv=None):
    opt = parse_source_args(argv)
    return prepare_manifests(opt) if opt.prepare_only else train_source(opt)


if __name__ == '__main__':
    main()
