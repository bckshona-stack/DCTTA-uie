"""Prepare an image-only review inventory; never alter or relabel source files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(source, eligibility, output):
    output.mkdir(parents=True, exist_ok=True)
    review = json.loads(eligibility.read_text(encoding='utf-8'))
    records = []
    for index, item in enumerate(review['eligible']):
        name = item['sample_id']
        with Image.open(source / 'raw-890' / name) as im:
            rgb = im.convert('RGB')
            width, height = rgb.size
            small = rgb.copy()
            small.thumbnail((256, 256), Image.Resampling.LANCZOS)
            a = np.asarray(small, dtype=np.float64) / 255
            y = a @ np.array([.2126, .7152, .0722])
            gray = np.asarray(rgb.convert('L').resize((32, 32)), dtype=float)
            dct_basis = np.cos(np.pi * (np.arange(32) + .5)[None, :] * np.arange(8)[:, None] / 32)
            dct_basis[0] *= 1 / np.sqrt(2)
            coeff = (dct_basis @ gray @ dct_basis.T).ravel()[1:]
            phash = sum(int(v > np.median(coeff)) << k for k, v in enumerate(coeff))
            means = a.mean(axis=(0, 1))
            row = dict(index=index, sample_id=name, width=width, height=height,
                       mean_r=float(means[0]), mean_g=float(means[1]), mean_b=float(means[2]),
                       luma_mean=float(y.mean()), dark_fraction=float((y < .12).mean()),
                       luma_p10=float(np.quantile(y, .1)), luma_p90=float(np.quantile(y, .9)),
                       luma_std=float(y.std()), phash=f'{phash:016x}',
                       raw_sha256=file_sha(source / 'raw-890' / name),
                       reference_sha256=file_sha(source / 'reference-890' / name))
            records.append(row)
        if (index + 1) % 100 == 0:
            print(f'Prepared {index + 1}/{len(review["eligible"])}', flush=True)
    with (output / 'raw_features.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / 'inventory.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    candidates = []
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            distance = bin(int(a['phash'], 16) ^ int(b['phash'], 16)).count('1')
            if distance <= 6:
                candidates.append(dict(first=a['sample_id'], second=b['sample_id'], distance=distance))
    (output / 'near_duplicate_candidates.json').write_text(json.dumps(candidates, indent=2), encoding='utf-8')
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    pages = output / 'review_pages'
    pages.mkdir(exist_ok=True)
    for start in range(0, len(records), 30):
        canvas = Image.new('RGB', (1300, 1128), '#f1f3f5')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 3), f'RAW ONLY / page {start // 30 + 1:02d} / no color correction', fill='black', font=font)
        for slot, item in enumerate(records[start:start + 30]):
            x, y = (slot % 5) * 260, 28 + (slot // 5) * 182
            with Image.open(source / 'raw-890' / item['sample_id']) as im:
                thumb = im.convert('RGB')
                thumb.thumbnail((254, 151), Image.Resampling.LANCZOS)
                canvas.paste(thumb, (x + (260 - thumb.width) // 2, y + (151 - thumb.height) // 2))
            draw.text((x + 5, y + 153), f'{item["index"]:03d} {item["sample_id"]}', fill='black', font=font)
        canvas.save(pages / f'{start // 30 + 1:02d}.jpg', quality=95, subsampling=0)
    print(json.dumps(dict(count=len(records), pages=(len(records) + 29) // 30,
                          near_duplicate_candidates=len(candidates)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--eligibility', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    prepare(args.source, args.eligibility, args.output)
