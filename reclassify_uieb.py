"""Materialize reviewed UIEB labels using byte-for-byte copies, without training."""
import argparse
import csv
import hashlib
import html
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CATEGORIES = ('blue_tint', 'green_tint', 'hazy', 'low_light')
SEVERITIES = ('light', 'moderate', 'severe')
LEVEL_FIELDS = ('blue_level', 'green_level', 'haze_level', 'lowlight_level')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def write_once(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite different metadata: {path}')
        return
    path.write_bytes(data)


def validate_rows(rows, eligibility):
    expected = {v['sample_id'] for v in eligibility['eligible']}
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('Label IDs must exactly cover the unique eligible set')
    for row in rows:
        name = row['sample_id']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError(f'Invalid image filename: {name}')
        if row['primary_category'] not in CATEGORIES:
            raise ValueError(f'Unknown category: {row}')
        levels = [int(row[k]) for k in LEVEL_FIELDS]
        if any(v < 0 or v > 3 for v in levels) or max(levels) == 0:
            raise ValueError(f'Invalid severity levels: {name}')
        if row['severity'] != SEVERITIES[max(levels) - 1]:
            raise ValueError(f'Severity disagrees with component levels: {name}')
        if levels[CATEGORIES.index(row['primary_category'])] != max(levels):
            raise ValueError(f'Primary category is not a maximum impairment: {name}')
        for role in ('raw', 'reference'):
            relative = f'{role}/{row["severity"]}/{row["primary_category"]}/{name}'
            if row[f'{role}_relative'] != relative:
                raise ValueError(f'Unexpected destination path: {name}')
    groups = defaultdict(set)
    for row in rows:
        if row.get('scene_group_id'):
            groups[row['scene_group_id']].add(row['severity'])
    if any('light' in values and 'severe' in values for values in groups.values()):
        raise ValueError('An identified near-duplicate group crosses light/severe')


def review_html(rows):
    cards = []
    names = dict(blue_tint='偏蓝', green_tint='偏绿', hazy='雾化', low_light='低照度',
                 light='轻度', moderate='中度（封存）', severe='重度')
    for r in rows:
        path = html.escape(r['raw_relative'], quote=True)
        name = html.escape(r['sample_id'])
        tags = html.escape(r['secondary_categories'] or '无其他明显退化')
        note = html.escape(r['review_note'])
        levels = ' / '.join(r[k] for k in LEVEL_FIELDS)
        cards.append(f'<article data-s="{r["severity"]}" data-c="{r["primary_category"]}">'
                     f'<a href="{path}" target="_blank"><img loading="lazy" src="{path}" alt="{name}"></a>'
                     f'<strong>{name}</strong><p>{names[r["severity"]]} / {names[r["primary_category"]]}</p>'
                     f'<p>蓝 / 绿 / 雾 / 暗：{levels}</p><p>次要：{tags}</p><p>{note}</p></article>')
    return ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>UIEB v2 划分审阅</title>
<style>body{font:15px system-ui;margin:24px;background:#f3f5f7;color:#182b40}header{position:sticky;top:0;background:#f3f5f7;padding:8px 0}select{padding:8px;margin:8px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}article{background:white;padding:12px;border-radius:6px}img{width:100%;height:190px;object-fit:contain;background:#e9edf0}p{margin:6px 0;font-size:13px}article[data-s=severe]{border-top:4px solid #c65338}article[data-s=light]{border-top:4px solid #41825c}article[data-s=moderate]{border-top:4px solid #b68c35}</style>
<header><h2>UIEB v2 原图与划分</h2><p>AI视觉研究初标，非官方标签。显示尺寸仅供浏览，点击可查看原始文件。未使用reference或源模型PSNR分类。中度封存。</p>
<select id="s"><option value="">全部程度</option><option value="light">轻度</option><option value="moderate">中度</option><option value="severe">重度</option></select>
<select id="c"><option value="">全部主类别</option><option value="blue_tint">偏蓝</option><option value="green_tint">偏绿</option><option value="hazy">雾化</option><option value="low_light">低照度</option></select><span id="count"></span></header><main>'''
            + ''.join(cards) + '''</main><script>function filter(){let n=0;document.querySelectorAll('article').forEach(e=>{const ok=(!s.value||e.dataset.s===s.value)&&(!c.value||e.dataset.c===c.value);e.hidden=!ok;if(ok)n++});document.getElementById('count').textContent=n+' 张'}const s=document.getElementById('s'),c=document.getElementById('c');s.onchange=c.onchange=filter;filter()</script></html>''').encode('utf-8')


def materialize(source, labels, eligibility_path, output, *, verify_only=False):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output and original dataset directories must be separate')
    with labels.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    eligibility = json.loads(eligibility_path.read_text(encoding='utf-8'))
    validate_rows(rows, eligibility)
    expected = {str(Path(r[f'{role}_relative'])) for r in rows for role in ('raw', 'reference')}
    if output.exists():
        observed = {str(p.relative_to(output)) for role in ('raw', 'reference')
                    for p in (output / role).rglob('*') if p.is_file()}
        if observed - expected:
            raise ValueError(f'Unexpected pre-existing images: {sorted(observed - expected)}')
        if verify_only and observed != expected:
            raise ValueError('Incomplete copied image set')
    elif verify_only:
        raise ValueError('Output does not exist')
    if not verify_only:
        parent = output
        while not parent.exists():
            parent = parent.parent
        remaining = sum((source / f'{role}-890' / row['sample_id']).stat().st_size
                        for row in rows for role in ('raw', 'reference')
                        if not (output / row[f'{role}_relative']).exists())
        if shutil.disk_usage(parent).free < remaining + 64 * 1024 * 1024:
            raise ValueError('Insufficient storage for exact image copies and metadata')
    # Verify every source and existing destination before creating any image copies.
    for row in rows:
        for role in ('raw', 'reference'):
            src = source / f'{role}-890' / row['sample_id']
            if sha256(src) != row[f'{role}_sha256']:
                raise ValueError(f'Source file changed since review: {src}')
            with Image.open(src) as im:
                if im.size != (int(row['width']), int(row['height'])):
                    raise ValueError(f'Image dimensions changed: {src}')
            dst = output / row[f'{role}_relative']
            if dst.exists() and sha256(dst) != row[f'{role}_sha256']:
                raise ValueError(f'Refusing to overwrite different image: {dst}')
    if not verify_only:
        for role in ('raw', 'reference'):
            for severity in SEVERITIES:
                for category in CATEGORIES:
                    (output / role / severity / category).mkdir(parents=True, exist_ok=True)
    image_bytes = 0
    for index, row in enumerate(rows):
        for role in ('raw', 'reference'):
            src = source / f'{role}-890' / row['sample_id']
            dst = output / row[f'{role}_relative']
            if not dst.exists():
                if verify_only:
                    raise ValueError(f'Missing copied image: {dst}')
                shutil.copy2(src, dst)
            if sha256(dst) != row[f'{role}_sha256']:
                raise ValueError(f'Byte-for-byte verification failed: {dst}')
            image_bytes += dst.stat().st_size
        if (index + 1) % 100 == 0:
            print(f'{"Verified" if verify_only else "Copied and verified"} {index + 1}/{len(rows)} pairs', flush=True)
    counts = {severity: {category: sum(r['severity'] == severity and r['primary_category'] == category
                                     for r in rows) for category in CATEGORIES} for severity in SEVERITIES}
    manifest = dict(version='UIEB_reclassified_v2', date='2026-10-15', status='complete',
                    pair_count=len(rows), image_files=2 * len(rows), image_bytes=image_bytes,
                    original_pair_count=eligibility['original_count'],
                    excluded_count=eligibility['excluded_count'],
                    labels_sha256=sha256(labels), eligibility_sha256=sha256(eligibility_path),
                    implementation_sha256=sha256(Path(__file__)),
                    category_counts=counts,
                    severity_counts=dict(Counter(r['severity'] for r in rows)),
                    copy_method='shutil.copy2: no decode/encode/resize/rename of destination images',
                    verified='all source file hashes, destination file hashes and source dimensions',
                    label_basis='raw-only AI visual research annotation, single annotator, not official physical ground truth',
                    middle_policy='sealed: no training, adaptation or tuning',
                    scene_audit='Exact RGB duplicates removed. Three visually confirmed pHash candidate groups recorded. No exhaustive scene-disjointness claim.',
                    source_model_psnr_used_for_labels=False,
                    references_used_for_labels=False, previous_labels_used=False,
                    samples=rows)
    if not verify_only:
        write_once(output / 'labels.csv', labels.read_bytes())
        write_once(output / 'manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
        exclusions = [dict(sample_id=r['sample_id'], reason=r['reason'],
                           group=';'.join(r['group']), retained_sample_id=r.get('retained_sample_id', ''))
                      for r in eligibility['excluded']]
        write_once(output / 'exclusions.csv', csv_bytes(exclusions, ['sample_id', 'reason', 'group', 'retained_sample_id']))
        write_once(output / 'review.html', review_html(rows))
    else:
        saved = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if saved != manifest or (output / 'labels.csv').read_bytes() != labels.read_bytes():
            raise ValueError('Saved metadata does not match verified inputs')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'samples'}, ensure_ascii=False, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--eligibility', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    materialize(args.source, args.labels, args.eligibility, args.output, verify_only=args.verify_only)
