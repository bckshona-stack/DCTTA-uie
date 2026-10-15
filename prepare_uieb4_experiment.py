"""新 UIEB 四类轻→重协议；只生成清单，不改动图像文件。"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / 'docs/10_20261015_dataset_reclassification'
CATEGORIES = ('blue_tint', 'green_tint', 'hazy', 'low_light')
VAL_COUNTS = dict(zip(CATEGORIES, (35, 14, 10, 3)))
LABEL_SHA = 'c01063c647365f29128157dea792a17152c366ed41f96bb03824674f3f2b8003'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def verify_runtime():
    path=ROOT/'docs/11_20261015_uieb4_source_g0/RUNTIME_FILES.json'
    expected=json.loads(path.read_text(encoding='utf-8'))
    for name,digest in expected.items():
        file=(ROOT/name).resolve()
        actual=hashlib.sha256(file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if ROOT not in file.parents or actual!=digest:
            raise ValueError('Runtime file missing/changed; synchronize listed files: '+name)


def immutable(path, blob):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != blob:
            raise ValueError('Prepared protocol differs; use another experiment directory: '+str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')


def csv_bytes(rows):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    return s.getvalue().encode('utf-8-sig')


def stratify(rows, val_counts, seed):
    """按类别做组级子集和，保证已审核同场景不跨训练/验证。"""
    groups = {}
    for row in rows:
        groups.setdefault(row['scene_id'], []).append(row)
    if any(len({r['primary_category'] for r in g}) != 1 for g in groups.values()):
        raise ValueError('Scene group crosses categories; review required')
    selected = set()
    for category, count in val_counts.items():
        candidates = [g for g in groups.values() if g[0]['primary_category'] == category]
        candidates.sort(key=lambda g: hashlib.sha256((str(seed)+':'+g[0]['scene_id']).encode()).hexdigest())
        dp = {0: ()}
        for group in candidates:
            size = len(group)
            for total in sorted(list(dp), reverse=True):
                if total+size <= count and total+size not in dp:
                    dp[total+size] = dp[total]+tuple(r['sample_id'] for r in group)
        if count not in dp:
            raise ValueError('Cannot meet validation count without breaking scene groups: '+category)
        selected.update(dp[count])
    return ([r for r in rows if r['sample_id'] not in selected],
            [r for r in rows if r['sample_id'] in selected])


def prepare(dataset_dir, experiment_dir, seed=2022):
    dataset = Path(dataset_dir).resolve()
    labels = dataset/'labels.csv'
    if (sha(labels) != LABEL_SHA or labels.read_text(encoding='utf-8-sig') !=
            (REVIEW/'UIEB_V2_LABELS.csv').read_text(encoding='utf-8-sig')):
        raise ValueError('Use the approved UIEB v2 labels without modifications')
    near_path = REVIEW/'UIEB_V2_NEAR_DUPLICATE_REVIEW.json'
    near = json.loads(near_path.read_text(encoding='utf-8'))['groups']
    rows = read_csv(labels)
    if len(rows) != 879 or len({r['sample_id'].casefold() for r in rows}) != 879:
        raise ValueError('Expected 879 unique retained pairs')
    ready = []
    for r in rows:
        if r['primary_category'] not in CATEGORIES or r['severity'] not in ('light','moderate','severe'):
            raise ValueError('Invalid category/severity')
        item = dict(sample_id=r['sample_id'], scene_id=near.get(r['sample_id']) or r['scene_group_id'] or r['sample_id'],
                    primary_category=r['primary_category'], severity=r['severity'],
                    width=int(r['width']), height=int(r['height']))
        for role in ('raw', 'reference'):
            path = (dataset/r[role+'_relative']).resolve()
            if dataset not in path.parents or path.name != r['sample_id'] or sha(path) != r[role+'_sha256']:
                raise ValueError('Image path/hash differs: '+str(path))
            item[role+'_path'] = str(path)
            item[role+'_sha256'] = r[role+'_sha256']
        ready.append(item)
    light = [r for r in ready if r['severity']=='light']
    target = [r for r in ready if r['severity']=='severe']
    if (len(light),len(target),len(ready)-len(light)-len(target)) != (306,172,401):
        raise ValueError('Severity counts differ')
    train, val = stratify(light, VAL_COUNTS, seed)
    splits = dict(source_train=train,source_val=val,target_eval=target)
    scenes = [{r['scene_id'] for r in split} for split in splits.values()]
    hashes = [{r['raw_sha256'] for r in split} for split in splits.values()]
    for i in range(3):
        for j in range(i):
            if scenes[i]&scenes[j] or hashes[i]&hashes[j]:
                raise ValueError('Scene/exact-raw leakage between splits')
    folder = Path(experiment_dir).resolve()/'manifests'
    blobs = {name+'.csv':csv_bytes(split) for name,split in splits.items()}
    audit = dict(protocol='UIEB4_light_to_severe_v1', seed=seed,
        counts={k:len(v) for k,v in splits.items()}, sealed_moderate=401,
        category_counts={k:{c:sum(r['primary_category']==c for r in v) for c in CATEGORIES} for k,v in splits.items()},
        labels_sha256=LABEL_SHA, near_review_sha256=sha(near_path),
        exhaustive_scene_audit=False, files={k:hashlib.sha256(v).hexdigest() for k,v in blobs.items()},
        preparation_sha256=sha(__file__), dataset_dir=str(dataset))
    blobs['audit.json'] = json_bytes(audit)
    # 先检查全部已有文件，避免发现协议冲突前写入部分新文件。
    for name,blob in blobs.items():
        path=folder/name
        if path.exists() and path.read_bytes()!=blob:
            raise ValueError('Existing manifest differs: '+str(path))
    for name,blob in blobs.items(): immutable(folder/name,blob)
    print('Prepared: source train 244 / val 62; target 172; moderate 401 sealed',flush=True)
    return audit


def main():
    verify_runtime()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_dir',default=str(ROOT/'dataset/UIEB_reclassified_v2'))
    p.add_argument('--experiment_dir',default=str(ROOT/'experiments/UIEB4_light_to_severe_NU2Net_v1'))
    p.add_argument('--phase',choices=('prepare','train'),required=True)
    args,remaining=p.parse_known_args()
    if args.phase=='prepare':
        if remaining: p.error('Training options are only accepted in train phase')
        prepare(args.dataset_dir,args.experiment_dir)
    else:
        from options import parse_source_args
        from train_source_uie import train_source
        if '--experiment_dir' in remaining: p.error('Duplicate experiment_dir')
        opt=parse_source_args(['--experiment_dir',args.experiment_dir,*remaining])
        audit=json.loads((Path(args.experiment_dir)/'manifests/audit.json').read_text())
        if audit['protocol']!='UIEB4_light_to_severe_v1' or audit['preparation_sha256']!=sha(__file__):
            raise ValueError('Prepare the new audited protocol first')
        if opt.seed!=audit['seed']: raise ValueError('Source seed must match split seed')
        dataset=Path(audit['dataset_dir'])
        opt.source_raw=str(dataset/'raw/light');opt.source_ref=str(dataset/'reference/light')
        opt.target_raw=str(dataset/'raw/severe');opt.target_ref=str(dataset/'reference/severe')
        opt.expected_source=306;opt.expected_target=172;opt.val_count=62;opt.exclude_source_ids=[]
        train_source(opt)


if __name__=='__main__': main()
