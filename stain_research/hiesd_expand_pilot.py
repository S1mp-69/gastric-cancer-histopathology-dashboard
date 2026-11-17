"""Label-coverage acquisition pilot, bounded to 8 GiB/12 additional slides.

This deliberately selected engineering subset is not an unbiased evaluation cohort.
Only frozen train/validation patients are examined. No classifier is trained here.
"""
import argparse
import csv
import json
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import numpy as np
from PIL import Image, ImageDraw
import openslide
from hiesd_pilot import ROOT, META, COLORS, GROUPS, NAMES, SESSION, digest, download, save

BASE = ROOT/'data/external/hiesd_pilot_v1'
OUT = ROOT/'data/external/hiesd_pilot_v2'
BUDGET = 8*1024**3
MAX_SLIDES = 12

def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def labels_for(path):
    mask = np.asarray(Image.open(path).convert('RGB'))
    labels = np.full(mask.shape[:2],-1,dtype=np.int8)
    for color,group in zip(COLORS,GROUPS):
        labels[np.all(mask==color,axis=2)] = group
    return labels

def eligible(labels):
    # Same non-overlapping 896-level-zero-pixel grid and >=90% purity as v1.
    for my in range(0,labels.shape[0]-13,14):
        for mx in range(0,labels.shape[1]-13,14):
            block = labels[my:my+14,mx:mx+14]
            counts = np.bincount(block[block>=0],minlength=len(NAMES))
            cls = int(counts.argmax())
            if counts[cls]/196 >= .9:
                yield mx,my,cls,float(counts[cls]/196)

def coverage(rows):
    result = {}
    for split in ['train','validation']:
        result[split] = {}
        for name in NAMES:
            subset = [r for r in rows if r['split']==split and r['class_code']==name]
            result[split][name] = {'patches':len(subset),'patients':sorted({r['patient_id'] for r in subset})}
    return result

def satisfied(cov):
    return all(cov[s][n]['patches'] >= (32 if s=='train' else 16)
               and len(cov[s][n]['patients'])>=2 for s in cov for n in NAMES)

def main(plan_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['annotations','slides','patches']:
        (OUT/name).mkdir(exist_ok=True)
    previous_plan = json.loads((BASE/'PLAN.json').read_text())
    revision = previous_plan['hf_revision']
    manifest = META/'prospective_grouped_manifest.csv'
    if digest(manifest) != previous_plan['manifest_sha256']:
        raise RuntimeError('Frozen patient split changed')
    mapping = {r['slide_id']:r for r in read_csv(manifest)}
    assert not {r['patient_id'] for r in mapping.values() if r['split']=='train'} & {r['patient_id'] for r in mapping.values() if r['split']=='validation'}
    if (OUT/'FAILURE.json').exists():
        (OUT/'FAILURE.json').rename(OUT/f'FAILURE_history_{time.time_ns()}.json')
    tree_response = SESSION.get(f'https://huggingface.co/api/datasets/JSGe-AI/HiESD/tree/{revision}/ESD_40X_annotation_downsample64?limit=1000',timeout=30)
    tree_response.raise_for_status()
    if 'next' in tree_response.links:
        raise RuntimeError('Unexpected paginated mask directory; refusing incomplete inventory')
    annotations = {Path(f['path']).stem:f for f in tree_response.json() if f['type']=='file'}
    files = json.loads((ROOT/'evaluation_package/publication_readiness/evidence/hiesd_figshare.json').read_text(encoding='utf-8-sig'))['files']
    original_ids = {f['slide_id'] for f in previous_plan['selected']}
    candidates = []
    for f in files:
        sid = f['name'].removesuffix('.svs')
        row = mapping.get(sid)
        if row and row['split'] in ['train','validation'] and sid not in original_ids:
            assert sid in annotations
            candidates.append(dict(f,slide_id=sid,patient_id=row['patient_id'],split=row['split']))
    save(OUT/'STATUS.json',{'stage':'scanning_non_test_annotations','masks_total':len(candidates),'masks_complete':0,'training_started':False})

    def inspect(f):
        a = annotations[f['slide_id']]
        path = OUT/'annotations'/f"{f['slide_id']}.png"
        download(f"https://huggingface.co/datasets/JSGe-AI/HiESD/resolve/{revision}/{quote(a['path'])}",path,a['size'],a['lfs']['oid'],'sha256')
        counts = Counter(cls for _,_,cls,_ in eligible(labels_for(path)))
        return dict(f,eligible_patch_counts=[min(counts[c],128) for c in range(len(NAMES))])

    inventory = []
    # Four network workers only; no CPU/GPU training process or AI polling.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(inspect,candidates):
            inventory.append(item)
            save(OUT/'STATUS.json',{'stage':'scanning_non_test_annotations','masks_total':len(candidates),'masks_complete':len(inventory),'training_started':False})
    save(OUT/'ANNOTATION_INVENTORY.json',inventory)
    rows = read_csv(BASE/'patch_manifest.csv')
    base_manifest_sha = digest(BASE/'patch_manifest.csv')
    cov = coverage(rows)

    def choose(pool,current):
        ranked = []
        for f in pool:
            gain = 0.0
            for cls,n in enumerate(NAMES):
                current_class = current[f['split']][n]
                target = 32 if f['split']=='train' else 16
                extra = f['eligible_patch_counts'][cls]
                if not extra:
                    continue
                gain += min(extra,max(0,target-current_class['patches']))/target
                if len(current_class['patients'])<2 and f['patient_id'] not in current_class['patients']:
                    gain += 1.0
            if gain>0:
                ranked.append((gain/(f['size']/1024**3+.1),f['slide_id'],f))
        return max(ranked,key=lambda x:(x[0],x[1]))[2] if ranked else None

    prospective = json.loads(json.dumps(cov))
    pool = list(inventory)
    selected = []
    while not satisfied(prospective) and len(selected)<MAX_SLIDES:
        f = choose([x for x in pool if sum(s['size'] for s in selected)+x['size']<=BUDGET],prospective)
        if f is None:
            break
        selected.append(f);pool.remove(f)
        for cls,name in enumerate(NAMES):
            if f['eligible_patch_counts'][cls]:
                entry = prospective[f['split']][name]
                entry['patches'] += f['eligible_patch_counts'][cls]
                entry['patients'] = sorted(set(entry['patients'])|{f['patient_id']})
    plan = {'version':'hiesd-coverage-pilot-v2','hf_revision':revision,'manifest_sha256':digest(manifest),
            'base_patch_manifest_sha256':base_manifest_sha,'raw_bytes':sum(f['size'] for f in selected),'selected':selected,
            'max_additional_bytes':BUDGET,'max_additional_slides':MAX_SLIDES,
            'selection':'Annotation-guided class coverage and distinct patients within frozen train/validation partitions; size-cost preference; excludes v1 slides. Biased feasibility subset, not representative evaluation.',
            'coverage_targets':{'train_patches_per_class':32,'validation_patches_per_class':16,'patients_per_class_per_split':2},
            'predicted_coverage':prospective,'predicted_targets_met':satisfied(prospective),
            'patch_protocol':previous_plan['patch_protocol'],'class_order':NAMES,
            'test_policy':'No reserved-test annotation or slide downloaded/decoded; no classifier output used in selection.',
            'training_policy':'No automatic training. Five-class mapping is provisional and requires source-task and annotation review.',
            'failure_policy':'Ambiguous alignments excluded; no inferred stretching/cropping, no automatic replacement downloads beyond frozen plan.'}
    if (OUT/'PLAN.json').exists() and json.loads((OUT/'PLAN.json').read_text()) != plan:
        raise RuntimeError('Existing v2 acquisition plan differs; refusing to overwrite')
    save(OUT/'PLAN.json',plan)
    print(f"Plan: {len(selected)} additional slides, {plan['raw_bytes']/1024**3:.2f} GiB; predicted coverage targets met: {plan['predicted_targets_met']}",flush=True)
    if plan_only:
        save(OUT/'STATUS.json',{'stage':'planned','additional_slides':len(selected),'raw_bytes':plan['raw_bytes'],'training_started':False})
        return
    if shutil.disk_usage(OUT).free < plan['raw_bytes']+15*1024**3:
        raise RuntimeError('Insufficient disk headroom')
    receipts = []
    started = time.perf_counter()
    for i,f in enumerate(selected):
        sid = f['slide_id']
        save(OUT/'STATUS.json',{'stage':'acquisition','slide':sid,'completed_slides':i,'total_slides':len(selected),'training_started':False})
        print(f"Acquiring {i+1}/{len(selected)} {f['split']} {sid}",flush=True)
        target = OUT/'slides'/f['name']
        download(f['download_url'],target,f['size'],f['computed_md5'],'md5')
        labels = labels_for(OUT/'annotations'/f'{sid}.png')
        with openslide.OpenSlide(str(target)) as slide:
            width,height = slide.dimensions
            receipt = {'slide_id':sid,'split':f['split'],'patient_id':f['patient_id'],
                       'md5':f['computed_md5'],'dimensions':[width,height],'mask_dimensions':list(labels.shape),
                       'mpp_x':slide.properties.get('openslide.mpp-x'),'objective':slide.properties.get('openslide.objective-power')}
            if abs(labels.shape[1]-width/64)>2 or abs(labels.shape[0]-height/64)>2:
                receipt['status'] = 'excluded_alignment_unverified'
                receipt['native_level_dimensions'] = [list(x) for x in slide.level_dimensions]
                receipts.append(receipt);save(OUT/'receipts.json',receipts)
                print('Excluded: ambiguous alignment',flush=True)
                continue
            counts = Counter()
            for mx,my,cls,purity in eligible(labels):
                x,y = mx*64,my*64
                if x+896>width or y+896>height or counts[cls]>=128:
                    continue
                patch = OUT/'patches'/f['split']/NAMES[cls]/f'{sid}_{x}_{y}.png'
                patch.parent.mkdir(parents=True,exist_ok=True)
                if not patch.exists():
                    slide.read_region((x,y),0,(896,896)).convert('RGB').resize((224,224),Image.Resampling.LANCZOS).save(patch)
                counts[cls] += 1
                rows.append({'image_path':str(patch.relative_to(ROOT)),'slide_id':sid,'patient_id':f['patient_id'],'split':f['split'],
                             'class_code':NAMES[cls],'class_id':cls,'x_level0':x,'y_level0':y,'mask_group_purity':purity})
            receipt['status'] = 'extracted_nominal_alignment_checked'
            receipt['patch_counts'] = {n:counts[c] for c,n in enumerate(NAMES)}
            receipts.append(receipt);save(OUT/'receipts.json',receipts)
            print(receipt['patch_counts'],flush=True)
    assert not {r['patient_id'] for r in rows if r['split']=='train'} & {r['patient_id'] for r in rows if r['split']=='validation'}
    assert digest(BASE/'patch_manifest.csv') == base_manifest_sha
    with (OUT/'patch_manifest.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    preview = []
    for split in ['train','validation']:
        for name in NAMES:
            preview.extend([r for r in rows if r['split']==split and r['class_code']==name][:3])
    canvas = Image.new('RGB',(3*224,((len(preview)+2)//3)*250),'white');draw = ImageDraw.Draw(canvas)
    for i,r in enumerate(preview):
        x,y = (i%3)*224,(i//3)*250
        with Image.open(ROOT/r['image_path']) as im:
            canvas.paste(im,(x,y))
        draw.text((x,y+224),r['split']+' '+r['class_code'],fill='black')
    canvas.save(OUT/'PATCH_QC.jpg')
    cov = coverage(rows)
    save(OUT/'STATUS.json',{'stage':'acquisition_complete_review_required','additional_slides':len(selected),'patches':len(rows),
                          'coverage':cov,'coverage_targets_met':satisfied(cov),
                          'excluded_slides':[r['slide_id'] for r in receipts if r['status']=='excluded_alignment_unverified'],
                          'training_started':False,'reserved_test_acquired':False,'elapsed_seconds':time.perf_counter()-started,
                          'limitations':['Annotation-guided biased feasibility subset','Small patient counts','Provisional class grouping','Coarse-mask coordinate approximation','No expert label review']})
    print('Acquisition complete. Review STATUS.json and PATCH_QC.jpg before any classifier pilot.',flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true')
    args = parser.parse_args()
    try:
        main(args.plan_only)
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        save(OUT/'FAILURE.json',{'error':str(exc),'time':time.time()})
        raise
