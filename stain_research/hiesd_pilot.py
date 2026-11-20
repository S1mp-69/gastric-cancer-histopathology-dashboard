"""Bounded HiESD acquisition and patch QC. Never acquires reserved-test slides."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from urllib.parse import quote
import requests
import numpy as np
from PIL import Image, ImageDraw
import openslide

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'data/external/hiesd_metadata'
OUT = ROOT / 'data/external/hiesd_pilot_v1'
COLORS = [(139,0,0),(255,0,255),(128,0,128),(75,0,130),(138,43,226),
          (0,0,255),(70,130,180),(0,255,0),(0,128,0),(255,255,0)]
NAMES = ['tumor','mucosa_inflammation','intestinal_metaplasia','lymphoid','other']
GROUPS = [0,0,0,4,1,1,1,3,2,2]
SESSION = requests.Session()

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def download(url, target, expected_size, checksum, algorithm):
    if target.exists():
        if target.stat().st_size == expected_size and digest(target, algorithm) == checksum:
            return
        raise RuntimeError(f'Existing file failed verification: {target}')
    partial = target.with_suffix(target.suffix+'.part')
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < expected_size:
                response = SESSION.get(url, headers={'Range':f'bytes={offset}-'} if offset else {},
                                       stream=True, timeout=(30,90))
                response.raise_for_status()
                if offset and response.status_code == 206:
                    if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Invalid resumed byte range')
                    mode = 'ab'
                else:
                    mode = 'wb'
                with response, partial.open(mode) as stream:
                    for block in response.iter_content(1024*1024):
                        stream.write(block)
                        if stream.tell() > expected_size:
                            raise RuntimeError('Download exceeded declared size')
            if partial.stat().st_size != expected_size or digest(partial, algorithm) != checksum:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f'Download integrity check failed: {target.name}')
            partial.replace(target)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2*(attempt+1))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'slides').mkdir(exist_ok=True)
    (OUT/'annotations').mkdir(exist_ok=True)
    rows = list(csv.DictReader((META/'prospective_grouped_manifest.csv').open()))
    mapping = {r['slide_id']:r for r in rows}
    files = json.loads((ROOT/'evaluation_package/publication_readiness/evidence/hiesd_figshare.json').read_text(encoding='utf-8-sig'))['files']
    selected = []
    for split, count in [('train',4),('validation',2)]:
        seen = set()
        for f in sorted(files, key=lambda f:(f['size'],f['name'])):
            r = mapping.get(f['name'].removesuffix('.svs'))
            if r and r['split'] == split and r['patient_id'] not in seen:
                selected.append(dict(f, slide_id=r['slide_id'], patient_id=r['patient_id'], split=split))
                seen.add(r['patient_id'])
                if len(seen) == count:
                    break
        assert len(seen) == count
    assert len({f['patient_id'] for f in selected}) == 6
    total = sum(f['size'] for f in selected)
    assert total < 2*1024**3 and all(f['split'] != 'reserved_test' for f in selected)
    if shutil.disk_usage(OUT).free < total + 15*1024**3:
        raise RuntimeError('Need acquisition bytes plus 15 GiB free headroom')
    revision = SESSION.get('https://huggingface.co/api/datasets/JSGe-AI/HiESD/revision/main',timeout=30)
    revision.raise_for_status()
    revision = revision.json()['sha']
    plan = {'version':'hiesd-pilot-v1','manifest_sha256':digest(META/'prospective_grouped_manifest.csv'),
            'hf_revision':revision,'selected':selected,'raw_bytes':total,
            'selection':'Four smallest train and two smallest validation slides, distinct patients, metadata only; feasibility subset, size-biased, not representative evaluation.',
            'patch_protocol':{'level0_span':896,'output_size':224,'nominal_magnification':'10x if source 40x',
                              'mask_downsample':64,'stride_level0':896,'minimum_group_purity':0.9,'cap_per_slide_per_class':128},
            'class_order':NAMES,'test_policy':'Reserved-test slides never downloaded or decoded.',
            'scope':'Acquisition/engineering pilot; coarse-mask labels require review; no clinical validation.'}
    plan_path = OUT/'PLAN.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Existing plan differs; refusing to overwrite frozen pilot')
    save(plan_path,plan)
    # Preserve previous failures, but clear the live error indicator on an explicit retry.
    if (OUT/'FAILURE.json').exists():
        (OUT/'FAILURE.json').rename(OUT/f'FAILURE_history_{time.time_ns()}.json')
    save(OUT/'STATUS.json',{'stage':'acquisition','completed_slides':0,'total_slides':len(selected)})
    tree = SESSION.get(f'https://huggingface.co/api/datasets/JSGe-AI/HiESD/tree/{revision}/ESD_40X_annotation_downsample64?limit=1000',timeout=30)
    tree.raise_for_status()
    annotations = {Path(f['path']).stem:f for f in tree.json() if f['type']=='file'}
    receipts, patches, previews = [], [], []
    started = time.perf_counter()
    for ordinal, f in enumerate(selected):
        sid = f['slide_id']
        print(f"Acquiring {ordinal+1}/6 {f['split']} {sid} ({f['size']/1024**2:.1f} MiB)",flush=True)
        save(OUT/'STATUS.json',{'stage':'acquisition','slide':sid,'completed_slides':len(receipts),'total_slides':6})
        slide_path = OUT/'slides'/f['name']
        download(f['download_url'],slide_path,f['size'],f['computed_md5'],'md5')
        a = annotations[sid]
        annotation = OUT/'annotations'/f'{sid}.png'
        # These masks are LFS objects; their content digest is recorded by the API.
        download(f"https://huggingface.co/datasets/JSGe-AI/HiESD/resolve/{revision}/{quote(a['path'])}",
                 annotation,a['size'],a['lfs']['oid'],'sha256')
        mask = np.asarray(Image.open(annotation).convert('RGB'))
        labels = np.full(mask.shape[:2],-1,dtype=np.int8)
        for color, group in zip(COLORS,GROUPS):
            labels[np.all(mask == color,axis=2)] = group
        counts = [0]*len(NAMES)
        slide_previews = []
        with openslide.OpenSlide(str(slide_path)) as slide:
            width,height = slide.dimensions
            # Check the stated coordinate relationship before patch extraction.
            if abs(mask.shape[1]-width/64)>2 or abs(mask.shape[0]-height/64)>2:
                # A native pyramid level matching the mask does not establish a
                # valid mapping to level zero when its aspect ratio is distorted.
                # Retain the verified source and exclude it instead of stretching
                # the labels, changing the patient split, or silently replacing it.
                receipts.append({'slide_id':sid,'split':f['split'],'patient_id':f['patient_id'],
                                 'md5':f['computed_md5'],'dimensions':[width,height],
                                 'mask_dimensions':list(mask.shape[:2]),
                                 'native_level_dimensions':[list(x) for x in slide.level_dimensions],
                                 'native_level_downsamples':list(slide.level_downsamples),
                                 'status':'excluded_alignment_unverified',
                                 'reason':'Released mask does not match nominal downsample64 dimensions; no coordinate correction inferred.',
                                 'patch_counts':dict(zip(NAMES,counts))})
                save(OUT/'receipts.json',receipts)
                print(f'Excluded from patch extraction: {sid}; ambiguous mask/level-zero alignment. Continuing.',flush=True)
                continue
            for my in range(0,mask.shape[0]-13,14):
                for mx in range(0,mask.shape[1]-13,14):
                    x,y = mx*64,my*64
                    if x+896>width or y+896>height:
                        continue
                    region = labels[my:my+14,mx:mx+14]
                    histogram = np.bincount(region[region>=0],minlength=len(NAMES))
                    cls = int(histogram.argmax())
                    purity = float(histogram[cls]/196)
                    if purity<.9 or counts[cls]>=128:
                        continue
                    target = OUT/'patches'/f['split']/NAMES[cls]/f'{sid}_{x}_{y}.png'
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():
                        image = slide.read_region((x,y),0,(896,896)).convert('RGB').resize((224,224),Image.Resampling.LANCZOS)
                        image.save(target)
                    counts[cls] += 1
                    patches.append({'image_path':str(target.relative_to(ROOT)),'slide_id':sid,'patient_id':f['patient_id'],
                                    'split':f['split'],'class_code':NAMES[cls],'class_id':cls,'x_level0':x,'y_level0':y,'mask_group_purity':purity})
                    if len(slide_previews)<5:
                        slide_previews.append((target,NAMES[cls]))
            thumb = slide.get_thumbnail((360,360)).convert('RGB')
            thumb.save(OUT/f'{sid}_overview.jpg')
            receipts.append({'slide_id':sid,'split':f['split'],'patient_id':f['patient_id'],'md5':f['computed_md5'],
                             'status':'extracted_nominal_alignment_checked',
                             'dimensions':list(slide.dimensions),'mpp_x':slide.properties.get('openslide.mpp-x'),
                             'objective':slide.properties.get('openslide.objective-power'),'mask_dimensions':list(mask.shape[:2]),
                             'patch_counts':dict(zip(NAMES,counts))})
        previews.extend(slide_previews)
        save(OUT/'receipts.json',receipts)
        print(f'Verified and extracted: {counts}',flush=True)
    if not patches:
        raise RuntimeError('No usable patches extracted; inspect receipts before proceeding')
    with (OUT/'patch_manifest.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(patches[0]));writer.writeheader();writer.writerows(patches)
    canvas = Image.new('RGB',(5*224,((len(previews)+4)//5)*250),'white')
    draw = ImageDraw.Draw(canvas)
    for i,(path,cls) in enumerate(previews):
        x,y = (i%5)*224,(i//5)*250
        canvas.paste(Image.open(path),(x,y));draw.text((x,y+224),cls,fill='black')
    canvas.save(OUT/'PATCH_QC.jpg')
    coverage = {split:{name:sum(p['split']==split and p['class_code']==name for p in patches) for name in NAMES} for split in ['train','validation']}
    save(OUT/'STATUS.json',{'stage':'acquisition_and_patch_qc_complete','slides':6,'patients':6,'raw_bytes':total,
                          'usable_slides':sum(r['status']=='extracted_nominal_alignment_checked' for r in receipts),
                          'excluded_slides':[r['slide_id'] for r in receipts if r['status']=='excluded_alignment_unverified'],
                          'patches':len(patches),'class_counts':coverage,'elapsed_seconds':time.perf_counter()-started,
                          'training_started':False,'reserved_test_downloaded':False,
                          'limitations':['Size-biased feasibility subset with explicitly excluded ambiguous alignments','Coarse annotation boundary uncertainty','No expert pathology review','No classification or novelty result']})
    print(json.dumps(coverage),flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            save(OUT/'FAILURE.json',{'error':str(exc),'time':time.time()})
        raise
