"""Fixed, bounded patient-coverage expansion with consistent released-mask QC.

No reserved-test acquisition, no split changes, and no performance-guided selection.
"""
import argparse,csv,json,math,shutil,subprocess,sys,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
from urllib.parse import quote
import numpy as np
from PIL import Image,ImageDraw
import openslide
from hiesd_pilot import ROOT,META,NAMES,SESSION,digest,download,save as original_save
from hiesd_expand_pilot import read_csv,labels_for,eligible,coverage

OUT=ROOT/'data/external/hiesd_expanded_v3'
PORTABLE=ROOT/'stain_research/results/hiesd_expanded_v3'
RUN=ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3'

def save(path,value):
    for attempt in range(20):
        try:return original_save(path,value)
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def plan():
    OUT.mkdir(parents=True,exist_ok=True);PORTABLE.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/'data/external/hiesd_pilot_v1/PLAN.json').read_text())
    assert digest(META/'prospective_grouped_manifest.csv')==old['manifest_sha256']
    inventory=json.loads((ROOT/'data/external/hiesd_pilot_v2/ANNOTATION_INVENTORY.json').read_text())
    existing=[]
    for version in ['v1','v2']:
        existing+=json.loads((ROOT/f'data/external/hiesd_pilot_{version}/PLAN.json').read_text())['selected']
    existing_ids={x['slide_id'] for x in existing}
    current=read_csv(ROOT/'data/external/hiesd_pilot_v2/patch_manifest.csv')
    by_patient=defaultdict(Counter)
    for r in current:by_patient[(r['split'],r['patient_id'])][int(r['class_id'])]+=1
    pool=[f for f in inventory if f['slide_id'] not in existing_ids]
    targets={'train':[6]*5,'validation':[4,7,5,2,3]}
    def score(f):
        split=f['split'];p=f['patient_id'];gain=0.
        for cls,n in enumerate(f['eligible_patch_counts']):
            represented=sum(v[cls]>=8 for (s,_),v in by_patient.items() if s==split)
            previous=by_patient[(split,p)][cls]
            if represented<targets[split][cls] and previous<8:
                gain+=min(n,8-previous)/8
        # Broader patient coverage even for ubiquitous mucosa.
        have={p for (s,p),v in by_patient.items() if s==split and sum(v.values())>0}
        if p not in have and len(have)<(12 if split=='train' else 7):gain+=2
        return gain/(.25+f['size']/1024**3)
    selected=[];budget=12*1024**3
    while len(selected)<24:
        affordable=[f for f in pool if sum(x['size'] for x in selected)+f['size']<=budget]
        if not affordable:break
        f=max(affordable,key=lambda x:(score(x),x['slide_id']))
        if score(f)<=0:break
        selected.append(f);pool.remove(f)
        by_patient[(f['split'],f['patient_id'])].update({c:n for c,n in enumerate(f['eligible_patch_counts'])})
    protocol={'version':'hiesd-expanded-v3','hf_revision':old['hf_revision'],'manifest_sha256':old['manifest_sha256'],
              'selected':selected,'existing':existing,'raw_bytes':sum(f['size'] for f in selected),'max_additional_bytes':budget,
              'selection':'Frozen metadata/annotation-based patient coverage, size-cost preference, no classifier predictions; selected cohort remains biased.',
              'patient_coverage_targets':targets,'minimum_patches_for_coverage_target':8,
              'qc':{'maximum_artifact_fraction':.05,'maximum_unknown_fraction':.05,'minimum_tissue_fraction':.8,
                    'minimum_annotation_group_purity':.9,'max_patches_per_patient_class':128,'patch_span':896,'patch_output':224,
                    'quality_coordinate_policy':'Normalized full-slide extent; reject aspect ratio disagreement above 0.5%; not landmark registration.'},
              'sampling':'After consistent QC, retain up to 128 patches/patient/class by SHA256(seed42,slide,x,y) order.',
              'test_policy':'Reserved-test slides and annotation/quality masks never opened.',
              'training':'One fresh ImageNet-pretrained ResNet18 seed42, unweighted CE, max40 epochs, patience7; no augmentation campaign.',
              'limitations':['Annotation-guided selected cohort','Validation lymphoid limited to two patients','Quality labels are automated proxies, not expert review']}
    target=OUT/'PLAN.json'
    if target.exists() and json.loads(target.read_text())!=protocol:raise RuntimeError('Frozen v3 plan changed')
    save(target,protocol);save(PORTABLE/'PLAN.json',protocol)
    save(OUT/'STATUS.json',{'stage':'planned','additional_slides':len(selected),'additional_GiB':protocol['raw_bytes']/1024**3})
    print(f"Fixed plan: {len(selected)} additional slides, {protocol['raw_bytes']/1024**3:.2f} GiB",flush=True)
    return protocol

def acquire(protocol):
    for name in ['slides','annotations','quality','patches']:(OUT/name).mkdir(exist_ok=True)
    if shutil.disk_usage(OUT).free<protocol['raw_bytes']+20*1024**3:raise RuntimeError('Insufficient space with 20 GiB headroom')
    revision=protocol['hf_revision']
    trees={}
    for name in ['ESD_40X_annotation_downsample64','ESD_40X_Quality_Assessment_Mask']:
        r=SESSION.get(f'https://huggingface.co/api/datasets/JSGe-AI/HiESD/tree/{revision}/{name}?limit=1000',timeout=30);r.raise_for_status()
        assert 'next' not in r.links
        trees[name]={Path(x['path']).stem:x for x in r.json() if x['type']=='file'}
    mapping={r['slide_id']:r for r in read_csv(META/'prospective_grouped_manifest.csv')}
    for i,f in enumerate(protocol['selected']):
        assert mapping[f['slide_id']]['split']==f['split'] and f['split'] in ['train','validation']
        save(OUT/'STATUS.json',{'stage':'downloading','completed_slides':i,'total_slides':len(protocol['selected']),'slide':f['slide_id']})
        print(f"Download {i+1}/{len(protocol['selected'])}: {f['slide_id']}",flush=True)
        download(f['download_url'],OUT/'slides'/f['name'],f['size'],f['computed_md5'],'md5')
    sources={}
    for version in ['v1','v2']:
        folder=ROOT/f'data/external/hiesd_pilot_{version}'
        for f in json.loads((folder/'PLAN.json').read_text())['selected']:sources[f['slide_id']]=(f,folder/'slides'/f['name'])
    for f in protocol['selected']:sources[f['slide_id']]=(f,OUT/'slides'/f['name'])
    candidates=[];receipts=[]
    # Checksum-verified released masks can exceed Pillow's generic size warning;
    # process only one grayscale QC mask at a time, with an explicit pixel bound.
    Image.MAX_IMAGE_PIXELS=None
    for ordinal,(sid,(f,source)) in enumerate(sources.items()):
        assert f['split'] in ['train','validation'] and mapping[sid]['split']==f['split']
        save(OUT/'STATUS.json',{'stage':'quality_screening','slide_index':ordinal+1,'total_qc_slides':len(sources),'slide':sid})
        for folder,tree_name in [('annotations','ESD_40X_annotation_downsample64'),('quality','ESD_40X_Quality_Assessment_Mask')]:
            item=trees[tree_name][sid];target=OUT/folder/f'{sid}.png'
            download(f"https://huggingface.co/datasets/JSGe-AI/HiESD/resolve/{revision}/{quote(item['path'])}",target,item['size'],item['lfs']['oid'],'sha256')
        labels=labels_for(OUT/'annotations'/f'{sid}.png')
        with openslide.OpenSlide(str(source)) as slide,Image.open(OUT/'quality'/f'{sid}.png') as quality:
            width,height=slide.dimensions
            record={'slide_id':sid,'patient_id':f['patient_id'],'split':f['split'],'dimensions':[width,height],'quality_dimensions':list(quality.size)}
            if abs(width/64-labels.shape[1])>2 or abs(height/64-labels.shape[0])>2:
                record['status']='excluded_annotation_alignment';receipts.append(record);continue
            if quality.mode not in ['L','P'] or quality.width*quality.height>1_000_000_000 or abs((quality.width/quality.height)/(width/height)-1)>.005:
                record['status']='excluded_quality_alignment_or_format';receipts.append(record);continue
            # Load only the current mask. Pixel-level quality fractions are
            # computed before resizing, avoiding nearest-neighbor loss of artifacts.
            quality.load();rejected=Counter();passed=0
            for mx,my,cls,purity in eligible(labels):
                x,y=mx*64,my*64
                if x+896>width or y+896>height:continue
                box=(int(x/width*quality.width),int(y/height*quality.height),math.ceil((x+896)/width*quality.width),math.ceil((y+896)/height*quality.height))
                q=np.array(quality.crop(box))
                if q.ndim!=2:raise RuntimeError('Quality mask is not indexed labels')
                artifact=float(((q>=2)&(q<=6)).mean());unknown=float(((q<1)|(q>7)).mean());tissue=float((q==1).mean())
                if artifact>.05:rejected['artifact']+=1;continue
                if unknown>.05:rejected['unknown']+=1;continue
                if tissue<.8:rejected['low_tissue']+=1;continue
                candidates.append({'slide_id':sid,'patient_id':f['patient_id'],'split':f['split'],'class_code':NAMES[cls],'class_id':cls,
                                   'x_level0':x,'y_level0':y,'mask_group_purity':purity,'artifact_fraction':artifact,'quality_tissue_fraction':tissue,
                                   'quality_unknown_fraction':unknown,'source':str(source)})
                passed+=1
            record.update(status='screened',passed_before_sampling=passed,rejected=dict(rejected));receipts.append(record)
        save(OUT/'qc_receipts.json',receipts)
    grouped=defaultdict(list)
    for r in candidates:grouped[(r['split'],r['patient_id'],r['class_id'])].append(r)
    chosen=[]
    import hashlib
    for key,items in sorted(grouped.items()):
        chosen+=sorted(items,key=lambda r:hashlib.sha256(f"42:{r['slide_id']}:{r['x_level0']}:{r['y_level0']}".encode()).hexdigest())[:128]
    rows=[]
    for sid in sorted({r['slide_id'] for r in chosen}):
        items=[r for r in chosen if r['slide_id']==sid]
        with openslide.OpenSlide(items[0]['source']) as slide:
            for r in items:
                target=OUT/'patches'/r['split']/r['class_code']/f"{sid}_{r['x_level0']}_{r['y_level0']}.png";target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():slide.read_region((r['x_level0'],r['y_level0']),0,(896,896)).convert('RGB').resize((224,224),Image.Resampling.LANCZOS).save(target)
                rows.append(dict({k:v for k,v in r.items() if k!='source'},image_path=str(target.relative_to(ROOT))))
    if not rows:raise RuntimeError('No QC-passing patches')
    hashes={'train':set(),'validation':set()}
    for r in rows:hashes[r['split']].add(digest(ROOT/r['image_path']))
    assert not hashes['train']&hashes['validation']
    assert not {r['patient_id'] for r in rows if r['split']=='train'}&{r['patient_id'] for r in rows if r['split']=='validation'}
    with (OUT/'patch_manifest.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cov=coverage(rows)
    ready=all(cov[s][n]['patches']>=(16 if s=='train' else 4) and len(cov[s][n]['patients'])>=2 for s in cov for n in NAMES)
    report={'stage':'qc_complete','patches':len(rows),'coverage':cov,'training_gate_passed':ready,'patient_overlap':False,'exact_patch_overlap':False,
            'reserved_test_accessed':False,'qc_rule':protocol['qc'],'manifest_sha256':digest(OUT/'patch_manifest.csv'),'receipts':receipts}
    save(OUT/'STATUS.json',report);save(PORTABLE/'QC_REPORT.json',report)
    preview=[]
    for split in ['train','validation']:
        for name in NAMES:preview += [r for r in rows if r['split']==split and r['class_code']==name][:3]
    canvas=Image.new('RGB',(672,math.ceil(len(preview)/3)*250),'white');draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(preview):
        x,y=i%3*224,i//3*250
        with Image.open(ROOT/r['image_path']) as image:canvas.paste(image,(x,y))
        draw.text((x,y+224),r['split']+' '+r['class_code'],fill='black')
    canvas.save(PORTABLE/'PATCH_QC.jpg')
    return ready

def main(plan_only):
    if not plan_only and (OUT/'PIPELINE_STATUS.json').exists() and json.loads((OUT/'PIPELINE_STATUS.json').read_text()).get('stage')=='complete':
        print('Pipeline already complete; no additional run launched.',flush=True);return
    if (OUT/'FAILURE.json').exists():
        (OUT/'FAILURE.json').rename(OUT/f'FAILURE_history_{time.time_ns()}.json')
    p=plan()
    if plan_only:return
    if not acquire(p):raise RuntimeError('Coverage gate failed after fixed QC; no training launched, review QC_REPORT.json')
    RUN.mkdir(parents=True,exist_ok=True)
    for mode in ['prepare','train']:
        save(OUT/'PIPELINE_STATUS.json',{'stage':'resnet_'+mode,'run':str(RUN)})
        command=[sys.executable,'-u',str(ROOT/'stain_research/hiesd_resnet18.py'),'--run',str(RUN)]
        if mode=='prepare':command+=['--prepare']
        elif (RUN/'last.pt').exists():command+=['--resume']
        with (RUN/f'{mode}.log').open('a') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
    with (RUN/'report.log').open('a') as log:
        subprocess.run([sys.executable,str(ROOT/'stain_research/hiesd_baseline_report.py'),'--run',str(RUN)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
    save(OUT/'PIPELINE_STATUS.json',{'stage':'complete','run':str(RUN),'reports':str(ROOT/'stain_research/results'/RUN.name)})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    try:main(args.plan_only)
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True);save(OUT/'FAILURE.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
