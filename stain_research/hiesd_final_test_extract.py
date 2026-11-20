"""Reserved-test extraction under frozen final protocol; no fitting."""
import csv,json,math,shutil,time,hashlib
from pathlib import Path
from urllib.parse import quote
from collections import defaultdict,Counter
import numpy as np
from PIL import Image,ImageDraw
import openslide
from hiesd_pilot import ROOT,META,NAMES,SESSION,digest,download
from hiesd_expand_v3 import save
from hiesd_expand_pilot import labels_for,eligible,read_csv
OUT=ROOT/'data/external/hiesd_final_test_v1'
PORTABLE=ROOT/'stain_research/results/final_test_hiesd_v1'

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
        assert mapping[f['slide_id']]['split']==f['split'] and f['split'] in ['reserved_test']
        save(OUT/'STATUS.json',{'stage':'downloading','completed_slides':i,'total_slides':len(protocol['selected']),'slide':f['slide_id']})
        print(f"Download {i+1}/{len(protocol['selected'])}: {f['slide_id']}",flush=True)
        download(f['download_url'],OUT/'slides'/f['name'],f['size'],f['computed_md5'],'md5')
    sources={f['slide_id']:(f,OUT/'slides'/f['name']) for f in protocol['selected']}
    candidates=[];receipts=[]
    # Checksum-verified released masks can exceed Pillow's generic size warning;
    # process only one grayscale QC mask at a time, with an explicit pixel bound.
    Image.MAX_IMAGE_PIXELS=None
    for ordinal,(sid,(f,source)) in enumerate(sources.items()):
        assert f['split'] in ['reserved_test'] and mapping[sid]['split']==f['split']
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
    with (ROOT/'stain_research/runs/hiesd_resnet_ablation_v1/cache_manifest.csv').open() as f:
        original_hashes={r['sha256'] for r in csv.DictReader(f)}
    test_hashes=[digest(ROOT/r['image_path']) for r in rows]
    assert not original_hashes & set(test_hashes),'Exact development/test collision; evaluation stopped'
    assert len(test_hashes)==len(set(test_hashes)),'Exact within-test duplicates; evaluation stopped'
    with (OUT/'patch_manifest.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cov={name:{'patches':sum(r['class_code']==name for r in rows),'patients':sorted({r['patient_id'] for r in rows if r['class_code']==name})} for name in NAMES}
    ready=all(v['patches']>0 for v in cov.values())
    report={'stage':'qc_complete','patches':len(rows),'coverage':cov,'all_five_classes_present':ready,'exact_development_test_patch_overlap':False,'qc_rule':protocol['qc'],'manifest_sha256':digest(OUT/'patch_manifest.csv'),'receipts':receipts}
    save(OUT/'STATUS.json',report);save(PORTABLE/'QC_REPORT.json',report)
    preview=[]
    for split in ['reserved_test']:
        for name in NAMES:preview += [r for r in rows if r['split']==split and r['class_code']==name][:3]
    canvas=Image.new('RGB',(672,math.ceil(len(preview)/3)*250),'white');draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(preview):
        x,y=i%3*224,i//3*250
        with Image.open(ROOT/r['image_path']) as image:canvas.paste(image,(x,y))
        draw.text((x,y+224),r['split']+' '+r['class_code'],fill='black')
    canvas.save(PORTABLE/'PATCH_QC.jpg')
    return ready
