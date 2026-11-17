"""Audit frozen pilot labels/coordinates, including stratified source-pixel checks.

Checks annotation consistency, not correctness of the original pathology labels.
Never opens reserved-test images or masks.
"""
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import openslide
from hiesd_pilot import ROOT,COLORS,GROUPS,NAMES,digest,save

def main():
    out=ROOT/'stain_research/results/hiesd_weighted_diagnostic_20261116'
    out.mkdir(parents=True,exist_ok=True)
    pilot=ROOT/'data/external/hiesd_pilot_v2'
    with (pilot/'patch_manifest.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    with (ROOT/'data/external/hiesd_metadata/prospective_grouped_manifest.csv').open(newline='') as stream:
        mapping={r['slide_id']:r for r in csv.DictReader(stream)}
    sources={}
    for version in ['v1','v2']:
        folder=ROOT/f'data/external/hiesd_pilot_{version}'
        for f in json.loads((folder/'PLAN.json').read_text())['selected']:
            sources[f['slide_id']]=(folder/'slides'/f['name'],folder/'annotations'/f"{f['slide_id']}.png")
    groups=defaultdict(list)
    for r in rows:
        assert r['split'] in ['train','validation']
        assert mapping[r['slide_id']]['split']==r['split'] and mapping[r['slide_id']]['patient_id']==r['patient_id']
        assert NAMES[int(r['class_id'])]==r['class_code']
        groups[r['slide_id']].append(r)
    regenerated=[];hashes={'train':set(),'validation':set()};purities=[];masks=[]
    for sid,slide_rows in groups.items():
        source,annotation=sources[sid]
        mask=np.array(Image.open(annotation).convert('RGB'));labels=np.full(mask.shape[:2],-1,dtype=np.int8)
        for color,group in zip(COLORS,GROUPS):labels[np.all(mask==color,axis=2)]=group
        masks.append({'slide_id':sid,'unmapped_mask_pixels':int((labels<0).sum()),'mask_sha256':digest(annotation)})
        checked=Counter()
        with openslide.OpenSlide(str(source)) as slide:
            assert abs(slide.dimensions[0]/64-mask.shape[1])<=2 and abs(slide.dimensions[1]/64-mask.shape[0])<=2
            for r in slide_rows:
                x,y=int(r['x_level0']),int(r['y_level0']);cls=int(r['class_id'])
                assert x%896==0 and y%896==0 and x+896<=slide.dimensions[0] and y+896<=slide.dimensions[1]
                block=labels[y//64:y//64+14,x//64:x//64+14]
                purity=float((block==cls).mean());assert block.shape==(14,14) and purity>=.9
                assert abs(purity-float(r['mask_group_purity']))<1e-12;purities.append(purity)
                path=ROOT/r['image_path'];hashes[r['split']].add(digest(path))
                with Image.open(path) as image:
                    assert image.size==(224,224) and image.mode=='RGB'
                    # Two patches per class/slide, including every represented
                    # 'other' slide; chosen without classifier predictions.
                    if checked[cls]<2:
                        expected=slide.read_region((x,y),0,(896,896)).convert('RGB').resize((224,224),Image.Resampling.LANCZOS)
                        assert np.array_equal(np.asarray(image),np.asarray(expected))
                        regenerated.append({'slide_id':sid,'class':NAMES[cls],'x':x,'y':y,'exact_source_reconstruction':True})
                        checked[cls]+=1
    assert not hashes['train']&hashes['validation']
    train_patients={r['patient_id'] for r in rows if r['split']=='train'}
    val_patients={r['patient_id'] for r in rows if r['split']=='validation'}
    assert not train_patients&val_patients
    class_patient_counts=[]
    for split in ['train','validation']:
        for name in NAMES:
            sub=[r for r in rows if r['split']==split and r['class_code']==name]
            class_patient_counts.append({'split':split,'class':name,'patches':len(sub),
                                         'by_patient':dict(Counter(r['patient_id'] for r in sub))})
    selected=[]
    for split in ['train','validation']:
        for patient in sorted({r['patient_id'] for r in rows if r['split']==split and r['class_code']=='other'}):
            selected.extend([r for r in rows if r['split']==split and r['patient_id']==patient and r['class_code']=='other'][:4])
    canvas=Image.new('RGB',(4*224,((len(selected)+3)//4)*260),'white');draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(selected):
        x,y=(i%4)*224,(i//4)*260
        with Image.open(ROOT/r['image_path']) as image:canvas.paste(image,(x,y))
        draw.text((x,y+224),r['split']+' '+r['patient_id']+' / other',fill='black')
        draw.text((x,y+239),f"purity {float(r['mask_group_purity']):.3f}",fill='black')
    canvas.save(out/'other_class_review.jpg')
    save(out/'patch_audit.json',{'patch_manifest_sha256':digest(pilot/'patch_manifest.csv'),'patches_checked':len(rows),
         'all_manifest_class_ids_match_release_color_group_mapping':True,'all_coarse_mask_purities_recomputed':True,
         'source_pixel_checks':regenerated,'mask_checks':masks,'minimum_group_purity':min(purities),
         'class_patient_counts':class_patient_counts,'exact_train_validation_patch_overlap':False,'patient_overlap':False,
         'reserved_test_accessed':False,'verdict':'No extraction/label-index inconsistency detected on implemented checks. Small selected cohort and severe class/patient imbalance remain.',
         'limits':['Nominal downsample64 coordinate check is not landmark registration','No expert verification of released tissue annotations','No GrandQC artifact filtering','Source RGB regenerated on a stratified sample, not all patches']})
    baseline=ROOT/'stain_research/runs/hiesd_baseline_seed42_20261116'
    save(out/'baseline_preservation.json',{name:digest(baseline/name) for name in ['initial.pt','best.pt','last.pt','config.json','RESULTS.json','history.json','validation_predictions.csv']})
    print(json.dumps({'patches_checked':len(rows),'source_pixel_checks':len(regenerated),'class_patient_counts':class_patient_counts}),flush=True)

if __name__=='__main__':main()
