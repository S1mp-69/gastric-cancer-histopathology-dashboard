"""TRAIN-only mask audit of one-factor benchmark-construction alternatives.

No RGB inference, training, test/validation mask reading, or label corrections.
"""
import csv,hashlib,json,math,time
from collections import Counter,defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from PIL import Image
from hiesd_pilot import ROOT,META,NAMES,digest
from hiesd_resnet18 import atomic
from hiesd_expand_pilot import labels_for,read_csv

def key(r):return (r['slide_id'],int(r['x_level0']),int(r['y_level0']),int(r['class_id']))

def main():
    folder=ROOT/'data/external/hiesd_expanded_v3'
    stamp=datetime.now(ZoneInfo('Asia/Calcutta')).strftime('%Y%m%d_%H%M%S')
    out=ROOT/'stain_research/results'/f'construction_audit_{stamp}';out.mkdir(parents=True,exist_ok=False)
    manifest_hash=digest(folder/'patch_manifest.csv');patient_hash=digest(META/'prospective_grouped_manifest.csv')
    mapping={r['slide_id']:r for r in read_csv(META/'prospective_grouped_manifest.csv')}
    existing=[r for r in read_csv(folder/'patch_manifest.csv') if r['split']=='train']
    receipts=json.loads((folder/'qc_receipts.json').read_text())
    training=[r for r in receipts if r['split']=='train']
    policies=[{'name':'reference','purity':.9,'artifact':.05,'qc':True,'cap':128}]
    policies += [{'name':f'purity_{p}','purity':p,'artifact':.05,'qc':True,'cap':128} for p in [.6,.75,.95]]
    policies += [{'name':f'artifact_{a}','purity':.9,'artifact':a,'qc':True,'cap':128} for a in [0.,.1,.2]]
    policies += [{'name':'quality_block_off','purity':.9,'artifact':.05,'qc':False,'cap':128},
                 {'name':'cap64','purity':.9,'artifact':.05,'qc':True,'cap':64},
                 {'name':'uncapped','purity':.9,'artifact':.05,'qc':True,'cap':None}]
    plan={'policies':policies,'manifest_sha256':manifest_hash,'patient_manifest_sha256':patient_hash,
          'scope':'Training masks only; fixed downloaded slide universe and geometry. Original alignment exclusions retained.',
          'quality_policy':'When enabled: tissue>=.8, unknown<=.05, artifact cutoff as specified. quality_block_off disables all three QC tests together; other alternatives change one scalar rule.',
          'sampling':'Original SHA256 coordinate order, cap per patient/class after selection.',
          'engineering_gate':'Any non-reference alternative has >=10% Jaccard membership churn OR >=.05 class-distribution total variation relative to reference. Descriptive sensitivity threshold, not significance or clinical importance.',
          'claim_limit':'Selection changes cannot establish annotation errors, model-ranking changes or classification benefits.',
          'automatic_training':False,'validation_test_masks_or_RGB_opened':False}
    atomic(out/'PLAN.json',plan);started=time.perf_counter();universe=[];exclusions=[];source_hashes={}
    Image.MAX_IMAGE_PIXELS=None
    for ordinal,r in enumerate(training):
        sid=r['slide_id'];assert mapping[sid]['split']=='train' and mapping[sid]['patient_id']==r['patient_id']
        if r['status']!='screened':exclusions.append({'slide_id':sid,'reason':r['status']});continue
        annotation=folder/'annotations'/f'{sid}.png';quality_path=folder/'quality'/f'{sid}.png'
        source_hashes[sid]={'annotation_sha256':digest(annotation),'quality_sha256':digest(quality_path)}
        labels=labels_for(annotation);width,height=r['dimensions']
        assert abs(width/64-labels.shape[1])<=2 and abs(height/64-labels.shape[0])<=2
        with Image.open(quality_path) as quality:
            assert quality.mode in ['L','P'] and quality.width*quality.height<=1_000_000_000
            assert abs((quality.width/quality.height)/(width/height)-1)<=.005
            quality.load()
            for my in range(0,labels.shape[0]-13,14):
                for mx in range(0,labels.shape[1]-13,14):
                    x,y=mx*64,my*64
                    if x+896>width or y+896>height:continue
                    block=labels[my:my+14,mx:mx+14];counts=np.bincount(block[block>=0],minlength=5)
                    cls=int(counts.argmax());purity=float(counts[cls]/196)
                    if purity<.6:continue
                    box=(int(x/width*quality.width),int(y/height*quality.height),math.ceil((x+896)/width*quality.width),math.ceil((y+896)/height*quality.height))
                    q=np.array(quality.crop(box))
                    universe.append({'slide_id':sid,'patient_id':r['patient_id'],'class_id':cls,'class_code':NAMES[cls],
                         'x_level0':x,'y_level0':y,'purity':purity,'artifact':float(((q>=2)&(q<=6)).mean()),
                         'unknown':float(((q<1)|(q>7)).mean()),'tissue':float((q==1).mean())})
        atomic(out/'progress.json',{'stage':'mask_screening','slide':ordinal+1,'training_slides':len(training),'candidate_patches':len(universe)})
        print(f'Training mask {ordinal+1}/{len(training)}; candidates={len(universe)}',flush=True)
    def select(p):
        groups=defaultdict(list)
        for r in universe:
            if r['purity']<p['purity']:continue
            if p['qc'] and (r['artifact']>p['artifact'] or r['unknown']>.05 or r['tissue']<.8):continue
            groups[(r['patient_id'],r['class_id'])].append(r)
        result=[]
        for group in groups.values():
            ordered=sorted(group,key=lambda r:hashlib.sha256(f"42:{r['slide_id']}:{r['x_level0']}:{r['y_level0']}".encode()).hexdigest())
            result += ordered if p['cap'] is None else ordered[:p['cap']]
        return result
    reference=select(policies[0]);ref_keys={key(r) for r in reference}
    assert ref_keys=={key(r) for r in existing},'Reference reconstruction failed: do not interpret alternatives'
    ref_counts=np.array([sum(r['class_id']==c for r in reference) for c in range(5)])
    summary=[];composition=[];memberships=[]
    for p in policies:
        chosen=select(p);keys={key(r) for r in chosen};counts=np.array([sum(r['class_id']==c for r in chosen) for c in range(5)])
        shares=counts/max(len(chosen),1);tv=float(abs(shares-ref_counts/len(reference)).sum()/2)
        patient_counts=Counter(r['patient_id'] for r in chosen);churn=1-len(keys&ref_keys)/max(len(keys|ref_keys),1)
        summary.append({'policy':p['name'],'patches':len(chosen),'patients':len(patient_counts),'slides':len({r['slide_id'] for r in chosen}),
                        'added_vs_reference':len(keys-ref_keys),'removed_vs_reference':len(ref_keys-keys),'membership_churn':churn,
                        'class_mix_total_variation':tv,'largest_patient_fraction':max(patient_counts.values(),default=0)/max(len(chosen),1)})
        for c,name in enumerate(NAMES):
            subset=[r for r in chosen if r['class_id']==c];pc=Counter(r['patient_id'] for r in subset)
            composition.append({'policy':p['name'],'class':name,'patches':len(subset),'share':float(shares[c]),'patients':len(pc),
                                'largest_patient_share_within_class':max(pc.values(),default=0)/max(len(subset),1),'patient_counts':dict(pc)})
        memberships += [{'policy':p['name'],**r} for r in chosen]
    for name,rows in [('policy_summary.csv',summary),('memberships.csv',memberships),('candidate_universe.csv',universe)]:
        with (out/name).open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gate=any(r['membership_churn']>=.1 or r['class_mix_total_variation']>=.05 for r in summary[1:])
    assert digest(folder/'patch_manifest.csv')==manifest_hash and digest(META/'prospective_grouped_manifest.csv')==patient_hash
    result={'stage':'complete','reference_exactly_reconstructed':True,'reference_training_n':len(reference),'candidate_universe_n':len(universe),
            'policy_summary':summary,'class_patient_composition':composition,'excluded_training_slides':exclusions,
            'source_mask_hashes':source_hashes,'engineering_selection_sensitivity_gate_passed':gate,'elapsed_seconds':time.perf_counter()-started,
            'original_manifests_unchanged':True,'model_inference_or_training_performed':False,'new_validation_test_pixel_access':False,
            'interpretation':'Descriptive change in the selected training cohort; not proof of mislabeled tissue, clinical relevance, changed model rankings, or novelty.'}
    atomic(out/'RESULTS.json',result);atomic(out/'progress.json',{'stage':'complete','selection_sensitivity_gate_passed':gate})
    text=['# Training-only benchmark construction audit','',f"The original {len(reference)}-patch training selection was reconstructed exactly from masks and fixed sampling rules.",
          '', '| Policy | Patches | Patients | Added | Removed | Membership churn | Class-mix TV |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:text.append(f"|{r['policy']}|{r['patches']}|{r['patients']}|{r['added_vs_reference']}|{r['removed_vs_reference']}|{r['membership_churn']:.1%}|{r['class_mix_total_variation']:.4f}|")
    text+=['',f'Predefined selection-sensitivity gate passed: {gate}.', '',result['interpretation'],
          '', 'The slide universe was previously annotation-selected. Changes in purity can change class and patient composition; changes after capping may replace rather than simply add patches. Artifact-threshold alternatives retain the original tissue/unknown tests. The quality-block-off alternative removes those three tests as one block. Original alignment exclusions remain fixed. None of these alternatives is established as clinically correct.',
          '', 'No model outputs, validation/test masks, new RGB images or GPU checks were used. Patient IDs/split metadata were read only to restrict the audit to training. No existing manifests, patches or checkpoints were overwritten.',
          '', 'Next, if worthwhile: freeze a common evaluation universe and patient/class-standardized analysis before inspecting saved model predictions. Study ranking stability separately from sample-selection sensitivity. Do not choose a threshold based on the most favorable accuracy or relabel tissue without expert evidence.']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8');print('REPORT',out,flush=True)

if __name__=='__main__':main()
