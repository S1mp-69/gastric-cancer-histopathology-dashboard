"""Publication data-flow counts from frozen metadata; no pixels or tuning."""
import csv,json
from collections import Counter
from pathlib import Path
from hiesd_pilot import ROOT,digest
def csvrows(p):
    with p.open(newline='') as stream:return list(csv.DictReader(stream))
def main():
    allocation=csvrows(ROOT/'data/external/hiesd_metadata/prospective_grouped_manifest.csv')
    development=csvrows(ROOT/'data/external/hiesd_expanded_v3/patch_manifest.csv')
    final=ROOT/'data/external/hiesd_final_test_v1/patch_manifest.csv'
    if not final.exists():return
    test=csvrows(final);rows=development+test;summary={}
    for split in ['train','validation','reserved_test']:
        items=[r for r in rows if r['split']==split]
        summary[split]={'patches':len(items),'patients':len({r['patient_id'] for r in items}),'slides_with_sampled_patches':len({r['slide_id'] for r in items}),
                        'class_counts':dict(Counter(r['class_code'] for r in items))}
    result={'public_release':{'patients':44,'slides':104},'allocated_patients':{s:len({r['patient_id'] for r in allocation if r['split']==s}) for s in ['train','validation','reserved_test']},
       'sampled_cohorts':summary,'development_manifest_sha256':digest(ROOT/'data/external/hiesd_expanded_v3/patch_manifest.csv'),'final_test_manifest_sha256':digest(final),
       'limitations':'Annotation/QC-selected subsets; allocated patients differ from sampled coverage. Coarse masks and small rare-class supports; public-release holdout, not independent cross-hospital validation.'}
    out=ROOT/'evaluation_package/publication_results';out.mkdir(exist_ok=True)
    (out/'DATA_FLOW.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text='# HiESD data flow\n\nThe public release contains 104 slides from 44 patients. Allocation is 30 training, seven validation and seven reserved-test patients. Selected/QC-passing sampled coverage is smaller than allocation and must not be presented as the whole release.\n\n| Cohort | Sampled patches | Patients | Slides represented |\n|---|---:|---:|---:|\n'
    for split,r in summary.items():text+=f"| {split} | {r['patches']} | {r['patients']} | {r['slides_with_sampled_patches']} |\n"
    text+='\nClass supports:\n\n| Group | Training | Validation | Reserved test |\n|---|---:|---:|---:|\n'
    for c in ['tumor','mucosa_inflammation','intestinal_metaplasia','lymphoid','other']:text+='| '+c+' | '+' | '.join(str(summary[s]['class_counts'].get(c,0)) for s in summary)+' |\n'
    text+='\n'+result['limitations']+' QC receipts identify each source alignment check, exclusion and artifact-screening result. No test cases were reassigned to improve these counts.\n'
    (out/'DATA_FLOW.md').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
