"""Post-test review queue from fixed predictions; never changes gold labels."""
import csv,json
from pathlib import Path
import numpy as np
from hiesd_pilot import ROOT,NAMES
def main():
    source=ROOT/'stain_research/results/final_test_hiesd_v1'
    result=source/'RESULTS.json'
    if not result.exists() or json.loads(result.read_text()).get('stage')!='complete':return
    with (ROOT/'data/external/hiesd_final_test_v1/patch_manifest.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    predictions={}
    for arm in ['standard','adaptive','within_class_shuffled']:
        for seed in [42,43,44]:
            with np.load(source/f'seed{seed}_{arm}_predictions.npz') as data:
                assert np.array_equal(data['labels'],[int(r['class_id']) for r in rows])
                predictions[(arm,seed)]=data['probabilities'].argmax(1)
    output=[]
    for i,r in enumerate(rows):
        truth=int(r['class_id']);record={k:r[k] for k in ['patient_id','slide_id','image_path','x_level0','y_level0','class_code']}
        record['interpretation']='Disagreement with released coarse annotation; not an expert-confirmed label error or clinical diagnosis.'
        errors=0
        for arm in ['standard','adaptive','within_class_shuffled']:
            votes=[int(predictions[(arm,s)][i]) for s in [42,43,44]]
            record[arm+'_predictions']=';'.join(NAMES[v] for v in votes)
            record[arm+'_errors']=sum(v!=truth for v in votes);errors+=record[arm+'_errors']
            record[arm+'_seed_disagreement']=len(set(votes))>1
        record['errors_among_nine_fixed_models']=errors;output.append(record)
    output.sort(key=lambda r:(-r['errors_among_nine_fixed_models'],r['patient_id'],r['slide_id'],int(r['x_level0']),int(r['y_level0'])))
    dest=ROOT/'evaluation_package/publication_results'
    with (dest/'EXPERT_REVIEW_QUEUE.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    (dest/'EXPERT_REVIEW_README.md').write_text('''# Error review queue — expert review pending

The CSV records fixed standard/adaptive/within-class predictions for all three seeds on each reserved patch, sorted by model disagreement with the released annotation. It reads saved predictions and metadata only. It does not modify annotations, fit a model, select a new checkpoint or tune thresholds.

A frequent model disagreement is not proof of a bad label: the model may be wrong. Coarse released masks, mixed morphology and broad class definitions require a qualified reviewer to assess any proposed explanation. No local pathology expert has adjudicated these cases. Do not claim corrected ground truth or clinical utility from this queue. Keep the frozen test metrics unchanged; future relabeling would be a separate documented study with independently reviewed labels and a new development/evaluation protocol.

Coordinates and local image paths refer to the public HiESD release. Raw images are not included in this portable CSV. Dataset source: doi:10.1038/s41597-025-05679-1.
''',encoding='utf-8')
    print('Saved post-test expert-review queue; no relabeling or tuning.',flush=True)
if __name__=='__main__':main()
