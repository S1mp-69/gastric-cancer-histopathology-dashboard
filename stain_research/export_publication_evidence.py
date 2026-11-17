"""Export portable metadata/results only; never raw pixels or model weights."""
import csv,json,hashlib,shutil,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'stain_research/protocols/publication_20261117'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    sources={
      'patient_allocation.csv':ROOT/'data/external/hiesd_metadata/prospective_grouped_manifest.csv',
      'development_patch_manifest.csv':ROOT/'data/external/hiesd_expanded_v3/patch_manifest.csv',
      'final_test_protocol.json':ROOT/'stain_research/runs/final_test_hiesd_v1/PROTOCOL.json',
      'final_test_protocol.sha256':ROOT/'stain_research/runs/final_test_hiesd_v1/PROTOCOL.sha256',
      'HiESD_QC_REPORT.json':ROOT/'data/external/hiesd_expanded_v3/QC_REPORT.json',
      'HiESD_campaign_config.json':ROOT/'stain_research/runs/hiesd_resnet_ablation_v1/campaign_config.json',
      'within_class_protocol.json':ROOT/'stain_research/runs/hiesd_within_class_v1/protocol.json',
      'demo_rehearsal.json':ROOT/'stain_research/runs/demo_20261113/rehearsal.json',
      'preservation.json':ROOT/'stain_research/runs/classifier_local_20261113/preservation_after.json',
    }
    for name,path in sources.items():
        if path.exists():shutil.copy2(path,DEST/name)
    freeze=ROOT/'stain_research/runs/final_test_hiesd_v1/CHECKPOINT_FREEZE.json'
    if freeze.exists():shutil.copy2(freeze,DEST/'CHECKPOINT_FREEZE.json')
    final=ROOT/'data/external/hiesd_final_test_v1/patch_manifest.csv'
    if final.exists():shutil.copy2(final,DEST/'final_test_patch_manifest.csv')
    # Original saved validation probabilities are small numeric research results.
    # The baseline also has a portable copy; include all added controls likewise.
    cam=ROOT/'stain_research/runs/hiesd_resnet_ablation_v1'
    dirs=[ROOT/'stain_research/runs/hiesd_resnet18_seed42_v3']+list(cam.glob('seed*_*'))+list((ROOT/'stain_research/runs/hiesd_within_class_v1').glob('within_seed*'))
    times=[]
    for p in dirs:
        if not (p/'RESULTS.json').exists():continue
        out=ROOT/'stain_research/results'/p.name;out.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','RESULTS.json','history.json','validation_predictions.csv','wrapper_provenance.json']:
            if (p/name).exists():shutil.copy2(p/name,out/name)
        history=json.loads((p/'history.json').read_text(encoding='utf-8'))
        seconds=[r['epoch_seconds'] for r in history]
        times.append({'run':p.name,'epochs':len(history),'mean_train_plus_validation_epoch_seconds':statistics.mean(seconds),
                      'training_images_per_epoch_second_proxy':2543/statistics.mean(seconds),
                      'definition':'2543 training images divided by total train+validation epoch wall time, not pure GPU/loader throughput'})
    (DEST/'OBSERVED_TIMING.json').write_text(json.dumps(times,indent=2),encoding='utf-8')
    (DEST/'MANIFEST.json').write_text(json.dumps({'scope':'Public data identifiers, metadata and numeric results; no raw slides, patch pixels, database, credentials, environment or weights.',
      'files':{p.name:digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='MANIFEST.json'}},indent=2),encoding='utf-8')
    print('Portable protocol and result evidence exported.')
if __name__=='__main__':main()
