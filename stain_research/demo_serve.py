"""Run the unchanged application with isolated demo storage, on CPU by default."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    parser.add_argument('--port',type=int,default=8000);parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    folder=ROOT/'stain_research/runs/demo_20261113';folder.mkdir(parents=True,exist_ok=True)
    database=folder/'demo.db';uploads=folder/'uploads';uploads.mkdir(exist_ok=True)
    original_hash=digest(ROOT/'app.db')
    if not database.exists():
        with sqlite3.connect((ROOT/'app.db').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(database) as target:
            source.backup(target)
    os.environ['DATABASE_URL']='sqlite:///'+database.as_posix()
    os.environ['UPLOAD_DIR']=str(uploads)
    if not args.gpu:os.environ['CUDA_VISIBLE_DEVICES']=''
    import uvicorn
    config=uvicorn.Config('app.main:app',host='127.0.0.1',port=args.port,log_level='warning')
    server=uvicorn.Server(config)
    if not args.check:
        print('Isolated demo storage: '+str(database),flush=True);server.run();return
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    base=f'http://127.0.0.1:{args.port}'
    try:
        for attempt in range(120):
            try:
                with urllib.request.urlopen(base+'/api/health',timeout=1) as r:health=json.load(r)
                break
            except Exception:time.sleep(.25)
        else:raise RuntimeError('Demo server did not become ready')
        with urllib.request.urlopen(base+'/api/predictions') as r:before=json.load(r)['summary']['total_predictions']
        import csv
        with (ROOT/'stain_research/runs/split_audit_20261113/manifest_exact_dedup.csv').open() as f:
            example=next(r for r in csv.DictReader(f) if r['split']=='train')
        image=Path(example['image_path']).read_bytes();boundary='GastricResearchDemoBoundary'
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="training_demo.png"\r\nContent-Type: image/png\r\n\r\n'.encode()+image+f'\r\n--{boundary}--\r\n'.encode())
        request=urllib.request.Request(base+'/api/predict',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary},method='POST')
        with urllib.request.urlopen(request,timeout=60) as r:prediction=json.load(r)
        with urllib.request.urlopen(base+'/api/predictions') as r:after=json.load(r)['summary']['total_predictions']
        assert len(prediction['class_probabilities'])==8 and after==before+1
        assert digest(ROOT/'app.db')==original_hash
        result={'health':health,'eight_class_probabilities':True,'sql_history_increment':after-before,
                'model_source':prediction['model_source'],'predicted_class':prediction['predicted_class'],
                'source_partition':'train','original_database_sha256_unchanged':True,
                'demo_database':str(database.relative_to(ROOT)),'device_policy':'CPU rehearsal leaves GPU for research',
                'test_images_used':False,'research_checkpoint_activated':False}
        (folder/'rehearsal.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    finally:
        server.should_exit=True;thread.join(timeout=15)


if __name__=='__main__':main()
