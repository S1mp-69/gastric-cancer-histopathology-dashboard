"""Training-count-weighted follow-up of frozen HiESD baseline; no test access."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[k]='1'
import argparse
import csv
import hashlib
import json
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import functional as F
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from hiesd_pilot import ROOT,NAMES,digest
sys.path.insert(0,str(ROOT))
from app.ml.model import GastricCancerRNN
from app.ml.preprocessing import image_to_patch_sequence

def now():return datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(timespec='seconds')

def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(20):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def checkpoint(path,value):
    temp=path.with_suffix('.tmp');torch.save(value,temp)
    for attempt in range(20):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

class Images(Dataset):
    def __init__(self,records,mean,std,train):
        self.records=records;self.mean=mean;self.std=std;self.train=train;self.epoch=0
        self.rgb=[]
        for r in records:
            with Image.open(ROOT/r['image_path']) as im:
                self.rgb.append(np.array(im.convert('RGB')))
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        image=Image.fromarray(self.rgb[index])
        if self.train:
            rng=random.Random(42*10**9+self.epoch*10**6+index)
            if rng.random()<.5:image=F.hflip(image)
            if rng.random()<.5:image=F.vflip(image)
            image=F.rotate(image,90*rng.randrange(4))
            operations=[(F.adjust_brightness,rng.uniform(.9,1.1)),(F.adjust_contrast,rng.uniform(.9,1.1)),
                        (F.adjust_saturation,rng.uniform(.9,1.1)),(F.adjust_hue,rng.uniform(-.02,.02))]
            rng.shuffle(operations)
            for func,value in operations:image=func(image,value)
        return image_to_patch_sequence(image,56,16,tuple(self.mean),tuple(self.std)),int(self.records[index]['class_id'])

def evaluate(model,loader):
    model.eval();probabilities=[];truth=[];nll=0.;count=0
    with torch.inference_mode():
        for x,y in loader:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):logits,_=model(x)
            assert torch.isfinite(logits).all()
            nll+=float(torch.nn.functional.cross_entropy(logits.float(),y,reduction='sum'))
            probabilities.append(logits.float().softmax(1).cpu().numpy());truth.extend(y.cpu().tolist());count+=len(y)
    prob=np.concatenate(probabilities);pred=prob.argmax(1)
    report=classification_report(truth,pred,labels=list(range(5)),target_names=NAMES,output_dict=True,zero_division=0)
    return {'accuracy':float(accuracy_score(truth,pred)),'macro_f1':float(f1_score(truth,pred,labels=list(range(5)),average='macro',zero_division=0)),
            'nll':nll/count,'classification_report':report,'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(5))).tolist()},prob,np.array(truth)

def main(run,prepare,resume):
    run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback permitted')
    manifest=ROOT/'data/external/hiesd_pilot_v2/patch_manifest.csv'
    with manifest.open(newline='',encoding='utf-8') as stream:records=list(csv.DictReader(stream))
    with (ROOT/'data/external/hiesd_metadata/prospective_grouped_manifest.csv').open(newline='') as stream:
        mapping={r['slide_id']:r for r in csv.DictReader(stream)}
    assert all(r['split'] in ['train','validation'] and mapping[r['slide_id']]['split']==r['split'] and mapping[r['slide_id']]['patient_id']==r['patient_id'] for r in records)
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='validation']
    assert not {r['patient_id'] for r in train}&{r['patient_id'] for r in val}
    sums=np.zeros(3);squares=np.zeros(3);pixels=0
    for r in train:
        with Image.open(ROOT/r['image_path']) as im:a=np.asarray(im,dtype=np.float64)/255
        sums+=a.sum((0,1));squares+=(a*a).sum((0,1));pixels+=a.shape[0]*a.shape[1]
    mean=sums/pixels;std=np.sqrt(np.maximum(squares/pixels-mean*mean,1e-8))
    class_counts=np.bincount([int(r['class_id']) for r in train],minlength=5)
    assert np.all(class_counts>0)
    class_weights=len(train)/(5*class_counts.astype(np.float64))
    class_weights_tensor=torch.tensor(class_weights,dtype=torch.float32,device='cuda')
    config={'seed':42,'arm':'standard_class_weighted','class_counts_train':class_counts.tolist(),'class_weights':class_weights.tolist(),'weight_rule':'N_train/(5*n_class); training labels only; PyTorch weighted label-smoothed CE','max_epochs':40,'early_stopping_patience':7,'minimum_f1_improvement':.0001,
            'batch_size':32,'loader_workers':0,'cpu_threads':2,'amp':'FP16','optimizer':'AdamW','learning_rate':.001,
            'weight_decay':.0001,'label_smoothing':.05,'gradient_clip':5,'scheduler':'ReduceLROnPlateau validation NLL, factor .5, patience 2',
            'class_order':NAMES,'model':'fresh CNN-BiLSTM-attention with five outputs; no HMU checkpoint transfer',
            'normalization_mean':mean.tolist(),'normalization_std':std.tolist(),'normalization_fit_split':'train only',
            'augmentation':'train H/V flips p=.5, right-angle rotation, brightness/contrast/saturation .9-1.1, hue +/-.02, shuffled operation order',
            'manifest_sha256':digest(manifest),'train_n':len(train),'validation_n':len(val),
            'source_sha256':digest(Path(__file__)),'model_source_sha256':digest(ROOT/'app/ml/model.py'),
            'limitations':['Annotation-guided selected subset','Small patient counts; 12 validation lymphoid patches','Coarse mask boundaries without expert review',
                           'Five-group semantics verified against HiESD Figure 1/Table 2; geometry/model/protocol differ from published benchmark'],
            'test_policy':'Reserved-test images, annotations and predictions are not accessed.'}
    config_path=run/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=config:raise RuntimeError('Frozen configuration differs')
    else:atomic(config_path,config)
    seed_all(42)
    trainset=Images(train,mean,std,True);valset=Images(val,mean,std,False)
    def train_loader(epoch):
        trainset.epoch=epoch
        return DataLoader(trainset,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(42+epoch*100003),num_workers=0,pin_memory=True)
    val_loader=DataLoader(valset,batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
    model=GastricCancerRNN(num_classes=5).cuda()
    baseline=ROOT/'stain_research/runs/hiesd_baseline_seed42_20261116'
    baseline_config=json.loads((baseline/'config.json').read_text())
    for key in baseline_config:
        if key not in ['arm','source_sha256']:
            assert config[key]==baseline_config[key],f'Unmatched configuration: {key}'
    if not (run/'initial.pt').exists():(run/'initial.pt').write_bytes((baseline/'initial.pt').read_bytes())
    assert digest(run/'initial.pt')==digest(baseline/'initial.pt')
    initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(initial['model_state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',init_scale=4096)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=2)
    def train_epoch(epoch,limit=None):
        model.train();n=0;total=0.;backoffs=0;start=time.perf_counter()
        for batch_index,(x,y) in enumerate(train_loader(epoch)):
            if limit is not None and batch_index>=limit:break
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits,_=model(x);loss=torch.nn.functional.cross_entropy(logits,y,weight=class_weights_tensor,label_smoothing=.05)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            previous=scaler.get_scale();scaler.step(optimizer);scaler.update();backoffs+=int(scaler.get_scale()<previous)
            n+=len(y);total+=float(loss.detach())*len(y)
        torch.cuda.synchronize()
        return {'loss':total/n,'images_per_second':n/(time.perf_counter()-start),'seconds':time.perf_counter()-start,'amp_backoffs':backoffs}
    if prepare:
        atomic(run/'progress.json',{'stage':'benchmarking','updated':now(),'max_epochs':40})
        warm=train_epoch(0,limit=2);bench=train_epoch(0,limit=6)
        start=time.perf_counter();_,_,_=evaluate(model,val_loader);torch.cuda.synchronize();val_seconds=time.perf_counter()-start
        free,_=torch.cuda.mem_get_info();estimated=len(train)/bench['images_per_second']+val_seconds
        atomic(run/'benchmark.json',{'warmup':warm,'measured':bench,'validation_seconds':val_seconds,'estimated_epoch_seconds':estimated,
                                    'estimated_40_epoch_minutes':40*estimated/60,'free_vram_MiB':free/1024**2,
                                    'gpu':torch.cuda.get_device_name(),'initial_checkpoint_sha256':digest(run/'initial.pt'),
                                    'scope':'Dry-run timings only; model/optimizer discarded before actual training. Checkpoint/report overhead and thermal changes may increase runtime.'})
        atomic(run/'progress.json',{'stage':'prepared','updated':now(),'max_epochs':40,'estimated_40_epoch_minutes':40*estimated/60})
        print((run/'benchmark.json').read_text(),flush=True);return
    if not (run/'benchmark.json').exists():raise RuntimeError('Prepare benchmark before starting pilot')
    start_epoch=1;best=-1.;bad=0;history=[]
    if resume:
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if state['manifest_sha256']!=config['manifest_sha256']:raise RuntimeError('Resume manifest differs')
        model.load_state_dict(state['model_state_dict']);optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict']);scaler.load_state_dict(state['scaler_state_dict'])
        torch.set_rng_state(state['torch_rng_state']);torch.cuda.set_rng_state_all(state['cuda_rng_state'])
        random.setstate(state['python_rng_state']);np.random.set_state(state['numpy_rng_state'])
        start_epoch=state['epoch']+1;best=state['best_f1'];bad=state['bad_epochs'];history=state['history']
    elif (run/'last.pt').exists():raise RuntimeError('Run already has state; use --resume')
    else:seed_all(42)
    started=time.perf_counter();stop='max_epochs'
    for epoch in range(start_epoch,41):
        if bad>=7:stop='early_stopping';break
        atomic(run/'progress.json',{'stage':'training','epoch':epoch,'max_epochs':40,'updated':now(),'best_validation_macro_f1':best,'completed_epochs':len(history)})
        epoch_start=time.perf_counter();training=train_epoch(epoch)
        metrics,prob,truth=evaluate(model,val_loader);scheduler.step(metrics['nll'])
        improved=metrics['macro_f1']>best+.0001
        if improved:best=metrics['macro_f1'];bad=0
        else:bad+=1
        history.append({'epoch':epoch,'train':training,'validation':metrics,'epoch_seconds':time.perf_counter()-epoch_start,'learning_rate':optimizer.param_groups[0]['lr']})
        state={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
               'scaler_state_dict':scaler.state_dict(),'epoch':epoch,'best_f1':best,'bad_epochs':bad,'history':history,'manifest_sha256':config['manifest_sha256'],
               'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all(),'python_rng_state':random.getstate(),'numpy_rng_state':np.random.get_state()}
        checkpoint(run/'last.pt',state)
        if improved:
            checkpoint(run/'best.pt',state);np.savez_compressed(run/'best_validation_predictions.npz',probabilities=prob,truth=truth)
        atomic(run/'history.json',history)
        atomic(run/'progress.json',{'stage':'epoch_complete','epoch':epoch,'max_epochs':40,'updated':now(),'validation':metrics,
                                   'best_validation_macro_f1':best,'bad_epochs':bad,'epoch_seconds':history[-1]['epoch_seconds'],
                                   'estimated_remaining_minutes':(40-epoch)*np.mean([h['epoch_seconds'] for h in history])/60})
        print(f"Epoch {epoch}/40 val F1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f} best={best:.4f} patience={bad}/7",flush=True)
        if bad>=7:stop='early_stopping';break
    best_state=torch.load(run/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(best_state['model_state_dict'])
    metrics,prob,truth=evaluate(model,val_loader)
    saved=np.load(run/'best_validation_predictions.npz')
    if not np.allclose(prob,saved['probabilities'],atol=1e-6,rtol=1e-5):raise RuntimeError('Reloaded best predictions mismatch')
    with (run/'validation_predictions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['image_path','patient_id','slide_id','class_id','predicted_class_id']+[f'p_{n}' for n in NAMES])
        for r,p,t in zip(val,prob,truth):writer.writerow([r['image_path'],r['patient_id'],r['slide_id'],int(t),int(p.argmax())]+p.tolist())
    result={'stage':'complete','updated':now(),'completed_epochs':len(history),'best_epoch':best_state['epoch'],'stop_reason':stop,'validation':metrics,
            'best_checkpoint_sha256':digest(run/'best.pt'),'reload_predictions_match':True,'elapsed_this_process_seconds':time.perf_counter()-started,
            'reserved_test_used':False,'limitations':config['limitations'],'run_directory':str(run)}
    atomic(run/'RESULTS.json',result);atomic(run/'progress.json',result)
    portable=ROOT/'stain_research/results'/run.name;portable.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','benchmark.json','history.json','RESULTS.json']:
        (portable/name).write_bytes((run/name).read_bytes())
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--prepare',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try:main(args.run.resolve(),args.prepare,args.resume)
    except Exception as exc:
        args.run.mkdir(parents=True,exist_ok=True)
        atomic(args.run/'FAILURE.json',{'error':str(exc),'time':now(),'traceback':traceback.format_exc()})
        raise
