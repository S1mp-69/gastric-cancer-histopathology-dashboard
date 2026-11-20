"""Shared deterministic source cache; random stain targets remain uncached."""
import json,random
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from hiesd_pilot import ROOT
from efficient_cache import cached_apply
from controller import strength
from publication_controls import assigned_strengths

CAM=ROOT/'stain_research/runs/hiesd_resnet_ablation_v1'
ARM='standard';SEED=42

def configure(arm,seed):
    global ARM,SEED
    ARM=arm;SEED=seed

class Images(Dataset):
    def __init__(self,records,mean,std,train):
        self.records=records;self.mean=mean;self.std=std;self.train=train;self.epoch=0;self.arm=ARM;self.seed=SEED
        directory=Path(json.loads((CAM/'cache_reference.json').read_text())['directory'])
        self.maps={n:np.load(directory/f'{n}.npy',mmap_mode='r') for n in ['rgb','basis','concentrations','diagnostics','valid']}
        import csv
        with (directory/'index.csv').open(newline='') as stream:index=list(csv.DictReader(stream))
        lookup={str(Path(r['image_path']).resolve()):i for i,r in enumerate(index)}
        self.indices=np.array([lookup[str((ROOT/r['image_path']).resolve())] for r in records])
        assert all(index[i]['split']==('train' if train else 'val') for i in self.indices)
        self.params=json.loads((directory/'train_parameters.json').read_text())
        valid=self.maps['valid'][self.indices]
        self.alpha=np.array([strength(self.maps['diagnostics'][i],self.params['controller']) if self.maps['valid'][i] else 0. for i in self.indices])
        self.valid=valid;self.codes=np.array([r['class_code'] for r in records]);self.assigned_epoch=-1
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        index=int(index)
        i=self.indices[index];image=np.array(self.maps['rgb'][i])
        if self.train and self.arm!='standard':
            if self.arm=='residual_fixed':fixed=1.
            elif self.arm=='uniform_weaker':fixed=self.params['uniform_strength']
            elif self.arm=='adaptive':fixed=None
            elif self.arm=='shuffled':
                if self.assigned_epoch!=self.epoch:
                    self.assigned=assigned_strengths(self.alpha,self.valid,self.codes,self.seed,self.epoch)
                    self.assigned_epoch=self.epoch
                fixed=float(self.assigned[index])
            else:raise ValueError(self.arm)
            # This RNG is separate from the unchanged geometric/RGB augmentation
            # stream. All stain arms share target/intensity draws per image/epoch.
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index,940301]))
            image,_,_=cached_apply(image,self.maps['basis'][i],self.maps['concentrations'][i],self.maps['diagnostics'][i],self.maps['valid'][i],self.params,rng,fixed)
        image=Image.fromarray(image)
        if self.train:
            rng=random.Random(self.seed*10**9+self.epoch*10**6+index)
            if rng.random()<.5:image=F.hflip(image)
            if rng.random()<.5:image=F.vflip(image)
            image=F.rotate(image,90*rng.randrange(4))
            operations=[(F.adjust_brightness,rng.uniform(.9,1.1)),(F.adjust_contrast,rng.uniform(.9,1.1)),
                        (F.adjust_saturation,rng.uniform(.9,1.1)),(F.adjust_hue,rng.uniform(-.02,.02))]
            rng.shuffle(operations)
            for func,value in operations:image=func(image,value)
        return F.normalize(F.to_tensor(image),self.mean,self.std),int(self.records[index]['class_id'])
