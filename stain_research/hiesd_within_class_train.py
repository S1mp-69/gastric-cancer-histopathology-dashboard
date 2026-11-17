"""Isolated within-class permutation control using frozen trainer/data bytes."""
import argparse,json
from pathlib import Path
import hiesd_ablation_train as trainer
import hiesd_ablation_data as data
from publication_controls import assigned_strengths

def permute(alpha,valid,codes,seed,epoch):return assigned_strengths(alpha,valid,codes,seed,epoch,within_class=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43,44],required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    data.assigned_strengths=permute;data.configure('shuffled',args.seed)
    trainer.ARM='within_class_shuffled';trainer.SEED=args.seed
    args.run.mkdir(parents=True,exist_ok=True)
    trainer.atomic(args.run/'wrapper_provenance.json',{'wrapper_sha256':trainer.digest(Path(__file__)),'trainer_sha256':trainer.digest(Path(trainer.__file__)),'data_sha256':trainer.digest(Path(data.__file__)),
         'assignment':'Epoch-wise permutation within training tissue classes; source RGB/basis/H stay attached; validation unchanged','seed':args.seed})
    try:trainer.main(args.run.resolve(),False,args.resume)
    except Exception as exc:
        trainer.atomic(args.run/'FAILURE.json',{'error':str(exc)});raise
