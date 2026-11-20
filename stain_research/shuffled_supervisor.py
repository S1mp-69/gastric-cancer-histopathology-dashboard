"""Independent bounded follow-up queue. No OpenAI calls or dashboard writes."""
import ctypes,itertools,json,os,statistics,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];CAM=ROOT/'stain_research/runs/shuffled_20261116';TZ=ZoneInfo('Asia/Calcutta')
def now():return datetime.now(TZ).isoformat(timespec='seconds')
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default
def save(p,d):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2),encoding='utf-8')
 for i in range(30):
  try:os.replace(temp,p);return
  except PermissionError:time.sleep(.1)
 raise RuntimeError('Cannot publish status '+str(p))
def alive(pid):
 if not pid:return False
 handle=ctypes.windll.kernel32.OpenProcess(0x00100000,False,int(pid))
 if not handle:return False
 try:return ctypes.windll.kernel32.WaitForSingleObject(handle,0)==258
 finally:ctypes.windll.kernel32.CloseHandle(handle)
def others():
 code="Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'overnight_trainer.py|shuffled_classifier.py.*train|training.train' } | Select-Object -ExpandProperty ProcessId"
 r=subprocess.run(['powershell','-NoProfile','-Command',code],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
 return r.stdout.strip()
def status(phase,seed=None):
 rows=[]
 for s in [42,43,44]:
  folder=CAM/f'seed{s}_shuffled';m=read(folder/'metrics.json');p=read(folder/'progress.json',{})
  rows.append({'seed':s,'state':'complete' if m else p.get('status','queued'),'epoch':m['completed_epochs'] if m else p.get('epoch',0),
     'best_macro_f1':m['validation']['macro_f1'] if m else p.get('best_macro_f1'),'updated_at':p.get('updated_at')})
 save(CAM/'FOLLOWUP_STATUS.json',{'updated_at':now(),'phase':phase,'active_seed':seed,'runs':rows,'test_evaluated':False})
def aggregate():
 rows=[]
 for seed in [42,43,44]:
  new=read(CAM/f'seed{seed}_shuffled/metrics.json');old=read(ROOT/f'stain_research/runs/overnight_20261113/seed{seed}_adaptive/metrics.json')
  assert new['initial_sha256']==old['initial_sha256'] and new['checkpoint_reload_verified'] and not new['test_evaluated']
  rows.append({'seed':seed,'adaptive_f1':old['validation']['macro_f1'],'shuffled_f1':new['validation']['macro_f1'],
   'adaptive_minus_shuffled_f1':old['validation']['macro_f1']-new['validation']['macro_f1'],
   'adaptive_tum_recall':old['validation']['per_class']['TUM']['recall'],'shuffled_tum_recall':new['validation']['per_class']['TUM']['recall'],
   'epochs':new['completed_epochs'],'stop_reason':new['stop_reason']})
 diffs=[r['adaptive_minus_shuffled_f1'] for r in rows];mean=statistics.mean(diffs);sd=statistics.stdev(diffs);half=4.302652729911275*sd/3**.5
 result={'completed_at':now(),'rows':rows,'mean_paired_f1_difference':mean,'paired_sd':sd,'exploratory_95_t_interval':[mean-half,mean+half],
  'limitations':'Post-hoc follow-up; n=3 seeds; repeated validation-selected checkpoints; image-level split; patient independence unverified. No novelty or clinical benefit established. Interval assumes approximately normal independent seed differences.'}
 out=ROOT/'stain_research/results/shuffled_20261116';out.mkdir(exist_ok=True);save(out/'summary.json',result)
 text='# Shuffled-strength follow-up\n\nCompleted '+now()+'\n\n| Seed | Adaptive F1 | Shuffled F1 | Difference |\n|---|---:|---:|---:|\n'
 for r in rows:text+=f"| {r['seed']} | {r['adaptive_f1']:.4f} | {r['shuffled_f1']:.4f} | {r['adaptive_minus_shuffled_f1']:+.4f} |\n"
 text+=f"\nMean difference {mean:+.4f}; exploratory 95% t interval [{mean-half:+.4f}, {mean+half:+.4f}].\n\n"+result['limitations']+'\n'
 (out/'RESULTS.md').write_text(text,encoding='utf-8')
def main():
 if not (CAM/'preflight.json').exists():raise RuntimeError('Preflight required')
 lock=CAM/'supervisor.lock.json';old=read(lock,{})
 if alive(old.get('pid')):return
 if lock.exists():lock.unlink()
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'started_at':now()},f)
 ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
 try:
  started=time.time()
  for seed in [42,43,44]:
   folder=CAM/f'seed{seed}_shuffled';folder.mkdir(exist_ok=True)
   if (folder/'metrics.json').exists():continue
   prior=read(folder/'process.json',{})
   while alive(prior.get('pid')):status('monitoring_existing_worker',seed);time.sleep(30)
   if (folder/'metrics.json').exists():continue
   while others():status('waiting_for_existing_training',seed);time.sleep(30)
   for attempt in range(3):
    if time.time()-started>8*3600:status('queue_time_budget_reached');return
    with (folder/'worker.log').open('a',encoding='utf-8') as log:
     child=subprocess.Popen([sys.executable,'-u',str(ROOT/'stain_research/shuffled_classifier.py'),'train','--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
     while child.poll() is None:status('training',seed);time.sleep(30)
    if (folder/'metrics.json').exists():break
    save(folder/'failure.json',{'attempt':attempt+1,'exit_code':child.returncode,'at':now()});time.sleep(5)
   else:status('technical_failure_requires_attention',seed);return
  aggregate();status('complete')
 except Exception:
  save(CAM/'supervisor_failure.json',{'at':now(),'traceback':traceback.format_exc()});raise
 finally:
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
  if read(lock,{}).get('pid')==os.getpid():lock.unlink(missing_ok=True)
if __name__=='__main__':main()
