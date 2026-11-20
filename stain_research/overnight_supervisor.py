"""Durable campaign supervisor and 09:00 IST snapshot writer.

Runs outside Codex's tool session, retains restartable worker checkpoints, and
continues the bounded campaign after the morning snapshot. No dashboard writes.
"""
import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
TZ=ZoneInfo('Asia/Calcutta')
ARMS=['standard','residual_fixed','uniform_weaker','adaptive']
_STATUS_LOCK=threading.RLock()
_CONTEXT={'active':None,'phase':'running'}


def timestamp(): return datetime.now(TZ).isoformat(timespec='seconds')


def read(path,default=None):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError): return default


def write(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temporary,path)


def alive(pid):
    if not pid: return False
    handle=ctypes.windll.kernel32.OpenProcess(0x00100000,False,int(pid))
    if not handle: return False
    try: return ctypes.windll.kernel32.WaitForSingleObject(handle,0)==258
    finally: ctypes.windll.kernel32.CloseHandle(handle)


def related_jobs():
    code="Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'overnight_trainer.py|training.train|training/train.py|classifier_pilot.py' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    try:
        r=subprocess.run(['powershell','-NoProfile','-Command',code],capture_output=True,text=True,timeout=10,
                          creationflags=subprocess.CREATE_NO_WINDOW)
        result=json.loads(r.stdout) if r.stdout.strip() else []
        return result if isinstance(result,list) else [result]
    except Exception: return []


def resource_sample():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=6,
                         creationflags=subprocess.CREATE_NO_WINDOW)
        return r.stdout.strip()
    except Exception as e: return str(e)


def _status_snapshot(campaign,config,active=None,phase='running'):
    rows=[]
    for seed in config['seeds']:
        for arm in ARMS:
            folder=campaign/f'seed{seed}_{arm}'
            metrics=read(folder/'metrics.json')
            progress=read(folder/'progress.json',{})
            history=read(folder/'history.json',[])
            best=max(history,key=lambda r:r['val']['macro_f1']) if history else None
            rows.append({'arm':arm,'seed':seed,'status':'complete' if metrics else progress.get('status','queued'),
                         'epoch':metrics['completed_epochs'] if metrics else progress.get('epoch',0),
                         'best_epoch':metrics['best_epoch'] if metrics else (best['epoch'] if best else None),
                         'validation':metrics['validation'] if metrics else (best['val'] if best else None),
                         'progress':progress,'metrics':metrics})
    state={'snapshot_at':timestamp(),'phase':phase,'supervisor_pid':os.getpid(),'active_run':active,
           'runs':rows,'config':config,'gpu_telemetry_csv':resource_sample(),
           'evaluation_package_ready':(ROOT/'evaluation_package/PACKAGE_READY.json').exists()}
    write(campaign/'campaign_state.json',state)
    lines=['# Morning research status','',f"Updated {state['snapshot_at']} (Asia/Calcutta).",
           '', 'Progress evaluation: 13 November 2026, 11:00 AM, B-305.',
           'Final evaluation: 19 November 2026, 11:00 AM. Final venue has not been independently verified.',
           '',f"Campaign phase: {phase}. Active run: {active or 'none'}. Supervisor PID: {os.getpid()}.",
           'The 09:00 snapshot is a reporting milestone, not a training stop time.', '',
           '## Actual validation progress','',
           '| Arm | Seed | State | Current/completed epoch | Best epoch | Best validation accuracy | Best macro-F1 |',
           '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        v=row['validation']
        lines.append(f"| {row['arm']} | {row['seed']} | {row['status']} | {row['epoch']} | {row['best_epoch'] or '-'} | "+
                     (f"{v['accuracy']:.4f} | {v['macro_f1']:.4f} |" if v else '- | - |'))
    lines+=['','These are best-so-far validation selections for active runs. Completion and convergence are separate from epoch count. No final test inference has run.','',
            '## Class-specific validation errors','']
    for row in rows:
        v=row['validation']
        if v:
            errors={c:int(round(v['per_class'][c]['support']*(1-v['per_class'][c]['recall']))) for c in v['per_class']}
            lines.append(f"{row['arm']} seed {row['seed']} validation false-negative counts by true class: "+json.dumps(errors))
    lines+=['',
            '## Active run and remaining time','']
    if active:
        folder=campaign/active
        progress=read(folder/'progress.json',{})
        history=read(folder/'history.json',[])
        lines.append('Latest worker progress: '+json.dumps({k:progress.get(k) for k in ['status','epoch','items_in_epoch','total_train_images','updated_at']}))
        seconds=sum(r['epoch_seconds'] for r in history)/len(history) if history else config['estimated_epoch_seconds']
        done=len(history)
        lines.append(f"Observed/benchmark-based epoch estimate: {seconds/60:.2f} minutes. Remaining to the 40-epoch ceiling: {max(0,40-done)*seconds/60:.1f} minutes; early stopping can finish earlier.")
        lines.append('Current per-class extraction fallbacks: '+json.dumps(progress.get('failures_by_class',{})))
    completed=sum(row['status']=='complete' for row in rows)
    lines.append(f"Completed runs: {completed}/{len(rows)}. Full remaining-queue upper-budget estimate: {(len(rows)-completed)*40*config['estimated_epoch_seconds']/3600:.2f} hours, refined as epochs complete. Replications start only where the 24-hour campaign budget permits.")
    lines+=['','## Compute and limits','',
            f"Selected batch size {config['batch_size']}, loader workers {config['workers']}, AMP={config['amp']}. Main CPU threads 2, each worker 1.",
            f"GPU sample (utilization %, used MiB, temperature C, power W): {state['gpu_telemetry_csv']}",
            'Versioned float64 decompositions and diagnostics are cached. Random targets, intensity factors and geometry remain stochastic by epoch and image.',
            'Candidate split: 23,761 training / 3,059 validation / 3,090 held-out test files. Exact-file duplicates are controlled. Patient/slide mapping and complete near-duplicate freedom remain unverified.',
            'Historical three-epoch pilot: adaptive macro-F1 0.2651, uniform weaker 0.2802, shuffled 0.2768. Those values are separate from this campaign.',
            'Preprocessing stability results measure synthetic-noise response, not classification accuracy. Historical notebook metrics are a separate source.', '',
            '## Evaluation deliverables','',
            f"Package ready marker: {state['evaluation_package_ready']}.",
            '- evaluation_package/Progress_Presentation.pptx',
            '- evaluation_package/Progress_Presentation.pdf',
            '- evaluation_package/Progress_Report.docx',
            '- evaluation_package/Progress_Report.pdf',
            '- evaluation_package/Overleaf_Paper_Source.zip',
            '- evaluation_package/Presentation_Script_and_Viva.md',
            '- evaluation_package/DEMO_CHECKLIST.md',
            '- evaluation_package/OVERLEAF_STATUS.md',
            'The report format is provisional because no prescribed institutional template was provided. Only the supplied short author name Amlan is verified.', '',
            '## Resuming after interruption','',
            'Supervisor logs, campaign_state.json, each run/history.json, last.pt and best.pt remain on disk. The last checkpoint includes optimizer, scheduler, scaler, RNG states and mid-epoch position.',
            'Restart command: `.venv-research/Scripts/python.exe -u stain_research/overnight_supervisor.py --campaign stain_research/runs/overnight_20261113`.',
            'A live-supervisor lock prevents launching a second supervisor. Original application, database, data and model activation are not written by this campaign.']
    text='\n'.join(lines)+'\n'
    target=ROOT/'MORNING_STATUS.md'; temp=target.with_suffix('.md.tmp');temp.write_text(text,encoding='utf-8')
    try:
        os.replace(temp,target)
    except PermissionError:
        # PowerShell Get-Content -Wait permits writing but holds a Windows
        # handle that prevents replacing the watched file. Keep it watchable.
        try:
            target.write_text(text,encoding='utf-8')
            temp.unlink(missing_ok=True)
        except PermissionError as exc:
            # A status viewer must never terminate the training scheduler.
            print('Status file temporarily locked: '+str(exc),flush=True)
    deadline=datetime.fromisoformat(config['morning_snapshot_due'])
    frozen=ROOT/'evaluation_package/MORNING_STATUS_20261113_0900.md'
    if datetime.now(TZ)>=deadline-timedelta(minutes=1) and not frozen.exists():
        frozen.write_text(text,encoding='utf-8')
        write(campaign/'morning_snapshot_receipt.json',{'scheduled_for':config['morning_snapshot_due'],'written_at':timestamp(),'prepared_by_deadline':datetime.now(TZ)<=deadline,'path':str(frozen),'campaign_continues':True})
    return state


def status_snapshot(campaign,config,active=None,phase='running'):
    with _STATUS_LOCK:
        _CONTEXT.update(active=active,phase=phase)
        return _status_snapshot(campaign,config,active,phase)


def background_status(campaign,config,stop):
    # This local thread makes no OpenAI calls. It keeps the deadline independent
    # of a slow Office/PDF refresh or a worker restart.
    while not stop.is_set():
        try:
            with _STATUS_LOCK:
                _status_snapshot(campaign,config,_CONTEXT['active'],_CONTEXT['phase'])
        except Exception as exc:
            print('Status writer error: '+str(exc),flush=True)
        until=(datetime.fromisoformat(config['morning_snapshot_due'])-timedelta(minutes=1)-datetime.now(TZ)).total_seconds()
        stop.wait(min(30,max(.5,until)) if until>0 else 30)


def refresh_package(campaign):
    script=ROOT/'stain_research/refresh_evaluation.py'
    if not script.exists() or not (ROOT/'evaluation_package/PACKAGE_READY.json').exists(): return
    with (campaign/'package_refresh.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([str(ROOT/'.venv-research/Scripts/python.exe'),str(script),'--campaign',str(campaign)],
                              cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    write(campaign/'last_package_refresh.json',{'at':timestamp(),'exit_code':result.returncode})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args();campaign=args.campaign.resolve();campaign.mkdir(parents=True,exist_ok=True)
    lock=campaign/'supervisor.lock.json'
    existing=read(lock,{})
    if alive(existing.get('pid')):
        print('Existing live supervisor detected; no duplicate launched.',flush=True);return
    if lock.exists(): lock.unlink()
    try:
        descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Another supervisor acquired the campaign lock')
    with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
        json.dump({'pid':os.getpid(),'started_at':timestamp()},stream)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    status_stop=threading.Event()
    try:
        config=read(campaign/'campaign_config.json')
        if config is None:
            benchmark=read(campaign/'throughput_benchmark.json')
            if not benchmark: raise RuntimeError('Complete throughput benchmarking first')
            selected=benchmark['selected']
            local=datetime.now(TZ);due=local.replace(hour=9,minute=0,second=0,microsecond=0)
            if local>=due: due+=timedelta(days=1)
            config={'created_at':timestamp(),'morning_snapshot_due':due.isoformat(),
                    'campaign_wall_hours':24,'seeds':[42,43,44],'arms':ARMS,'max_epochs':40,'patience':7,'min_delta':.0001,
                    'batch_size':selected['batch_size'],'workers':selected['workers'],'amp':True,
                    'learning_rate':.001,'weight_decay':.0001,'checkpoint_seconds':120,
                    'estimated_epoch_seconds':23761/selected['images_per_second']+15,
                    'early_stopping':'Validation macro-F1, improvement >0.0001, patience7; common 40-epoch ceiling',
                    'scheduler':'ReduceLROnPlateau validation loss, patience2, factor0.5',
                    'evaluation':'Raw resized RGB, train-fitted channel standardization, deterministic AMP validation; no stain normalization or test-time augmentation',
                    'split':'Provisional exact-file-controlled historical image split; patient independence UNVERIFIED',
                    'comparison':'Same split, initial model per seed, geometry, order, optimizer and stopping rules',
                    'authorization':'User instruction on 13 November explicitly permits exploratory provisional-split training despite unavailable patient mapping',
                    'source_versions':{n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','pandas','Pillow']},
                    'dashboard_activation':False,'test_evaluation':False}
            write(campaign/'campaign_config.json',config)
        threading.Thread(target=background_status,args=(campaign,config,status_stop),daemon=True).start()
        started=datetime.fromisoformat(config['created_at']);last_refresh=0;failures=read(campaign/'worker_failures.json',{})
        for seed in config['seeds']:
            for arm in ARMS:
                name=f'seed{seed}_{arm}';folder=campaign/name
                if (folder/'metrics.json').exists(): continue
                if (datetime.now(TZ)-started).total_seconds()>config['campaign_wall_hours']*3600:
                    status_snapshot(campaign,config,None,'bounded_time_budget_complete');return
                if not (seed==42 and arm=='standard'):
                    while not (ROOT/'evaluation_package/PACKAGE_READY.json').exists():
                        status_snapshot(campaign,config,None,'waiting_for_evaluation_materials_before_extra_runs');time.sleep(30)
                prior=read(folder/'process.json',{})
                if alive(prior.get('pid')):
                    while alive(prior['pid']):
                        status_snapshot(campaign,config,name,'monitoring_existing_worker');time.sleep(30)
                    if (folder/'metrics.json').exists(): continue
                while related_jobs():
                    status_snapshot(campaign,config,None,'waiting_for_existing_related_training');time.sleep(30)
                folder.mkdir(exist_ok=True)
                while not (folder/'metrics.json').exists():
                    if failures.get(name,0)>=3:
                        status_snapshot(campaign,config,name,'technical_failure_requires_attention');return
                    with (folder/'worker.log').open('a',encoding='utf-8') as log:
                        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'stain_research/overnight_trainer.py'),'train',
                                '--campaign',str(campaign),'--arm',arm,'--seed',str(seed)],cwd=ROOT,
                                stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                        write(campaign/'active_worker.json',{'pid':child.pid,'run':name,'started_at':timestamp()})
                        while child.poll() is None:
                            status_snapshot(campaign,config,name)
                            if time.time()-last_refresh>=1800:
                                refresh_package(campaign);last_refresh=time.time()
                            time.sleep(30)
                        if child.returncode!=0:
                            failures[name]=failures.get(name,0)+1;write(campaign/'worker_failures.json',failures)
                            print(f'Worker {name} failed ({child.returncode}); restart attempt {failures[name]}/3',flush=True)
                            time.sleep(15)
                status_snapshot(campaign,config,None)
                refresh_package(campaign);last_refresh=time.time()
        status_snapshot(campaign,config,None,'bounded_campaign_complete')
        refresh_package(campaign)
        # If every bounded run finishes before the requested snapshot, remain a
        # lightweight writer until 09:00. No extra training is invented.
        while datetime.now(TZ)<datetime.fromisoformat(config['morning_snapshot_due']):
            status_snapshot(campaign,config,None,'bounded_campaign_complete_waiting_for_0900_snapshot');time.sleep(30)
        status_snapshot(campaign,config,None,'bounded_campaign_complete')
    except Exception:
        write(campaign/'supervisor_failure.json',{'at':timestamp(),'traceback':traceback.format_exc()})
        traceback.print_exc();raise
    finally:
        status_stop.set()
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if read(lock,{}).get('pid')==os.getpid(): lock.unlink(missing_ok=True)


if __name__=='__main__': main()
