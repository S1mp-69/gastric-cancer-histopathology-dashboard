"""Read-only cross-PC Git check and private artifact-manifest verifier."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BRANCH='research/local-backup-2026-11-13'
def git(*args):
 env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
 return subprocess.check_output(['git',*args],cwd=ROOT,env=env,text=True,stderr=subprocess.DEVNULL).strip()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--verify-manifest',type=Path);args=parser.parse_args()
 result={'project':str(ROOT),'branch':git('branch','--show-current'),'local_commit':git('rev-parse','HEAD'),'working_changes':git('status','--porcelain').splitlines()}
 try:result['live_remote_commit']=git('ls-remote','--heads','origin','refs/heads/'+BRANCH).split()[0]
 except (subprocess.CalledProcessError,IndexError):result['live_remote_commit']=None
 result['source_synced']=result['branch']==BRANCH and result['local_commit']==result['live_remote_commit'] and not result['working_changes']
 result['scope']='Git-tracked source only; databases, datasets, environments and checkpoints require separate verification'
 if args.verify_manifest:
  manifest=json.loads(args.verify_manifest.read_text(encoding='utf-8'));counts={'matched':0,'missing':0,'different':0};differences=[]
  for entry in manifest['files']:
   p=(ROOT/entry['path']).resolve()
   if not p.is_relative_to(ROOT):raise ValueError('Manifest path escapes project root')
   state='missing' if not p.is_file() else 'matched' if p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'] else 'different'
   counts[state]+=1
   if state!='matched':differences.append({'path':entry['path'],'state':state})
  result['artifact_verification']={'counts':counts,'differences':differences}
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
