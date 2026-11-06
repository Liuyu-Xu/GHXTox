"""Sequential real training then evaluation; resumable completed arms."""
import os,subprocess,sys,json,time,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
def status(data):
 tmp=P/'driver_status.tmp';tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(P/'driver_status.json')
def main():
 m=json.loads((P/'manifest.json').read_text(encoding='utf-8'))
 assert m['source_sha256']==hashlib.sha256((R/'src/ghxtox/lora_mamba_1d.py').read_bytes()).hexdigest()
 assert m['role_contract_sha256']==hashlib.sha256((R/'analysis/group_weight_trial_20261027/role_weight_contract.json').read_bytes()).hexdigest()
 assert m['protocol_sha256']==hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest()
 t=time.time()
 for phase in ['train','evaluate']:
  for f in [1,2,3,4]:
   fold=P/f'fold{f}';done=fold/('status.json' if phase=='train' else 'evaluation_metrics.json')
   if done.exists() and (phase!='train' or json.loads(done.read_text(encoding='utf-8')).get('state')=='completed'):
    print('reuse',phase,f,flush=True);continue
   status({'state':'running','phase':phase,'fold':f,'pid':os.getpid(),'seconds':time.time()-t});print('starting',phase,f,flush=True)
   command=[sys.executable,'-u',str(fold/('run.py' if phase=='train' else 'evaluate.py'))]
   if phase=='train':command.append('--execute')
   subprocess.run(command,cwd=str(R),check=True,env=os.environ.copy())
 subprocess.run([sys.executable,'-u',str(P/'summarize.py')],cwd=str(R),check=True,env=os.environ.copy())
 status({'state':'completed','seconds':time.time()-t,'report':str(P/'REPORT.md')});print('ALL COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  status({'state':'failed','error':repr(e),'pid':os.getpid()});raise
