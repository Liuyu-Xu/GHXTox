import sys,os,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
def save(x):(P/'driver_status.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
t=time.time()
try:
 for phase,script,args in [('training','run.py',['--execute']),('evaluation','evaluate.py',[])]:
  save({'state':'running','phase':phase,'pid':os.getpid()});print('START',phase,flush=True)
  subprocess.run([sys.executable,'-u',str(P/script),*args],cwd=str(R),check=True,env=os.environ.copy())
 save({'state':'completed','seconds':time.time()-t,'report':str(P/'REPORT.md')});print('ALL COMPLETE',flush=True)
except BaseException as e:save({'state':'failed','error':repr(e)});raise
