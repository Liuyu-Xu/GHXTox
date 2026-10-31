"""Four new B-only fits, then evaluation and group-aware summary."""
import os,subprocess,sys,json,time,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def status(v):
 t=P/'driver_status.tmp';t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(P/'driver_status.json')
def run():
 m=read(P/'manifest.json');assert m['source_sha256']==hashlib.sha256((R/'src/ghxtox/lora_mamba_1d.py').read_bytes()).hexdigest();assert m['role_sha256']==hashlib.sha256((R/'analysis/group_weight_trial_20261027/role_weight_contract.json').read_bytes()).hexdigest();assert m['protocol_sha256']==hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest()
 start=time.time()
 for phase in ['train','evaluate']:
  for f in [0,1,2,4]:
   folder=P/f'fold{f}';marker=folder/('status.json' if phase=='train' else 'evaluation_status.json')
   if marker.exists():
    x=read(marker)
    if (phase=='train' and x.get('state')=='completed') or (phase=='evaluate' and x.get('completed') is True):print('REUSE',phase,f,flush=True);continue
   status({'state':'running','phase':phase,'fold':f,'pid':os.getpid(),'elapsed_seconds_at_stage_start':time.time()-start});print('START',phase,f,flush=True)
   args=[sys.executable,'-u',str(folder/'run.py'),'--execute'] if phase=='train' else [sys.executable,'-u',str(P/'evaluate_fold.py'),'--fold',str(f)]
   subprocess.run(args,cwd=str(R),check=True,env=os.environ.copy())
 status({'state':'running','phase':'summarize','pid':os.getpid()});subprocess.run([sys.executable,'-u',str(P/'summarize.py')],cwd=str(R),check=True,env=os.environ.copy())
 metrics=read(P/'metrics.json')['four_followup_folds']['all'];seq=metrics['sequence'];fus=metrics['fusion']
 summary=f"4折（不含已观察的第3折）一维MCC A/B为{seq['A']['MCC']:.4f}/{seq['B']['MCC']:.4f}，错{seq['A']['errors']}/{seq['B']['errors']}，漏判{seq['A']['FN']}/{seq['B']['FN']}，AUPRC {seq['A']['AUPRC']:.4f}/{seq['B']['AUPRC']:.4f}；条件融合MCC {fus['A']['MCC']:.4f}/{fus['B']['MCC']:.4f}。"
 (P/'STATUS.md').write_text('# 低Cys加权4折复核已完成\n\n'+summary+'\n\n单种子，已有研究数据；仅条件三维融合，不能当新外部盲测或完整三种子系统。[报告](REPORT.md)。\n',encoding='utf-8')
 for path in [R/'README.md',R/'reports/研究分析总览_20261025.md']:
  s=path.read_text(encoding='utf-8');prefix='analysis/' if path.name=='README.md' else '../analysis/'
  marker='> **低Cys加权多折复核（2026-10-28）：**'
  line=marker+summary+' 单种子、历史数据、条件三维融合；详见[报告]('+prefix+'lowcys_weight_multifold_20261028/REPORT.md)。'
  matches=[x for x in s.splitlines() if x.startswith(marker)]
  if matches:s=s.replace(matches[0],line,1)
  else:
   parts=s.split('\n',1);s=parts[0]+'\n\n'+line+'\n\n'+parts[1].lstrip('\n')
  path.write_text(s,encoding='utf-8')
 plan=R/'.planning/2026-10-28-lowcys-multifold';(plan/'task_plan.md').write_text('# 低Cys加权多折验证\n1. [完成] 冻结其余4折协议。\n2. [完成] 复用无加权基线，训练4折加权。\n3. [完成] 各折校准、留出评估、分组与成对区间。\n4. [完成] 结果报告与机器核验；是否继续完整三种子应根据结果决定。\n',encoding='utf-8')
 with (plan/'progress.md').open('a',encoding='utf-8') as f:f.write('\n完成训练、校准、评价和核验。'+summary+'\n')
 status({'state':'completed','seconds':time.time()-start,'report':str(P/'REPORT.md')});print('ALL COMPLETE',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:status({'state':'failed','error':repr(e),'pid':os.getpid()});raise
