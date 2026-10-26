from pathlib import Path
import json
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
O=Path(__file__).resolve().parent;read=lambda n:json.loads((O/n).read_text(encoding='utf-8-sig'))
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(n,x):(O/n).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
P=read('supplement_protocol.json');rows=read('sample_contributions.json');n=len(rows);y=np.array([r['label'] for r in rows]);fold=np.array([r['fold'] for r in rows]);_,gi=np.unique([r['group_id'] for r in rows],return_inverse=True);ng=gi.max()+1
f=lambda key:np.array([r['features'][key] if r['features'][key] is not None else np.nan for r in rows]);base=np.array([r['baseline4_probability'] for r in rows]);cys=f('cys_count');arms=['fusion_3Di4','fusion_masked4'];rng=np.random.default_rng(20261027);weights=np.array([np.bincount(rng.integers(0,ng,ng),minlength=ng) for _ in range(2000)],dtype=float);size=np.bincount(gi,minlength=ng);norm=weights@size
strata={'all':np.ones(n,dtype=bool),'Cys_lt4':cys<4,'Cys_ge4':cys>=4,'membrane_mention':np.array([bool(r['mechanism_mentions']['membrane_or_lysis']) for r in rows]),'no_cross_source_label_conflict':np.array([not r['cross_source_label_conflict'] for r in rows])};out=[];st=[];threshold=[]
for a in arms:
 role=np.array([r[a]['role'] for r in rows]);delta=(role=='corrected').astype(int)-(role=='harmed');rescue=role=='corrected';harm=role=='harmed';p=np.array([r[a]['probability'] for r in rows])
 st.append(dict(arm=a,ensemble_rescues=int(rescue.sum()),both_paired_seeds_reproduce_rescue=sum(rescue[i] and all(r[a]['seed_roles'][str(s)]=='corrected' for s in [20261026,20261027]) for i,r in enumerate(rows)),ensemble_harms=int(harm.sum()),both_paired_seeds_reproduce_harm=sum(harm[i] and all(r[a]['seed_roles'][str(s)]=='harmed' for s in [20261026,20261027]) for i,r in enumerate(rows))))
 for label in [0,1]:
  for name,mask in strata.items():
   m=mask&(y==label);g=np.bincount(gi,weights=delta*m,minlength=ng);ds=(weights@g)*n/norm
   seeds=[]
   for seed in [20261026,20261027]:
    rr=np.array([r[a]['seed_roles'][str(seed)] for r in rows]);seeds.append(int(((rr=='corrected')&m).sum()-((rr=='harmed')&m).sum()))
   out.append(dict(arm=a,label=label,stratum=name,n=int(m.sum()),net_errors_removed=int(delta[m].sum()),CI95_net_count=np.quantile(ds,[.025,.975]),fold_net=[int(delta[m&(fold==j)].sum()) for j in range(5)],paired_seed_net=seeds))
  for name,mask in [('corrected',rescue),('harmed',harm),('persistent_wrong',role=='persistent_wrong')]:
   m=mask&(y==label);v=np.abs(base[m]-.5)
   threshold.append(dict(arm=a,label=label,role=name,n=int(m.sum()),baseline_margin_median=float(np.median(v)) if len(v) else None,baseline_margin_le01=int((v<=.1).sum()),baseline_margin_le02=int((v<=.2).sum()),median_probability_shift=float(np.median(p[m]-base[m])) if m.any() else None))
save('subgroup_net_uncertainty.json',out);save('seed_overlap.json',st);save('threshold_diagnostics.json',threshold)
X=np.stack([f(k) for k in ['sequence_length','cys_count','cys_span_fraction','net_charge_proxy','mean_hydropathy','mean_plddt']]+[base],axis=1);adjust=[]
for feature in ['hydrophobic_contact_excess','ca_radius_gyration','charge_hydrophobic_separation']:
 v=f(feature);valid=np.isfinite(v)&np.isfinite(X).all(axis=1);res=np.full(n,np.nan)
 for j in range(5):
  tr=valid&(fold!=j);te=valid&(fold==j);sc=StandardScaler().fit(X[tr]);model=Ridge(alpha=1).fit(sc.transform(X[tr]),v[tr]);res[te]=v[te]-model.predict(sc.transform(X[te]))
 for a in arms:
  rr=np.array([r[a]['role'] for r in rows])
  for label in [0,1]:
   c=valid&(y==label)&(rr=='corrected');d=valid&(y==label)&(rr=='persistent_wrong');cs=np.bincount(gi[c],weights=res[c],minlength=ng);cn=np.bincount(gi[c],minlength=ng);ds=np.bincount(gi[d],weights=res[d],minlength=ng);dn=np.bincount(gi[d],minlength=ng);nc=weights@cn;nd=weights@dn;ok=(nc>0)&(nd>0);dist=(weights[ok]@cs)/nc[ok]-(weights[ok]@ds)/nd[ok]
   delta=float(res[c].mean()-res[d].mean());fd=[float(res[c&(fold==j)].mean()-res[d&(fold==j)].mean()) if (c&(fold==j)).any() and (d&(fold==j)).any() else None for j in range(5)]
   tail=min(1.,2*min((1+(dist<=0).sum())/(len(dist)+1),(1+(dist>=0).sum())/(len(dist)+1)))
   adjust.append(dict(arm=a,label=label,feature=feature,n_corrected=int(c.sum()),n_persistent_wrong=int(d.sum()),residual_difference=delta,CI95=np.quantile(dist,[.025,.975]),fold_differences=fd,bootstrap_sign_tail=tail))
# Twelve comparisons: both candidates, both labels, three spatial descriptors.
order=np.argsort([r['bootstrap_sign_tail'] for r in adjust]);q=1.
for rank in range(len(order),0,-1):
 i=order[rank-1];q=min(q,adjust[i]['bootstrap_sign_tail']*len(order)/rank);adjust[i]['BH_q_12']=q
save('adjusted_spatial_comparisons.json',adjust)
print(json.dumps(clean(dict(seed_overlap=st,subgroups=out,adjusted=adjust)),ensure_ascii=False,indent=2))
