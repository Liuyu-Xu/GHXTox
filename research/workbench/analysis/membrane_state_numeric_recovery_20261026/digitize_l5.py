"""Scientific pixel measurement of published bars, not original numeric data.
All11 peptides retained. Calibration/tolerance saved; no model fitting.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import label,find_objects
O=Path(__file__).resolve().parent;R=O.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,o):(O/n).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
fig6=np.array(Image.open(O/'sources/L5_fig6.png').convert('RGB'));fig3=np.array(Image.open(O/'sources/L5_fig3.png').convert('RGB'))
# Coordinates refer to the inspected tool-rendered viewports, scaled to original pixels.
def boxes(img,view,roi,kind,baseline):
 sy=img.shape[0]/view[1];sx=img.shape[1]/view[0];x0,y0,x1,y1=roi
 x0,x1=int(x0*sx),int(x1*sx);y0,y1=int(y0*sy),int(y1*sy)
 a=img[y0:y1,x0:x1].astype(float)
 if kind=='black':mask=np.max(a,axis=2)<75
 elif kind=='gray':mask=(np.max(a,axis=2)<130)&(np.max(a,axis=2)-np.min(a,axis=2)<8)
 else:
  targets={'blue':[79,129,189],'red':[192,80,77],'green':[155,187,89]}
  mask=np.max(abs(a-np.array(targets[kind])),axis=2)<22
 lab,n=label(mask);out=[]
 for sl in find_objects(lab):
  if sl is None:continue
  ys,xs=sl;w=xs.stop-xs.start;h=ys.stop-ys.start;bottom=(y0+ys.stop)/sy
  if w<6*sx or h<4*sy or abs(bottom-baseline)>7:continue
  out.append(dict(xmin=x0+xs.start,xmax=x0+xs.stop-1,ymin=y0+ys.start,ymax=y0+ys.stop-1,center_x_view=(x0+(xs.start+xs.stop-1)/2)/sx,top_y_view=(y0+ys.start)/sy,width_view=w/sx,height_view=h/sy))
 out.sort(key=lambda x:x['xmin']);assert len(out)==11,(kind,len(out),out)
 return out
cal={
 'hemolysis_percent':dict(figure='6A',view=[2048,1043],roi=[140,80,825,270],kind='black',baseline=267,ticks=[[267,0],[98,80]],unit='percent at128ug/mL'),
 'helicity_TFE_percent':dict(figure='6B_top',view=[2048,1043],roi=[140,380,825,540],kind='black',baseline=536,ticks=[[536,60],[400,84]],unit='CD-derived percent at50%TFE'),
 'theta_product_TFE_scaled':dict(figure='6B_middle',view=[2048,1043],roi=[140,575,825,754],kind='gray',baseline=751,ticks=[[751,250],[627,700]],unit='theta222*theta208*1e-6 at50%TFE')}
for wave,roi,baseline,ticks in [('222',[1115,838,1865,1010],1007,[[1007,0],[844,30]]),('208',[1115,1036,1865,1226],1223,[[1223,0],[1061,30]])]:
 for env,color in [('TFE','blue'),('SDS','red'),('DPC','green')]:cal[f'abs_theta{wave}_{env}_scaled']=dict(figure='3D',view=[1887,1344],roi=roi,kind=color,baseline=baseline,ticks=ticks,unit='absolute mean residue ellipticity*1e-3',environment={'TFE':'50%TFE','SDS':'10mMSDS','DPC':'2mMDPC'}[env])
measured={};debug={}
for name,c in cal.items():
 img=fig6 if c['figure'].startswith('6') else fig3
 b=boxes(img,c['view'],c['roi'],c['kind'],c['baseline']);debug[name]=b
 (y0,v0),(y1,v1)=c['ticks'];slope=(v1-v0)/(y1-y0);tol=abs(slope)*4
 measured[name]=[dict(value=v0+(r['top_y_view']-y0)*slope,lower=v0+(r['top_y_view']-y0)*slope-tol,upper=v0+(r['top_y_view']-y0)*slope+tol,unit=c['unit'],tolerance_basis='conservative4viewportpixel envelope for bar edge and axis reading; not a statistical confidence interval',source_figure=c['figure']) for r in b]
compounds=read(R/'analysis/hemolysis_selectivity_audit_20261026/compounds.json');lookup={r['name']:r for r in compounds if r['group']=='L5K5W_position_isomers'}
assert set(lookup)=={'W'+str(i) for i in range(1,12)}
rows=[]
for i in range(11):
 name='W'+str(i+1);r=lookup[name]
 vals={k:v[i] for k,v in measured.items()}
 a,b=vals['abs_theta222_TFE_scaled'],vals['abs_theta222_SDS_scaled']
 delta=dict(value=a['value']-b['value'],lower=a['lower']-b['upper'],upper=a['upper']-b['lower'],unit='difference of absolute ellipticity*1e-3;TFE-minusSDS',not_water_to_membrane_delta=True)
 rows.append(dict(name=name,sequence=r['sequence'],chemical_form=r['chemical_form'],source_pmid='23344198',existing_direct_MHC5=r['hemolysis'],digitized_values=vals,environment_contrast=delta,water_value=None,water_missing_reason='Fig3A overlays unlabeled per-peptide curves; individual identities not recovered',data_status='published_figure_digitization_not_raw_measurements'))
def r2(x,y):return float(np.corrcoef(x,y)[0,1]**2)
y=np.array([r['digitized_values']['hemolysis_percent']['value'] for r in rows]);he=np.array([r['digitized_values']['helicity_TFE_percent']['value'] for r in rows]);prod=np.array([r['digitized_values']['theta_product_TFE_scaled']['value'] for r in rows]);repro=dict(helicity_r2_digitized=r2(he,y),helicity_r2_published=.7321,theta_product_r2_digitized=r2(prod,y),theta_product_r2_published=.9022,purpose='digitization quality check reproducing original authors in-sample result; not a new mechanistic discovery or classifier performance')
assert abs(repro['helicity_r2_digitized']-.7321)<.06
assert abs(repro['theta_product_r2_digitized']-.9022)<.06
for r in rows:
 assert sorted(r['sequence'])==sorted(rows[0]['sequence'])
save('L5_digitized_all11.json',rows);save('digitization_calibration.json',dict(manual_axis_calibration=cal,feature_extraction='color/black connected components, all11 ordered by original W1..W11 labels',pixel_tolerance_viewport=4,image_dimensions=dict(fig6=list(fig6.shape[:2][::-1]),fig3=list(fig3.shape[:2][::-1])),source_hashes={n:hashlib.sha256((O/'sources'/n).read_bytes()).hexdigest() for n in ['L5_fig6.png','L5_fig3.png']}));save('digitization_bar_boxes.json',debug);save('digitization_quality.json',repro)
print(json.dumps(repro,indent=2));print('Recovered11matchedpeptides,9publishedbarquantities each; no water-state identity or atomic coordinates imputed.')
