"""Conservative intervals at222nm from the native Fig2 raster in the PDF.
Overlapping TFE/SDS curves are deliberately not assigned invented separate values.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
O=Path(__file__).resolve().parent;R=O.parents[1]
path=O/'sources/PMAP_page3_image0_X1.jpg';a=np.array(Image.open(path).convert('RGB'));assert a.shape[:2]==(703,990)
configs={
 'RI21':dict(x195=85,x250=460,y_zero=153,y_tick=263,tick_value=-4,water_y=[156,175],mimic_y=[202,232]),
 'RI18':dict(x195=594,x250=969,y_zero=153,y_tick=263,tick_value=-4,water_y=[158,180],mimic_y=[200,230]),
 'TI15':dict(x195=85,x250=460,y_zero=504,y_tick=611,tick_value=-3,water_y=[507,524],mimic_y=[549,575]),
 'RI12':dict(x195=594,x250=969,y_zero=486,y_tick=611,tick_value=-1.5,water_y=[490,513],mimic_y=[524,550])}
comp=json.loads((R/'analysis/hemolysis_selectivity_audit_20261026/compounds.json').read_text(encoding='utf-8-sig'));comp={x['name']:x for x in comp if x['group']=='PMAP36_truncations'}
if not comp:
 comp={x['name']:x for x in json.loads((R/'analysis/hemolysis_selectivity_audit_20261026/compounds.json').read_text(encoding='utf-8-sig')) if x['pmc']=='PMC4890124'}
rows=[]
for name,c in configs.items():
 x=c['x195']+(222-195)*(c['x250']-c['x195'])/55;lo=int(x-7);hi=int(x+8);slope=c['tick_value']/(c['y_tick']-c['y_zero']);vals={}
 for kind,roi in [('water',c['water_y']),('membrane_mimic_union',c['mimic_y'])]:
  y0,y1=roi;mask=np.max(a[y0:y1,lo:hi],axis=2)<100;ys=np.where(mask.any(axis=1))[0]+y0;ys=ys[abs(ys-c['y_zero'])>3];assert len(ys)>0
  ymin,ymax=int(ys.min()),int(ys.max());values=sorted([(ymin-3-c['y_zero'])*slope,(ymax+3-c['y_zero'])*slope]);vals[kind]=dict(lower=values[0],upper=values[1],value=None,ink_y_pixels=[ymin,ymax],unit='theta222 times1e-4,deg*cm2/dmol',tolerance='marker/stroke envelope plus3nativepixels and wavelength window about222±1nm; not95%CI')
 w=vals['water'];m=vals['membrane_mimic_union'];delta=dict(lower=m['lower']-w['upper'],upper=m['upper']-w['lower'],value=None,unit=m['unit'],scope='envelope for eitherTFE/SDS minuswater; does not rank separate TFE vsSDS')
 source=comp[name]
 rows.append(dict(name=name,sequence=source['sequence'],chemical_form=source['chemical_form'],direct_host_endpoint=source['hemolysis'],source_pmid='27251456',native_image=str(path),figure='Figure2/PDFpage3',wavelength_nm=222,x_window_pixels=[lo,hi],calibration=c,digitized_intervals=vals,state_contrast_interval=delta,TFE_separate_value=None,SDS_separate_value=None,reason='curves overlap near222nm; shared conservative union only',status='approximate_published_figure_intervals_not_raw_spectra'))
for name,obj in [('PMAP_digitized_all4.json',rows),('PMAP_digitization_provenance.json',dict(source_pdf_sha256=hashlib.sha256((O/'sources/PMAP_article.pdf').read_bytes()).hexdigest(),native_image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),image_dimensions=[990,703],source_page=3,all_four_derivatives_retained=True,no_separate_overlapping_curves_invented=True))]: (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
for r in rows:print(r['name'],r['digitized_intervals'],r['direct_host_endpoint'])
