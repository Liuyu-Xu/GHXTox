"""Rebuild evidence-only audit. No model or GHXTox prediction is read.
Run: D:/anaconda/envs/pytorch_env/python.exe analysis/hemolysis_selectivity_audit_20261026/audit.py
Optional --fetch refreshes only the three original full-text XML files and figure.
"""
from pathlib import Path
import csv,json,re,hashlib,collections,itertools,sys,xml.etree.ElementTree as E
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]; SRC=OUT/'sources'
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
urls={p+'.xml':f'https://www.ebi.ac.uk/europepmc/webservices/rest/{p}/fullTextXML' for p in ['PMC4890124','PMC6405874','PMC6270530']}
urls['L5figure.png']='https://pub.mdpi-res.com/molecules/molecules-18-00859/article_deploy/html/images/molecules-18-00859-g001.png'
if '--fetch' in sys.argv:
 import requests
 for name,url in urls.items():
  r=requests.get(url,timeout=60);r.raise_for_status();(SRC/name).write_bytes(r.content)
def txt(x):return ''.join(x.itertext()).strip()
docs={p:E.parse(str(SRC/(p+'.xml'))) for p in ['PMC4890124','PMC6405874','PMC6270530']}
def rows(p,tid):
 t=next(t for t in docs[p].findall('.//table-wrap') if t.get('id')==tid)
 return [[txt(c) for c in tr] for tr in t.findall('.//tr')]
def md(p,typ):return next(txt(a) for a in docs[p].findall('.//article-id') if a.get('pub-id-type')==typ)
groups=[]; compounds=[]; pairs=[]
def group(p,scaffold,unit,conditions,limitations):
 g=dict(pmc=p,pmid=md(p,'pmid'),doi=md(p,'doi'),scaffold=scaffold,unit=unit,source_url=f'https://pmc.ncbi.nlm.nih.gov/articles/{p}/',conditions=conditions,limitations=limitations)
 groups.append(g);return g
def compound(g,name,seq,hem,source):
 c=dict(group=g['scaffold'],pmc=g['pmc'],name=name,sequence=seq,chemical_form=dict(c_terminal='amide',n_terminal='free as drawn; no added cap reported',stereochemistry='standard residues as written',topology='linear chain; PMAP-36 Cys oxidation not explicitly established'),hemolysis=hem,sequence_source=source,measurement_type='direct experiment reported in original article',same_chemical_form_basis='same named synthetic preparation used across assays; no separate assay-specific form reported')
 compounds.append(c);return c
def pair(g,c,strain,mic,original=True,cell=''):
 p=dict(group=g['scaffold'],pmc=g['pmc'],pmid=g['pmid'],name=c['name'],sequence=c['sequence'],chemical_form=c['chemical_form'],strain=strain,mic_value=float(mic),mic_unit=g['unit'],mic_current_article_experiment=original,mic_raw_cell=cell,hemolysis=c['hemolysis'],source_url=g['source_url'],strict_HC50_MIC_eligible=False)
 pairs.append(p)
g=group('PMC4890124','PMAP36_truncations','uM',dict(mic_medium='MHB',mic_time_h=18,mic_temperature_C=37,mic_inoculum='approximately 1e5 CFU/mL suspension before equal-volume mixing; final value not silently inferred',rbc='human',hemolysis_buffer='PBS',hemolysis_time_h=1,hemolysis_temperature_C=37,rbc_density=None),['PMAP-36 and RI12 six MIC cells each flagged c: values determined in previous reference 15','C77-31 named S. typhi in table but S. enterica serovar typhimurium in methods: unresolved','truncation series, not all equal-length point mutants','PMAP-36 lone Cys oxidation state unspecified'])
seq={r[0].rstrip('c'):r[1].split('-NH')[0] for r in rows(g['pmc'],'t1')[1:]}
hem={r[0]:r[1] for r in rows(g['pmc'],'t4') if len(r)>=6 and r[0] in seq}
cs={n:compound(g,n,s,dict(endpoint='MHC_5percent',operator='>' if hem[n].startswith('>') else '=',value=float(hem[n].lstrip('>')),unit='uM',rbc='human',source='Table 4; Methods Quantification of hemolytic activity',censoring='right' if hem[n].startswith('>') else 'none',not_HC50=True),'Table 1') for n,s in seq.items()}
strains=['E. coli ATCC25922','E. coli EC183','E. coli UB1005','C77-31 species conflict table/methods','S. aureus ATCC29213','S. aureus ATCC25923','S. epidermidis ATCC12228','B. subtilis CMCC63501']
for r in rows(g['pmc'],'t2'):
 if r[0] in cs:
  for strain,cell in zip(strains,r[1:]):pair(g,cs[r[0]],strain,re.match(r'\d+',cell)[0],not ('c' in cell.split('(')[0]),cell)
g=group('PMC6405874','Pap12','uM',dict(mic_medium='1% peptone',mic_time_h=16,mic_temperature_C=37,mic_inoculum='2e6 CFU/mL bacterial suspensions',rbc='fresh human',hemolysis_buffer=None,hemolysis_time_h=None,hemolysis_temperature_C=None,rbc_density=None),['Table 2 HC10=200 uM for Pap12-1..5 is two-times-maximum substitution, not observed threshold','Pap12-6 Table 2 HC10=100 conflicts with body: only 3% hemolysis at 100 uM','hemolysis detailed methods delegated to ref23 and not recovered; no numeric zero inferred from no detectable hemolysis'])
cs={}
for r in rows(g['pmc'],'Tab1')[1:]:
 if r[0].startswith('Pap12'):
  n=r[0];cs[n]=compound(g,n,r[1].split('-NH')[0],dict(endpoint='percent_at_100uM',value=3 if n=='Pap12-6' else None,description='3% at 100 uM' if n=='Pap12-6' else 'no detectable hemolysis through 100 uM; exact values/detection limit unavailable',dose=100,dose_unit='uM',rbc='human',published_HC10_value=100 if n=='Pap12-6' else 200,published_HC10_is_observed=False,source='Results Cytotoxicity; Table 2; Figure 2A not digitized',not_HC50=True),'Table 1')
strainmap=dict(zip(['E. coli','P. aeruginosa','A. baumannii','S. typhimurium','MDREC 1229','MDRPA 2002','MDRAB 12035','MDRST 8007','S. aureus','B. subtilis','S. epidermidis','MRSA 3126'],['E. coli KCTC1682','P. aeruginosa KCTC2004','A. baumannii KCTC2508','S. typhimurium KCTC1926','E. coli CCARM1229','P. aeruginosa CCARM2002','A. baumannii CCARM12035','S. typhimurium CCARM8007','S. aureus KCTC1621','B. subtilis KCTC3068','S. epidermidis KCTC1917','S. aureus CCARM3126']))
for r in rows(g['pmc'],'Tab2'):
 if r[0] in strainmap:
  for c,cell in zip(cs.values(),r[1:7]):pair(g,c,strainmap[r[0]],cell,cell=cell)
g=group('PMC6270530','L5K5W_position_isomers','ug/mL',dict(mic_medium='broth microdilution; detailed medium delegated to previous ref21',mic_time_h=None,mic_temperature_C=None,rbc='human',hemolysis_buffer='PBS',hemolysis_time_h=1,hemolysis_temperature_C=None,rbc_density='10% v/v suspension; mixing dilution not specified here'),['positional isomers/sequence rotations, not eleven independent scaffolds','MHC lowest tested dose causing >5% hemolysis; not HC50','temperature and MIC medium/time not explicitly recovered; never borrow from other paper'])
seqs=['WLKKLLKKLLK','KWLKKLLKKLL','LKWLKKLLKKL','LLKWLKKLLKK','KLLKWLKKLLK','KKLLKWLKKLL','LKKLLKWLKKL','LLKKLLKWLKK','KLLKKLLKWLK','KKLLKKLLKWL','LKKLLKKLLKW']
# Manually transcribed and visually verified against original Figure 1 (sources/L5figure.png).
hs=[int(r[1]) for r in rows(g['pmc'],'molecules-18-00859-t002') if r[0].startswith('L5')]
cs=[compound(g,f'W{i+1}',s,dict(endpoint='MHC_gt5percent',operator='=',value=h,unit='ug/mL',rbc='human',source='Table 2; Methods 3.6',not_HC50=True),'Original Figure 1 visual transcription; agrees with cached Hemolytik PMID23344198') for i,(s,h) in enumerate(zip(seqs,hs))]
strainmap=dict(zip(['B. subtilis','S. aureus','S. epidermis','M. luteus','E. coli','S. dysentariae','S. typhimorium','K. pneumoniae','P. aeruginosa','MRSA-TK784'],['B. subtilis ATCC6633','S. aureus ATCC6538p','S. epidermis ATCC12228','M. luteus ATCC10240','E. coli ATCC25922','S. dysentariae ATCC9752','S. typhimurium ATCC14028','K. pneumoniae ATCC10031','P. aeruginosa ATCC27853','S. aureus MRSA-TK784']))
for r in rows(g['pmc'],'molecules-18-00859-t001'):
 if r[0] in strainmap:
  for c,cell in zip(cs,r[1:]):pair(g,c,strainmap[r[0]],cell,cell=cell)
def edit(a,b):
 d=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nd=[i]
  for j,y in enumerate(b,1):nd.append(min(nd[-1]+1,d[j]+1,d[j-1]+(x!=y)))
  d=nd
 return d[-1]
edges=[]
for a,b in itertools.combinations(compounds,2):
 if a['group']!=b['group']:continue
 dist=edit(a['sequence'],b['sequence']);rel=dist/max(len(a['sequence']),len(b['sequence']))
 if rel<=.2:edges.append(dict(group=a['group'],a=a['name'],b=b['name'],edit_distance=dist,normalized_edit=rel))
save('groups.json',groups);save('compounds.json',compounds);save('paired_records.json',pairs);save('near_variant_edges.json',edges)
with (OUT/'paired_records.csv').open('w',encoding='utf-8-sig',newline='') as f:
 flat=[{**{k:v for k,v in p.items() if k not in ['chemical_form','hemolysis']},'chemical_form':json.dumps(p['chemical_form'],ensure_ascii=False),'hemolysis':json.dumps(p['hemolysis'],ensure_ascii=False)} for p in pairs]
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
validation=[]
for p in pairs:
 h=p['hemolysis']
 if p['group']=='L5K5W_position_isomers' and p['strain']=='E. coli ATCC25922':validation.append(dict(name=p['name'],endpoint='MHC_gt5percent / MIC; within-study screening ratio only',ratio=h['value']/p['mic_value']))
assert len(compounds)==22 and len(pairs)==222 and sum(not p['mic_current_article_experiment'] for p in pairs)==12
assert len(validation)==11 and len({p['sequence'] for p in pairs})==22
save('minimal_verification.json',dict(assertions_passed=True,model_training=False,ghxtox_predictions_read=False,within_study_screening_ratios=validation,range=[min(v['ratio'] for v in validation),max(v['ratio'] for v in validation)],interpretation='No HC50, no cross-paper pooled ratio. Same human RBC source does not imply same assay conditions.'))
manifest=[]
for name,url in urls.items():
 p=SRC/name;manifest.append(dict(file=str(p),url=url,status=200,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role='original article XML or original Figure 1'))
for name,url,status in [('PMC12044605.xml','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12044605/fullTextXML',500),('PG1.xml','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12044605/fullTextXML?format=xml',500),('PG1.html','https://pmc.ncbi.nlm.nih.gov/articles/PMC12044605/',403),('L5_figure1.jpg','https://www.mdpi.com/molecules/molecules-18-00859/article_deploy/html/images/molecules-18-00859-g001.png',403)]:
 p=SRC/name;manifest.append(dict(file=str(p),url=url,status=status,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role='failed retrieval; NOT evidence'))
raw=ROOT/'analysis/environment_feasibility_20261025/sources/Hemolytik2_complete_data.csv'
with raw.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
manifest.append(dict(file=str(raw),url='https://webs.iiitd.edu.in/raghava/hemolytik2/Hemolytik2_complete_data.csv',sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),rows=len(rr),columns=list(rr[0]),role='reused local official database cache; lacks paired MIC fields; no same-PMID-only joins'))
save('source_manifest.json',manifest)
summary=dict(original_articles_recovered=3,independent_scaffold_series=3,unique_compounds=22,compound_counts=dict(collections.Counter(c['group'] for c in compounds)),paired_mic_strain_rows=222,paired_current_article_mic_rows=210,previous_publication_mic_rows=12,exact_HC50_compounds=0,threshold_MHC5_compounds=16,right_censored_MHC5_compounds=2,qualitative_or_fixed_dose_Pap12_compounds=6,author_HC10_surrogate_values_not_experimental=6,near_variant_edges_20percent= len(edges),near_variant_edges_by_group=dict(collections.Counter(e['group'] for e in edges)),strict_condition_complete_multistudy_panel=False,training_recommended=False,PG1='not recovered: original paper supports experimental subset but no raw selected-variant table recovered; SLAY and predicted variants excluded',decision='Can do small within-study diagnostic; insufficient for credible multi-paper/multi-scaffold predictive generalization; endpoint heterogeneity and censoring remain',stop_condition='No model training until >=5 independent paper/scaffold blocks each >=8 exact-identity+chemistry dual-endpoint variants, >=3 usable near-variant pairs, known RBC/strain and assay conditions, >=40 non-surrogate comparable quantitative outcomes total; threshold is planning gate, not power proof')
summary.pop('author_HC10_surrogate_values_not_experimental')
summary.update(author_HC10_two_times_maximum_placeholders=5,author_HC10_conflicts_with_body=1,independent_scaffold_definition='three named design series; no claim of statistical or evolutionary independence')
save('summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
