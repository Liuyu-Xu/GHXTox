import requests,pathlib,xml.etree.ElementTree as E
p=pathlib.Path('analysis/hemolysis_chemistry_audit_20261026/sources')
for pmc in ['PMC8606745','PMC8348142','PMC5322399']:
 r=requests.get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML',timeout=45);r.raise_for_status();(p/(pmc+'.xml')).write_bytes(r.content)
 root=E.fromstring(r.content)
 print('\n',pmc)
 for sec in root.iter('sec'):
  title=sec.find('title');t='' if title is None else ''.join(title.itertext())
  if any(s in t.lower() for s in ['hemol','haemol','synth','peptide design','peptide preparation']): print(E.tostring(sec,encoding='unicode'))
