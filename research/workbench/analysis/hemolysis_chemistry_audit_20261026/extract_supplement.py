import zipfile,xml.etree.ElementTree as E,pathlib
p=pathlib.Path('analysis/hemolysis_chemistry_audit_20261026/sources')
for n in ['Table_1.DOCX','Table_4.DOCX']:
 with zipfile.ZipFile(p/n) as z: r=E.fromstring(z.read('word/document.xml'))
 ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
 lines=[' | '.join(''.join(c.itertext()) for c in row.findall('w:tc',ns)) for row in r.findall('.//w:tr',ns)]
 print(n)
 for line in lines:
  if 'RR4' in line or '375' in line or 'Hemol' in line: print(line)
 (p/(n+'.txt')).write_text('\n'.join(lines),encoding='utf8')
