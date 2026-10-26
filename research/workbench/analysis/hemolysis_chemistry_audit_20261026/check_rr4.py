import csv
for r in csv.DictReader(open('analysis/environment_feasibility_20261025/sources/Hemolytik2_complete_data.csv',encoding='utf-8-sig')):
 if r['pmid']=='34819923' and r['name'] in ['RR4-OH','RR4-NH2']:print(r['id'],r['name'],r['activity'])
