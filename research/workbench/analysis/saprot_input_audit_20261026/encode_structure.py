"""Encode cached N/CA/C backbone to approximate3Di using supported missing-CB fallback.
Never reinterpret the cached sidechain centroid as a real CB atom.
"""
from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
import torch
O=Path(__file__).resolve().parent;R=O.parents[1];sys.path.insert(0,str(O/'vendor'));import mini3di
cache=R.parent/'GHXTox-ziliao/train_chemical_sites_final_esm2_multisource_1_30_full2795.pt'
old=json.loads((R/'analysis/error_evidence_linkage_20261025/all_training_linked.json').read_text(encoding='utf-8-sig'));byid={r['sample_id']:r for r in old}
c=torch.load(cache,map_location='cpu',weights_only=False,mmap=True);enc=mini3di.Encoder();rows=[];h=hashlib.sha256();checks=[]
def encode(b,m):
 xyz=np.array(b,dtype=np.float64);mask=np.array(m,dtype=bool)
 n=xyz[:,0,:].copy();ca=xyz[:,1,:].copy();carbon=xyz[:,2,:].copy()
 for a,k in [(n,0),(ca,1),(carbon,2)]:a[~mask[:,k]]=np.nan
 # The package reconstructs missing CB from N/CA/C. This is an explicit approximation.
 return enc.encode_atoms(ca=ca,cb=np.full_like(ca,np.nan),n=n,c=carbon)
for i,r in enumerate(c['records']):
 sid=r['sample_id'];ref=byid[sid];assert r['sequence']==ref['sequence'] and int(r['label'])==int(ref['label'])
 b=np.array(r['backbone_coords'].tolist());m=np.array(r['backbone_mask'].tolist(),dtype=bool);pl=np.array(r['plddt'].tolist(),dtype=np.float64)
 assert b.shape==(len(r['sequence']),5,3) and m.shape==b.shape[:2]
 assert np.isfinite(b).all() and np.isfinite(pl).all()
 if pl.max()<=1.00001:pl=pl*100
 assert pl.min()>=0 and pl.max()<=100.01
 states=encode(b,m);string=enc.build_sequence(states).lower();invalid=np.ma.getmaskarray(states);masked=invalid|(pl<70)
 letters=['#' if bad else ch for ch,bad in zip(string,masked)]
 combined=''.join(aa+ch for aa,ch in zip(r['sequence'],letters));allmasked=''.join(aa+'#' for aa in r['sequence'])
 assert len(letters)==len(r['sequence']) and set(letters)<=set('acdefghiklmnpqrstvwy#')
 row=dict(sample_id=sid,sequence=r['sequence'],label=int(r['label']),fold=int(ref['fold']),group_id=ref['group_id'],combined=combined,all_structure_masked=allmasked,length=len(letters),structural_invalid_n=int(invalid.sum()),low_plddt_n=int((pl<70).sum()),unmasked_structure_n=int((~masked).sum()),backbone_N_CA_C_complete=bool(m[:,:3].all()),cb_policy='allCBmissing;mini3di reconstructs fromN_CA_C;cached centroid NOTused')
 rows.append(row);h.update(sid.encode());h.update(b[:,:4].tobytes());h.update(m[:,:4].tobytes());h.update(pl.tobytes())
 if i in [0,3193,6386]:
  rotated=b.copy();rotated[:]=b@np.array([[0,-1,0],[1,0,0],[0,0,1]])+np.array([17.,-31.,4.]);again=encode(rotated,m)
  assert np.array_equal(np.ma.getmaskarray(again),invalid)
  assert np.array_equal(again.compressed(),states.compressed())
  checks.append(dict(sample_id=sid,rigid_rotation_translation_states_equal=True))
assert len(rows)==6387
summary=dict(n=6387,coordinates_source=str(cache),coordinate_field_fingerprint_sha256=h.hexdigest(),mini3di_version=mini3di.__version__,backbone_complete_n=sum(r['backbone_N_CA_C_complete'] for r in rows),entirely_structure_masked_n=sum(r['unmasked_structure_n']==0 for r in rows),total_residues=sum(r['length'] for r in rows),unmasked_structure_residues=sum(r['unmasked_structure_n'] for r in rows),rigid_transform_checks=checks,scope='backbone-reconstructed3Di,notfull-nativeCB Foldseek equivalence; frozen predicted structure, not experimental biology labels',official_mask_threshold_plddt=70,raw_pdb_required_not_recovered=True,sidechain_centroid_not_used_as_CB=True)
for n,obj in [('structure_sequences.json',rows),('input_summary.json',summary)]: (O/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
