"""Extract structure confidence from the actual cached records; CPU, no model inference."""
import csv
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ASSETS = Path('D:/dachuang2026/GHXTox-ziliao')
FILES = {
    'training_oof': ASSETS / 'train_chemical_sites_final_esm2_multisource_1_30_full2795.pt',
    'test1': ASSETS / 'test1_chemical_sites_final_esm2.pt',
    'test2': ASSETS / 'test2_chemical_sites_final_esm2.pt',
}
EXPECTED_TRAIN_HASH = 'f8a6ddf7dd81198142ad8c30a0f7926de1967618b01720bd90cb483a45dfdb5b'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    prior_audit = json.loads((ROOT / 'analysis/reviewer_response_20261102/chemical_free_candidate_20261106/audit.json').read_text(encoding='utf-8'))
    cases = {}
    with (ROOT / 'analysis/six_seed_fusion_cys_20261106/cases.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = (row['split'], row['sample_id'])
            assert key not in cases
            cases[key] = row
    output, provenance = [], {}
    for split, path in FILES.items():
        digest = sha(path)
        if split == 'training_oof':
            assert digest == EXPECTED_TRAIN_HASH, 'training cache differs from 3D experiment'
        else:
            assert digest == prior_audit['datasets'][split]['input_cache_sha256'], 'test cache differs from deployment evaluation'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        records = payload['records']
        assert len(records) == sum(key[0] == split for key in cases)
        seen = set()
        for record in records:
            key = (split, record['sample_id'])
            assert key in cases and key not in seen
            row = cases[key]
            assert record['sequence'] == row['sequence'] and int(record['label']) == int(row['label'])
            p = [float(x) for x in record['plddt'].flatten().tolist()]
            assert len(p) == len(row['sequence']) and all(0 <= x <= 1 for x in p)
            output.append({'split': split, 'sample_id': key[1], 'mean_plddt': sum(p) / len(p),
                           'min_plddt': min(p), 'low_plddt_fraction': sum(x < .55 for x in p) / len(p),
                           'structure_source': record.get('structure_source', '')})
            seen.add(key)
        provenance[split] = {'path': str(path), 'sha256': digest, 'n': len(records),
                             'identity_checks': 'sample_id, sequence, original label',
                             'cache_hash_matches_prediction_run': True}
        del payload, records
    assert len(output) == len(cases)
    with (OUT / 'quality.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    (OUT / 'quality_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2))

if __name__ == '__main__':
    main()
