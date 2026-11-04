"""Generate matched pLDDT condition configs; does not train or certify provenance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def make_conditions(base):
    model = base['model']
    if not model.get('chemical_site_branch', False):
        raise ValueError('Baseline must include the chemical-site branch')
    if not model.get('chemical_site_use_plddt', True):
        raise ValueError('Baseline must retain pLDDT chemical-site inputs')
    if not model.get('chemical_site_use_plddt_weighting', True):
        raise ValueError('Baseline must enable chemical-site confidence weighting')
    if model.get('fusion_gate', 'plddt').lower() in {'none', 'off', 'disabled'}:
        raise ValueError('Baseline must enable fusion gating')
    if float(model.get('plddt_edge_min', .25)) >= 1:
        raise ValueError('Baseline EGNN confidence weighting is already disabled')
    out = {}
    for arm, disable_weight, disable_gate in (('A',False,False),('B',True,False),('C',False,True),('D',True,True)):
        cfg = copy.deepcopy(base)
        if disable_weight:
            cfg['model']['plddt_edge_min'] = 1.0
            cfg['model']['chemical_site_use_plddt_weighting'] = False
        if disable_gate:
            cfg['model']['fusion_gate'] = 'none'
        out[arm] = cfg
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-config', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    source = Path(args.base_config)
    raw = source.read_bytes()
    base = json.loads(raw.decode('utf-8-sig'))
    configs = make_conditions(base)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for arm, cfg in configs.items():
        target = out / f'{arm}.json'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
    for arm, cfg in configs.items():
        (out / f'{arm}.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    manifest = {'base_config': str(source.resolve()), 'base_sha256': hashlib.sha256(raw).hexdigest(),
                'status': 'configuration_only_not_trained', 'conditions': {
                    'A': 'baseline', 'B': 'disable explicit EGNN and chemical-site confidence multipliers',
                    'C': 'disable fusion node/global gates', 'D': 'B and C'},
                'warning': 'Conditional ablation; pLDDT remains in features and possibly coordinate noise. '
                           'Upstream training provenance, cross-fold isolation, and paired retraining must be established.'}
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    main()
