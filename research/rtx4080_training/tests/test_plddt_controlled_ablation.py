import copy
import importlib.util
from pathlib import Path

import torch

from ghxtox.models.chemical_sites import ChemicalSiteInteractionBranch
from ghxtox.models.layers import PLDDTAwareFusion, PLDDTAwareEGNNLayer, RadialBasis


def test_message_switch_preserves_features_and_changes_only_confidence_weights():
    torch.manual_seed(41)
    branch = ChemicalSiteInteractionBranch(hidden_dim=8, site_hidden_dim=8, dropout=0.0)
    other = copy.deepcopy(branch)
    other.use_plddt_weighting = False
    assert branch.state_dict().keys() == other.state_dict().keys()
    coords = torch.tensor([[0.,0.,0.],[1.,0.,0.]])
    kwargs = dict(coordinates=coords, orientations=torch.ones(2,3),
                  orientation_mask=torch.ones(2, dtype=torch.bool), types=torch.ones(2,8),
                  confidence=torch.tensor([.25, 1.]), source=torch.tensor([0]), target=torch.tensor([1]),
                  distances=torch.tensor([1.]), radius_of_gyration=torch.tensor(1.))
    features_a, weights_a = branch._edge_features(**kwargs)
    features_b, weights_b = other._edge_features(**kwargs)
    torch.testing.assert_close(features_a, features_b, rtol=0, atol=0)
    torch.testing.assert_close(weights_a, torch.tensor([.5]))
    torch.testing.assert_close(weights_b, torch.tensor([1.]))


def test_gate_off_keeps_parameter_shapes_and_sets_both_gates_to_one():
    torch.manual_seed(42)
    gated = PLDDTAwareFusion(residue_feature_dim=4,hidden_dim=8,num_heads=2,dropout=0.,gate_mode='learned_confidence')
    ungated = copy.deepcopy(gated)
    ungated.gate_mode = 'none'
    assert gated.state_dict().keys() == ungated.state_dict().keys()
    seq, spatial, res = torch.randn(1,3,8), torch.randn(1,3,8), torch.randn(1,3,4)
    args = (seq, spatial, res, torch.tensor([[.2,.6,.9]]), torch.ones(1,3,dtype=torch.bool))
    original, a = gated(*args)
    disabled, c = ungated(*args)
    torch.testing.assert_close(c['node_gate'], torch.ones_like(c['node_gate']))
    torch.testing.assert_close(c['global_gate'], torch.ones_like(c['global_gate']))
    assert not torch.equal(a['node_gate'], c['node_gate'])
    assert not torch.equal(original, disabled)


def test_egnn_switch_keeps_message_inputs_and_removes_confidence_multiplier():
    torch.manual_seed(11)
    weighted = PLDDTAwareEGNNLayer(hidden_dim=8, rbf_bins=4, dropout=0., plddt_edge_min=.1)
    unweighted = copy.deepcopy(weighted)
    unweighted.plddt_edge_min = 1.
    captured = []
    weighted.message_mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    unweighted.message_mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    h = torch.randn(1, 3, 8)
    coords = torch.tensor([[[0.,0.,0.],[2.,0.,0.],[0.,3.,0.]]])
    plddt = torch.tensor([[.2,.6,.9]])
    mask = torch.ones(1,3,dtype=torch.bool)
    rbf = RadialBasis(4, 12.)
    result_a = weighted(h,coords,plddt,rbf,mask)
    result_b = unweighted(h,coords,plddt,rbf,mask)
    torch.testing.assert_close(captured[0],captured[1],rtol=0,atol=0)
    assert not torch.allclose(result_a[0],result_b[0])


def test_config_matrix_is_only_predeclared_fields():
    path = Path(__file__).parents[1]/'analysis/reviewer_response_20261102/prepare_plddt_ablations.py'
    spec = importlib.util.spec_from_file_location('prepare_plddt', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = {'model': {'chemical_site_branch': True, 'fusion_gate': 'learned_confidence',
                      'plddt_edge_min': .1}, 'train': {'seed': 42}}
    configs = mod.make_conditions(base)
    assert configs['A'] == base
    assert {k for k in configs['B']['model'] if configs['B']['model'].get(k) != base['model'].get(k)} == {'plddt_edge_min','chemical_site_use_plddt_weighting'}
    assert {k for k in configs['C']['model'] if configs['C']['model'].get(k) != base['model'].get(k)} == {'fusion_gate'}
    assert {k for k in configs['D']['model'] if configs['D']['model'].get(k) != base['model'].get(k)} == {'plddt_edge_min','chemical_site_use_plddt_weighting','fusion_gate'}
    assert all(configs[k]['train'] == base['train'] for k in configs)
