# ProtTox-SpectralRefine v1: frozen OOF decision

## Scope

This is a training-OOF-only comparison on the existing audited 6,387-record
Frozen ProtT5 cache and five-fold group manifest. Test1 and Test2 were not read
or used for feature, model, threshold, or promotion selection.

## Candidate

The candidate adds six deterministic physicochemical DCT-II channels (66
features) to a frozen ProtT5 TriStat + PhysChem Stage-1 MLP by a zero-initialized
gated logit residual. It is one user-visible prediction, but internally a
coarse-logit plus refinement topology. It does not use ESMFold, 3D inputs,
GNNs, LoRA, Mamba, or structural fusion.

## Paired gated-residual result versus the retained MLP baseline

| Seed | Delta MCC | Delta AUPRC | Delta Brier |
|---:|---:|---:|---:|
| 20260905 | -0.000664 | +0.000062 | -0.000724 |
| 20260906 | +0.001074 | +0.000621 | -0.000460 |
| Mean | +0.000205 | +0.000341 | -0.000592 |

The frozen promotion rule required non-lower MCC in both seeds, mean MCC
advantage at least `+0.005`, and mean AUPRC decline no worse than `-0.002`.
The candidate fails the first two conditions. Decision: **retain as ablation**.

## Required controls

- Spectral-only: MCC 0.2114 / 0.1983. The DCT descriptors alone are not a
  viable toxicity predictor.
- Ungated concatenation: MCC 0.8199 / 0.8232. It has good calibration scores
  but does not establish an MCC advantage over the paired MLP baseline.

No Test1 audit is authorized by this result. The next architecture must be
frozen before another OOF run; do not search DCT/FFT/wavelet settings using the
current candidate outcomes.
