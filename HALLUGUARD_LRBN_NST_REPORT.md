# HalluGuard-LRBN + NST Complementarity Report

Status: implementation in progress.

## Parent

Main claim-clean parent: `unified_revin_rdn_hybrid`.

## Candidate Variants

- `unified_revin_rdn_hybrid`: parent baseline.
- `nst_lightweight`: shared-runner NST-style stationarization baseline.
- `lrbn_unified_nst_residual`: claim-clean residual stationarization on top of
  LRBN normalized coordinates.
- `lrbn_nst_output_blend`: diagnostic train-split blend of LRBN and NST
  branches.
- `lrbn_nst_feature_gate`: train-split context-feature gate between LRBN and NST
  branches.

## Results

### Smoke 1

Command:

```powershell
python scripts\run_halluguard_lrbn_nst.py --datasets ETTm1 --models DLinear,PatchTST --horizons 96 --variants unified_revin_rdn_hybrid,nst_lightweight,lrbn_unified_nst_residual,lrbn_nst_output_blend --data-root external\ETDataset --prediction-dir baseline_predictions\halluguard_lrbn_nst_smoke --raw-prediction-dir baseline_predictions\halluguard_lrbn_nst_smoke_raw --output-dir experiments\halluguard\results\halluguard_lrbn_nst_smoke --epochs 1 --max-train-windows 128 --max-eval-windows 32 --device cpu --continue-on-error
```

Observed:

- DLinear: parent `unified_revin_rdn_hybrid` MSE `3.089931`; residual NST
  candidate MSE `3.139437`; diagnostic output blend MSE `3.113672`.
- PatchTST: parent MSE `3.286239`; residual NST candidate MSE `3.291171`;
  diagnostic output blend MSE `3.163492`.

Interpretation: residual stationarization does not improve the LRBN parent in
this first smoke, suggesting redundancy between NST-style stationarization and
LRBN center/scale. The diagnostic output blend improves PatchTST, so there may
be branch complementarity worth testing with a deployable context gate.

## Claim Boundary

This experiment does not claim to reproduce the full Non-stationary Transformer
architecture unless the official attention de-stationary factors are integrated
and evaluated. The first pass tests whether NST-style stationarization is
complementary to HalluGuard-LRBN under the existing fair adapter protocol.
