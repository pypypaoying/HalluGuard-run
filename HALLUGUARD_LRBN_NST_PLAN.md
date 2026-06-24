# HalluGuard-LRBN + NST Complementarity Plan

## Parent Line

The claim-clean parent is `HalluGuard-LRBN unified_revin_rdn_hybrid`.
`fixed_hybrid_output_blend` remains diagnostic only.

## Research Question

Can NST-style stationarization add complementary signal on top of the
`unified_revin_rdn_hybrid` reversible boundary normalizer, without simply
duplicating RevIN/LRBN center-scale normalization?

## Fixed Comparison Contract

- Datasets: ETTm1, ETTh1
- Backbones: DLinear, PatchTST
- Horizons: 96, 192, 336, 720
- seq_len: 96
- splits: train for fitting model and trainable normalizer parameters, val/test
  for exported prediction contract and reporting only
- Primary comparison: candidate MSE/MAE versus `unified_revin_rdn_hybrid`
- Secondary comparison: raw, lightweight NST, and diagnostic output blend
- Test threshold leakage: must remain False

## First Candidate Family

1. `lrbn_unified_nst_residual`
   - Use the LRBN hybrid center and scale.
   - Normalize context by LRBN.
   - Apply a second NST-style stationarization only inside the normalized
     residual coordinates.
   - De-stationarize from NST residual space, then reverse LRBN.
   - This is claim-clean if it improves, because it tests whether local residual
     stationarity complements boundary/instance reversible normalization.

2. `lrbn_nst_output_blend`
   - Train two branches: `unified_revin_rdn_hybrid` and lightweight NST.
   - Learn one train-split blend weight.
   - Diagnostic only, because it is an ensemble-style upper-bound on
     complementarity rather than a single normalization mechanism.

3. `lrbn_nst_feature_gate`
   - Train two branches, but choose the LRBN/NST blend from target-free context
     features: robust-vs-instance scale, boundary displacement, tail-median
     displacement, roughness ratio, and last derivative.
   - This is a deployable candidate if it improves over the parent without
     becoming just a fixed output blend.

## Smoke

```bash
LRBN_NST_VARIANTS=unified_revin_rdn_hybrid,nst_lightweight,lrbn_unified_nst_residual,lrbn_nst_output_blend,lrbn_nst_feature_gate \
DATASET_SET=ETTm1 MODELS=DLinear,PatchTST HORIZONS=96 \
EPOCHS=1 MAX_TRAIN_WINDOWS=128 MAX_EVAL_WINDOWS=32 \
  bash scripts/run_halluguard_lrbn_nst_table.sh
```

Promote only if a candidate improves over `unified_revin_rdn_hybrid` or clearly
reduces PatchTST harm without weakening DLinear by more than a small smoke-level
margin.

## Full Run

```bash
LRBN_NST_VARIANTS=unified_revin_rdn_hybrid,nst_lightweight,lrbn_unified_nst_residual,lrbn_nst_output_blend,lrbn_nst_feature_gate \
DEVICE=cuda EPOCHS=10 MAX_TRAIN_WINDOWS=8192 MAX_EVAL_WINDOWS=1024 \
  bash scripts/run_halluguard_lrbn_nst_table.sh
```

## Interpretation

- If `lrbn_unified_nst_residual` wins, LRBN and NST residual stationarity are
  complementary.
- If only `lrbn_nst_output_blend` wins, there is ensemble complementarity but
  not yet a clean mechanism.
- If neither wins, NST-style stationarization is likely redundant with LRBN's
  center-scale normalization in this setup.
