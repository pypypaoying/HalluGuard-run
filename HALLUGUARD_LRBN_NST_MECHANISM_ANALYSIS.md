# HalluGuard-LRBN + NST Mechanism Analysis

## Question

NST-style branching gave a small but consistent gain on top of
`HalluGuard-LRBN unified_revin_rdn_hybrid`, especially for PatchTST. The goal of
this analysis is to identify what part of the correction is responsible before
optimizing structure.

## Evidence Files

- Main prediction table:
  `experiments/halluguard/results/halluguard_lrbn_nst_main/lrbn_metrics.csv`
- Parent comparison:
  `experiments/halluguard/results/halluguard_lrbn_nst_main/parent_comparison.json`
- New decomposition:
  `experiments/halluguard/results/halluguard_lrbn_nst_main/nst_gain_group_decomposition.csv`
- Config decomposition:
  `experiments/halluguard/results/halluguard_lrbn_nst_main/nst_gain_config_decomposition.csv`
- Sample decomposition:
  `experiments/halluguard/results/halluguard_lrbn_nst_main/nst_gain_sample_decomposition.csv`
- RevIN paper text extraction:
  `experiments/halluguard/results/halluguard_lrbn_nst_main/revin_openreview_text.txt`

## Decomposition Method

For each test prediction error vector `e = prediction - target`, decompose MSE
into three orthogonal components:

- `level`: projection onto a constant vector, capturing mean/level shift.
- `trend`: projection onto a centered linear ramp, capturing first-order trend
  mismatch.
- `shape`: residual MSE after removing level and linear trend.

Then compare `unified_revin_rdn_hybrid` against `lrbn_nst_feature_gate`.

## Main Finding

The NST-gated gain is mostly a level/mean restoration gain, not a frequency or
shape gain.

| group | MSE gain | level share | trend share | shape share | improved sample rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| ALL | 0.029212 | 74.74% | 12.52% | 12.74% | 52.23% |
| PatchTST | 0.065479 | 76.60% | 14.56% | 8.84% | 54.27% |
| DLinear | -0.007055 | mostly level/trend harm | mostly harm | small shape gain | 50.20% |
| ETTm1 | 0.014644 | 93.04% | -5.86% | 12.82% | 50.51% |
| ETTh1 | 0.043780 | 68.62% | 18.67% | 12.71% | 53.96% |

PatchTST gets the useful part:

- PatchTST mean MSE gain over LRBN parent: `0.065479`.
- About `0.050159` of that comes from level error reduction.
- Trend contributes `0.009533`.
- Shape contributes only `0.005787`.

Largest config-level gain:

- `ETTh1 / PatchTST / 720`: MSE gain `0.327357`; level gain `0.273580`.

This suggests the gate is not primarily learning high-frequency repair. It is
selecting a branch that better restores or shifts the future level.

## RevIN Paper Takeaways

Source: Kim et al., "Reversible Instance Normalization for Accurate Time-Series
Forecasting against Distribution Shift", ICLR 2022, official OpenReview PDF and
official GitHub implementation.

What RevIN explicitly targets:

- Time series often have changing mean and variance.
- RevIN removes instance mean/std at input and restores the same statistics at
  output.
- The paper describes baseline predictions as shifted and scaled, and frames
  RevIN as correcting those distribution errors.

Important limitation exposed by the paper's own motivation:

- Plain input normalization can remove non-stationary information that is useful
  for forecasting.
- Without denormalization, the model must reconstruct the original distribution
  only from normalized input.
- RevIN fixes this by restoring the removed input statistics, but this also
  means the denormalization statistics are inherited from the input window.

Important assumption:

- The future mean and variance can be represented as offsets from input
  statistics. The appendix states the model focuses on learning the difference
  from input distribution to output distribution.

Important sensitivity:

- RevIN computes mean/std across the entire input sequence.
- The appendix states input sequence length is a crucial hyperparameter because
  RevIN uses those statistics in normalization and denormalization.

Official implementation facts:

- The official layer computes per-instance temporal mean and stdev.
- It detaches those statistics.
- It optionally applies learnable affine parameters.
- Denormalization uses the same mean/stdev from the input window.

## Implication For HalluGuard-LRBN

`unified_revin_rdn_hybrid` is already a principled RevIN improvement:

```text
center = beta * boundary_anchor + (1-beta) * instance_mean
scale  = gamma * robust_tail_scale + (1-gamma) * instance_std
```

This directly addresses a RevIN weakness: the full-window instance mean/std may
not be the best statistics to restore for the future, especially under local
boundary level changes.

The NST feature gate improves mostly by choosing a branch with better level
restoration for PatchTST. That means the right optimization direction is likely:

1. Learn or calibrate the future `center`/anchor better.
2. Use tail/boundary/trend-aware anchors instead of full-window mean alone.
3. Make restoration model-family-safe, because DLinear already benefits from
   LRBN and can be harmed by extra branch mixing.
4. Treat scale as secondary until decomposition shows stronger scale-driven
   gains.

## Next Candidate Directions

### 1. Future-Level Restoration Gate

Replace the generic NST branch gate with a specialized level-restoration module:

```text
center_future = center_lrbn + g(context_features) * delta_center
```

Candidate `delta_center` sources:

- last value minus instance mean
- tail median minus instance mean
- short-horizon linear extrapolated anchor minus boundary anchor
- validation-learned blend of boundary anchor and instance mean

Expected benefit: keep PatchTST's level correction while avoiding DLinear harm.

### 2. Anchor Selection Instead Of Branch Blending

Rather than blending two full model predictions, keep one base forecast and
learn only which reversible center to restore:

```text
z = (x - selected_center) / selected_scale
y_hat = model(z) * selected_scale + selected_center
```

This is cleaner than `lrbn_nst_feature_gate`, because the mechanism is directly
about reversible statistic choice.

### 3. Horizon-Aware Center Drift

The strongest gain is on `ETTh1 / PatchTST / 720`, and the RevIN paper notes
long sequence forecasting benefits strongly from denormalization. Add a small
horizon ramp to the restored center:

```text
center_t = center + ramp[t] * learned_drift(context_features)
```

Guardrail: only train on train split; val/test reporting only.

### 4. Scale Gate As Secondary Ablation

Run a scale-only restoration ablation after center variants:

```text
scale = gamma * robust_tail_scale + (1-gamma) * instance_std
```

The current decomposition does not make scale the leading explanation, so this
should not be the first optimization.

## Working Verdict

NST did not introduce a broad new mechanism. It exposed where LRBN can still be
improved: future level/statistic restoration for PatchTST-like forecasts.

Recommended parent remains:

```text
HalluGuard-LRBN unified_revin_rdn_hybrid
```

Recommended next research line:

```text
HalluGuard-LRBN Future-Level Restoration
```

Do not optimize another generic NST stack before testing a targeted center/level
restoration module.
