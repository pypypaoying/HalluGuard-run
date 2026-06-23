# Stage 13 Result To Claim

## Claim Supported

`partial`

## What The Results Support

Stage 13 supports a narrowed adaptive-router claim:

A validation-only, feature-based router can choose among boundary repair,
smoothing, and no correction to improve balanced robustness over the Stage 12
single `boundary_only` rule while preserving boundary-discontinuity mechanism
evidence.

Evidence:

- Clean full table completed `16/16` configs with no test threshold leakage.
- `rule_router` clean mean MSE delta is `-1.289319%`.
- `rule_router` improves `16/16` clean configs.
- `rule_router` beats matched random action routing in `15/16` clean configs
  with paired win rate `0.9500`.
- `rule_router` beats matched sparse smoothing in `12/16` clean configs.
- `rule_router` beats Stage 12 `boundary_only` in `15/16` clean configs.
- Boundary-discontinuity stress completed `16/16`, with MSE delta
  `-1.482901%`, paired random wins `15/16`, and matched-smoothing wins
  `15/16`.
- The action distribution is not single-action degenerate on the clean full
  table: max action rate is `0.8789`.
- Boundary action alignment is interpretable: boundary action rate is `0.6173`
  in the high boundary-score bin and `0.0000` in low/mid bins.

## What The Results Do Not Support

The results do not support a broad claim that HalluGuard replaces smoothing for
point-MSE optimization:

- `median_smoothing` clean MSE delta is `-1.886801%`, stronger than
  `rule_router` at `-1.289319%`.
- `naive_smoothing` and `ema_smoothing` are also strong clean baselines.
- High-frequency perturbation and variance-shift stress improve under the
  router, but matched/full smoothing controls remain stronger or explain much
  of the gain.
- The Stage 12 compact external fixture has weaker paired-random evidence
  (`9/16`, paired win `0.6000`), so it should be treated as integration smoke,
  not claim-level evidence.

The results also do not support a frequency-repair claim. The strongest
mechanism evidence remains boundary/dynamics continuity.

## Paper Claim Recommendation

Narrow the claim to:

> HalluGuard-Dynamics is a validation-calibrated post-processing router that
> uses context/prediction dynamics features to select boundary repair,
> smoothing, or abstention. On ETT DLinear/PatchTST prediction tables, it
> improves clean and boundary-stress robustness over boundary-only repair,
> random action routing, and matched sparse smoothing controls, while full
> smoothing remains a stronger point-MSE baseline in several settings.

Avoid claiming:

- superiority over median smoothing on pure MSE;
- frequency repair;
- universal forecast improvement beyond the tested ETT prediction tables;
- external generalization from the compact fixture alone.

## Next Experiment

Use Stage 13 as the next parent line, but test it on truly external prediction
tables with more rows per split:

- ETTm2 or ETTh2, if predictions are available;
- Weather or Electricity predictions from an external TSF framework;
- at least one non-DLinear/PatchTST model family.

The next key question is whether the router's clean and boundary-stress
advantage over random action routing remains stable outside the current ETTm1
and ETTh1 prediction assets.
