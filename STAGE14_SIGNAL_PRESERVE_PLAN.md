# Stage 14 Signal-Preserving Component Router Plan

Date: 2026-05-15

## Problem Anchor

Design a signal-preserving test-time correction/router so HalluGuard keeps the boundary-repair advantage while reducing mistaken smoothing of real high-frequency, turning, and local-change signal, and narrowing the gap to full smoothing without merely becoming full smoothing.

## Method Thesis

`HalluGuard Signal-Preserving Component Router` decomposes context and prediction evidence into boundary, trend, high-frequency residual, and local curvature components. It uses validation-only thresholds to route each sample to boundary repair, smoothing, or abstention. Smoothing is allowed only when high-frequency roughness is unsupported by the context; high-frequency or local-change patterns that look context-supported are preserved.

## Non-Goals

- Do not train a new forecasting model.
- Do not modify DLinear, PatchTST, dataloaders, loss, hidden states, or model internals.
- Do not run a large TSFM experiment.
- Do not package simple full smoothing as a new mechanism.
- Do not use test targets to choose thresholds, actions, or variants.
- Do not treat the compact external fixture as scientific generalization evidence.

## Primary Claim

A black-box, validation-calibrated component router can reduce over-smoothing by distinguishing unsupported prediction roughness from context-supported signal changes, while preserving the proven boundary-discontinuity correction mechanism.

## Anti-Claims To Rule Out

- The gain is just median/naive/EMA smoothing.
- The gain is just `boundary_then_median` under another name.
- The module suppresses true PatchTST signal and creates external-fixture harm.
- The module uses test thresholds or test labels.
- The module degenerates to one action on more than 90% of samples.

## Signal-Support Design

Per sample, compute target-free features from context and prediction:

- boundary score and boundary signed gap;
- low-frequency trend mismatch;
- high-frequency energy ratio and high-frequency excess;
- spectral distance;
- diff-sign continuation between context tail and prediction head;
- autocorrelation/period support from context tail;
- local turning/curvature support;
- prediction/context variance and diff-std ratios;
- context volatility and horizon.

Derived gates:

- `boundary_zone`: boundary discontinuity is high; prefer `boundary_only`.
- `unsupported_hf_zone`: prediction roughness is high and context support is low; allow smoothing.
- `supported_hf_zone`: prediction roughness is high but context supports volatility, periodicity, or turning; preserve/no-correction unless boundary is also high.
- `uncertain_zone`: abstain/no-correction.

## Variants To Implement

Required variants:

- `no_correction`
- `boundary_only`
- `median_smoothing`
- `naive_smoothing`
- `ema_smoothing`
- `stage13_rule_router`
- `signal_preserve_router`
- `component_router_without_signal_support`
- `signal_support_only_ablation`
- `matched_smoothing_control`
- `random_action_router`
- `validation_best_single_action`
- `oracle_test_ceiling` diagnostic only

## Experiment Order

1. Smoke table on:
   - ETTm1/DLinear/96
   - ETTm1/PatchTST/720
   - ETTh1/DLinear/720
   - ETTh1/PatchTST/336
2. Clean full 16-config table.
3. Stress table with six Stage 13 stress types.
4. External batch fixture smoke and harm diagnostic by model family.
5. Audit and result-to-claim.

## Success Gates

`signal_preserve_router` passes Stage 14 only if:

- clean full completes `16/16`;
- `test_threshold_leakage=False`;
- clean mean MSE delta is better than Stage 13 `rule_router` (`-1.289319%`);
- PatchTST clean mean beats Stage 13 PatchTST (`-0.297552%`), ideally at least `-0.5%` or with clear harm reduction;
- DLinear clean mean is not materially below Stage 13 DLinear (`-2.281087%`);
- external PatchTST harm drops from `4/8` to at most `1/8`, or failure is clearly explained;
- boundary-discontinuity mechanism is retained;
- high-frequency perturbation is not purely full-smoothing-explained, ideally beating matched smoothing;
- dominant single action rate stays below `0.90`;
- max MSE/MAE harm stays below `3%`.

## Failure Interpretation

If Stage 14 fails while Stage 13 still passes, the project should narrow its claim: HalluGuard is a boundary/dynamics reliability layer, not a general point-MSE enhancer or signal-preserving smoother. If it is partial, the next round should focus on the smallest failing slice, not a broader external table.
