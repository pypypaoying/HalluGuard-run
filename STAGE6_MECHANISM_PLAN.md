# Stage 6 Mechanism Validation Plan

## Goal

Validate three next-line HalluGuard mechanisms on the existing compact real-forecast assets before committing to any larger TableA implementation:

1. Mirror Residual Corrector (MRC): test whether post-LRBN residuals are learnable and calibratable.
2. Trajectory Arbitration Engine (TAE): test whether multiple target-free correction experts provide a real oracle upper bound and whether a router/ranker can capture part of it.
3. Frequency Online Meta-Calibrator (FOMC): test whether matured-label residual frequency drift supports leakage-safe online calibration.

This is a compact mechanism validation stage, not a paper-scale table.

## Fixed Contract

- Input: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
- Scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192`, seed `2026`
- Split contract: validation may train/select policies; test may only evaluate.
- Parent baseline: HalluGuard-LRBN.
- Stage 5 comparators: SRA-BP-safe and SRA-BP-balanced from `experiments/halluguard/results/lrbn_sra_bp_stage5/`.
- Output: `experiments/halluguard/results/stage6_mechanism/`.
- Required subdirs: `mrc/`, `tae/`, `fomc/`.
- Leakage flag must remain `False`.

## Implementation Map

- Add `experiments/halluguard/halluguard_stage6_mechanism.py` with reusable utilities for features, MRC, TAE, and FOMC.
- Add `experiments/halluguard/run_stage6_mechanism.py` as the fixed Stage 6 runner.
- Add `STAGE6_MECHANISM_VALIDATION_REPORT.md` after the run.
- Update `CANDIDATE_BOARD.md` and `results_halluguard.tsv` after metrics are verified.

## Experiments

### MRC

- Train ridge residual heads by horizon on validation.
- Compare LRBN, mean residual, ridge residual, risk-abstained ridge, SRA-safe, SRA-balanced.
- Calibrate residual intervals from validation residual errors.
- Report point results, quantile calibration, abstention curve, slices, bootstrap CI, verdict.

### TAE

- Generate interpretable candidate trajectories: keep LRBN, raw, SRA-safe, SRA-balanced, BP-always, level bias, phase shifts, amplitude scaling, volatility shrink, smoothing controls, and ensemble median.
- Compute oracle best-of-experts upper bound.
- Train target-free router and candidate ranker on validation.
- Report oracle gain, failure-mode separability, router/ranker metrics, decision-level evaluation, verdict.

### FOMC

- Build chronological replay with validation as matured historical buffer and test as online event stream.
- Compare no-update, rolling mean residual, time EMA residual, and spectral residual adapter.
- Report spectral autocorrelation, online adapter results, conformal coverage, protocol guard, verdict.

## Success Interpretation

- MRC Go: residual point correction improves LRBN by at least 1%, calibrated intervals are close, abstention reduces harm, and at least one non-SRA slice improves.
- TAE Go: oracle expert upper bound is large and router/ranker captures useful gain beyond boundary-only experts.
- FOMC Go: spectral online adapter beats LRBN and rolling residual baselines without protocol violations.

If a line fails, keep the artifact as mechanism evidence and do not promote it.

## Revision Log

- 2026-06-28: Created Stage 6 compact mechanism validation plan.
