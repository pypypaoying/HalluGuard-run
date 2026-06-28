# Stage 11 CGA-HM Plan

## Objective

Validate the first phase proposed by `deep-research-report (2).md`:

> Can family-level soft mixture with validation-only harm-aware admission safely capture more of the Stage 10 CGA oracle space than hard candidate selection?

This is the **mechanism validation** phase only. If it fails the go/no-go gate, do not run mini-extension or TableA candidate stages.

## Parent Evidence

- Stage 10 full CGA oracle MSE: `3.074767`.
- Stage 10 deployable Safe/Balanced-CGA MSE delta vs LRBN: `-1.274774%`.
- Stage 10 oracle gain fraction captured by deployable selector: `0.034291`.
- Stage 10 max config harm: `0.260417`.
- Stage 10 family top-2 hit: `0.619792`.
- Stage 10 candidate top-2 hit: `0.093750`.

## Compact Protocol

- Parent baseline: frozen `HalluGuard-LRBN`.
- Scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192`, seed `2026`.
- Split: existing `val`/`test`; validation split is inner-train/inner-calib.
- Calibration: validation-only.
- Evaluation: test-only.
- Bootstrap: `2000`.
- Test threshold leakage: must remain `False`.

## Required Mechanism Modules

1. Shared-safe expert family mixture.
2. Harm-aware family admission layer.
3. Residual-quantile candidate bank.

Stage 11 may include boundary-veto variants inside the same mechanism stage, but only as validation-calibrated variants; no test tuning is allowed.

## Stage 1 Gate

Proceed to mini-extension only if the best deployable Stage 11 policy satisfies:

- `oracle_gain_fraction >= 0.08`.
- `max_config_harm <= 0.18`.
- MSE delta vs LRBN remains negative.
- Test threshold leakage is `False`.

If the gate fails, stop and report the failure reason.

## Outputs

- `experiments/halluguard/results/stage11_cga_hm/`
- `STAGE11_CGA_HM_REPORT.md`
- Updated `CANDIDATE_BOARD.md`
- Appended `results_halluguard.tsv`

