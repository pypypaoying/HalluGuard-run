# Stage 12 CGA Low-Harm Priority Validation Plan

## Objective

Validate the priority plan from `deep-research-report (1).md`: keep the Stage 10 CGA candidate-generation mechanism, but repair deployable arbitration with sparse family mixture, residual-family quantile simplex, pairwise/no-harm selective admission, and uncertainty-conditioned lambda plus boundary veto.

## Compact Protocol

- Scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192` x seed `2026`.
- Baseline: frozen `HalluGuard-LRBN`.
- Candidate pool: Stage 10 CGA deployable candidates.
- Calibration: validation-only inner calibration; no test threshold tuning.
- Evaluation: test-only.
- Bootstrap: `2000`.
- Stop rule: if compact gate fails, do not run mini-extension or TableA candidate.

## Variants

1. `Sparse-Family-CGA`: sparse top-k family admission with median family representatives.
2. `Sparse-Residual-Simplex-CGA`: same sparse family admission, but residual-distribution family uses score-weighted quantile simplex over residual candidates.
3. `NoHarm-Selective-CGA`: adds validation-selected expected-harm gating.
4. `LambdaVeto-CGA`: adds boundary-aware smoothing veto and uncertainty-conditioned lambda shrink.

Baselines in the same output:

- `LRBN`
- `SRA-BP-balanced`
- `oracle_stage12_cga_full`

## Compact Gates

Primary final gate for moving to mini-extension:

- MSE delta vs LRBN is negative.
- Bootstrap upper delta is below zero.
- Oracle gain fraction is at least `0.08`.
- Max config harm is at most `0.18`; stricter target for final lambda/veto is `0.12`.
- Known harmed config `ETTm1 / DLinear / 192` is not harmed on mean MSE.
- Boundary-heavy slice does not worsen on mean MSE.
- Test threshold leakage is `False`.

## Outputs

- `experiments/halluguard/results/stage12_cga_low_harm/`
- `STAGE12_CGA_LOW_HARM_REPORT.md`
- Updated `CANDIDATE_BOARD.md`
- Appended `results_halluguard.tsv`

