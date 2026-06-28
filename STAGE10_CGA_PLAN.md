# Stage 10 CGA Plan

## Objective

Validate **HalluGuard-CGA: Candidate Generation + Hierarchical Arbitration** as the next architecture-level direction after Stage 9.

The compact question is whether richer deployable trajectory candidates from smoothing-teacher, residual-distribution, and retrieval-memory families expand oracle space, and whether a family-aware arbitrator can safely capture part of that space.

## Protocol

- Parent baseline: frozen `HalluGuard-LRBN`.
- Scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192`, seed `2026`.
- Input: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`.
- Split: existing `val`/`test`; validation is split into inner-train and inner-calib.
- Calibration: validation-only.
- Evaluation: test-only.
- Bootstrap: configurable, default `2000`.
- Leakage flag: must remain `False`.

## Validation Points

1. Build old-pool, Stage 9 expanded-pool, and Stage 10 three-family candidate pools.
2. Measure restricted oracle space and family/candidate oracle distributions.
3. Test whether family-level top-k suitability is easier than candidate-level top-k suitability.
4. Calibrate Safe-CGA and Balanced-CGA policies on inner-calib only.
5. Evaluate overall, per-config, slice, selection, memory, bootstrap, and failure-case diagnostics on test.
6. Decide whether CGA is a deployable parent or only an oracle-space direction.

## Pass Gates

Mechanism pass:

- Stage 10 three-family oracle improves old deployable oracle by at least 5%.
- New families jointly account for at least 30% of oracle selections.
- Family top-2 hit exceeds candidate top-2 hit by at least 15pp.
- At least one new family improves a non-boundary slice in oracle analysis.

Safe-CGA pass:

- MSE delta vs LRBN <= -2.2%.
- Harm <= 0.03.
- Max config harm <= 0.10.
- Config improved ratio >= 0.75.
- Oracle gain fraction >= 0.15.
- Bootstrap upper bound for mean delta < 0.

Balanced-CGA pass:

- MSE delta vs LRBN <= -3.0%.
- Harm <= 0.08.
- Max config harm <= 0.18.
- Config improved ratio >= 0.75.
- Oracle gain fraction >= 0.20.

