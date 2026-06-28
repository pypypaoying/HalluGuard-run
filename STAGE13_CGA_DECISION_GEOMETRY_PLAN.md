# Stage 13 CGA Decision-Geometry Validation Plan

## Objective

Validate the architecture-level plan from `deep-research-report (3).md`: the bottleneck is not residual correction or safety itself, but candidate-level hard choice plus marginally calibrated safety. This stage tests whether decision-geometry changes can safely capture more Stage 10 CGA oracle space.

## Compact Protocol

- Scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192` x seed `2026`.
- Baseline: frozen `HalluGuard-LRBN`.
- Candidate pool: Stage 10 CGA deployable candidates.
- Calibration: validation-only inner calibration.
- Evaluation: test-only.
- Bootstrap: `2000`.
- Leakage rule: no test thresholds, no test policy selection.

## Candidates

1. `Residual-Prior Convex Mixer`
   - Family residual priors are blended inside a bounded convex hull around LRBN.
2. `Time-Step Gated Hybrid Editor`
   - Smoothing and local/boundary edits are selected at time-step level instead of sample level.
3. `Selection-Conditional Conformal Family Editor`
   - Per-family selected subset calibration with bias centering and dead-zone.
4. `Retrieval-Augmented Local Residual Editor`
   - Retrieval/memory only supplies local residual priors and agreement, not hard replacement.
5. `Conservative Challenger Comparator`
   - Family recall proposes challengers; a conservative pairwise rule accepts only clear wins.

## Compact Gates

Shared:

- MSE delta vs LRBN is negative.
- Bootstrap high raw delta is below zero.
- No test threshold leakage.

Primary success signals:

- `Residual-Prior Convex Mixer`: MSE delta <= `-1.8%`, harm <= `0.06`, max config harm <= `0.15`, oracle capture >= `0.08`.
- `Time-Step Gated Hybrid Editor`: q4 boundary delta < 0, non-boundary delta <= `-2.0%`, max config harm <= `0.15`.
- `Selection-Conditional Conformal Family Editor`: expected-vs-observed harm gap <= `5pp`, MSE delta < 0.
- `Retrieval-Augmented Local Residual Editor`: non-boundary and low-gap/high-repair improve, boundary not harmed, oracle capture improves by >= `3pp`.
- `Conservative Challenger Comparator`: accept precision >= `0.60`, oracle capture >= `0.10`, harm no higher than its generator.

If no candidate passes compact gates, stop before mini-extension.

## Outputs

- `experiments/halluguard/results/stage13_cga_decision_geometry/`
- `STAGE13_CGA_DECISION_GEOMETRY_REPORT.md`
- Updated `CANDIDATE_BOARD.md`
- Appended `results_halluguard.tsv`

