# Stage 9 Architecture Validation Plan

## Objective

Stop incremental Safe-TAE threshold/lambda tuning and validate the large-architecture hypotheses from `deep-research-report (1).md`. This stage asks whether the next performance ceiling is more likely to be broken by candidate-pool redesign, learned eligibility/routing, multiscale residual distribution correction, trajectory energy scoring, or online spectral calibration.

## Fixed Compact Protocol

- Parent baseline: frozen `HalluGuard-LRBN`.
- Compact scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192`, seed `2026`.
- Splits: validation-only inner train/calib for all fitted policies; test split for final evaluation only.
- Bootstrap: `2000`.
- Existing assets:
  - `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
  - `experiments/halluguard/results/lrbn_sra_bp_stage5`
  - `experiments/halluguard/results/stage6_mechanism`
  - `experiments/halluguard/results/stage7_safe_tae`
  - `experiments/halluguard/results/stage8_safe_tae_pareto`
- Output: `experiments/halluguard/results/stage9_architecture_validation`.

## Prototype Slices

1. Restricted-oracle candidate-pool benchmark.
   - Question: is the ceiling now candidate-pool limited or arbitration-limited?
   - Test: compare current deployable expert oracle to an expanded pool with smoothing teacher, validation-memory residual, robust residual median, and jump-aware boundary candidates.

2. Leave-LRBN + eligibility distillation.
   - Question: can gain/harm and expert suitability be learned more stably than current SafeTAE pairwise heads?
   - Test: global pairwise gain/harm probes over sample fingerprint plus candidate trajectory features.

3. Multiscale residual distribution calibrator.
   - Question: can an MRC-v2 residual distribution model beat shallow MRC and approach SRA/SafeTAE while keeping harm controlled?
   - Test: ridge + validation-memory + robust median residual candidates with validation-only shrink/risk/interval calibration.

4. Energy-style trajectory feasibility scorer.
   - Question: can trajectory plausibility ranking outperform point-error surrogate and constrained smoothing?
   - Test: train a loss-delta/energy regressor on validation inner train, calibrate constrained selection on inner calib, evaluate score-gain Spearman and decision utility on test.

5. Matured-label-only online spectral meta-calibration.
   - Question: is FOMC/FAC-like online calibration stable enough to become a second main line?
   - Test: reuse the chronology-preserving Stage 6 replay with matured labels only.

## Gates

Safe candidate gate:

- MSE delta vs LRBN <= `-2.0%`
- harm <= `0.03`
- max config harm <= `0.10`
- config improved ratio >= `0.75`
- no test threshold leakage

Tradeoff/prototype gate:

- MSE delta vs LRBN <= `-3.0%`
- harm <= `0.08`
- max config harm <= `0.18`
- oracle gain fraction >= `0.15` when available
- no test threshold leakage

Diagnostic directions can be recommended for redesign even if deployable gate fails, but only if the oracle/ranking evidence is strong and the failure mode is specific.
