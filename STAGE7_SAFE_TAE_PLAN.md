# Stage 7 Safe-TAE Plan

## Objective

Validate Safe-TAE, a risk-aware trajectory arbitration layer over the frozen HalluGuard-LRBN default. The method uses validation-only pairwise gain/harm heads, a no-change gate, risk-tiered residual blending, and optional MRC residual-prior consistency.

## Compact Protocol

- Input forecast table: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
- Stage 5 dependency: `experiments/halluguard/results/lrbn_sra_bp_stage5`
- Stage 6 dependency: `experiments/halluguard/results/stage6_mechanism`
- Output directory: `experiments/halluguard/results/stage7_safe_tae`
- Scope: ETTm1/ETTh1, DLinear/PatchTST, horizons 96/192, seed 2026
- Splits: `val` calibrates heads/thresholds; `test` evaluates final variants only
- Bootstrap: 2000 paired resamples by default

## Variants

- `TAE-oracle-best`: diagnostic upper bound, not deployable
- `TAE-router-stage6`: Stage 6 reference
- `TAE-ranker-stage6`: Stage 6 reference
- `SafeTAE-pairwise-hard`
- `SafeTAE-pairwise-blend`
- `SafeTAE-tiered-blend`
- `SafeTAE-mrc-consistency`
- `SafeTAE-safe`
- `SafeTAE-balanced`

## Safety Gates

Safe candidate passes only if it improves LRBN, keeps sample harm and per-config harm low, preserves SRA-safe coverage, and has a bootstrap MSE delta CI upper bound below zero.

Balanced candidate passes only if it improves LRBN more strongly than the safe variant, keeps harm bounded, captures meaningful oracle gain, and is Pareto-competitive with SRA-balanced.

## Deliverables

- Reusable Safe-TAE implementation
- CLI runner
- Required CSV/JSON/Parquet outputs
- Stage report with H1-H5 verdicts and TableA recommendation
