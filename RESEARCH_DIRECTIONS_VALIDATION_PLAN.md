# HalluGuard Research Directions Validation Plan

## Goal

Validate the innovation directions from `HalluGuard_research_directions_validation_plan.docx` with a fast, falsifiable sample-level analysis suite before investing in larger implementations.

## Input Contract

- Forecast source: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
- Configs: `ETTm1, ETTh1 x DLinear, PatchTST x horizons 96, 192 x seed 2026`
- Actions: `raw_no_correction`, `HalluGuard-LRBN`, `matched_sparse_smoothing`, `naive_smoothing`, `ema_smoothing`, `median_smoothing`
- Splits: validation trains/calibrates lightweight diagnostics; test evaluates.
- No test threshold tuning.

## Validation Slices

1. E0 unified forecast table
2. E1 residual alignment analysis
3. E2 oracle action separability
4. E3 no-harm predictability and risk-coverage
5. E4 residual basis decomposition
6. E5 dynamic consistency projection prototype
7. E6 multiscale support analysis and edit prototype
8. E7 trajectory critic separability
9. E8 TSFM disagreement analysis, marked blocked unless foundation forecasts exist
10. E9 regime stability

## Outputs

All outputs are written under:

`experiments/halluguard/results/research_direction_validation/`

Key files:

- `sample_features.csv`
- `action_alignment.csv`
- `alignment_summary.csv`
- `oracle_action_separability.csv`
- `risk_coverage.csv`
- `basis_summary.csv`
- `projection_summary.csv`
- `multiscale_summary.csv`
- `critic_summary.csv`
- `regime_summary.csv`
- `direction_verdicts.csv`
- `summary.md`

## Success Logic

This is a pilot validation suite, not a paper-grade final table. A direction is marked `promising` only if the relevant signal is visible on held-out test samples and beats a trivial or shuffled control. Otherwise it is marked `weak`, `inconclusive`, or `blocked`.
