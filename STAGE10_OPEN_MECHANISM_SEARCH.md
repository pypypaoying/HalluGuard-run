# Stage 10 Open Mechanism Search

## Status

Stage 10 completed open mechanism search beyond the old HalluGuard trend/frequency trigger.

Outputs:

- `experiments/halluguard/results/stage10_open_mechanism_search/candidate_ledger.csv`
- `experiments/halluguard/results/stage10_open_mechanism_search/smoke/`
- `experiments/halluguard/results/stage10_open_mechanism_search/clean_full_table/`
- `experiments/halluguard/results/stage10_open_mechanism_search/stress_table/`
- `experiments/halluguard/results/stage10_open_mechanism_search/diagnostics/`

All Stage 10 policy selection used `val` only. All final metrics below are from `test`. `test_threshold_leakage=False` for every completed run.

## Candidates Tested

| Candidate | Family | Scope Reached | Mean MSE Delta | Rule Beats Random | Paired Win | Beats Matched | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `s10_c1_error_predictive_residual` | error-predictive diagnostic trigger | smoke | `-2.435762%` | `3 / 4` | `0.800` | `3 / 4` | archive: max MSE harm `3.115223%`, diagnostic score weak |
| `s10_c2_dynamics_consistency` | dynamics-consistency trigger | clean + stress | `-0.664967%` clean | `15 / 16` clean | `0.925` clean | `4 / 16` clean | partial: strong rule/random, failed matched smoothing |
| `s10_c3_perturbation_stability` | perturbation stability trigger | smoke | `-0.081699%` | `3 / 4` | `0.700` | `0 / 4` | archive: matched smoothing explains signal |
| `s10_c4_residual_shape_model` | residual-shape correction | smoke | catastrophic harm | `1 / 4` | `0.250` | `1 / 4` | archive: unstable residual model |
| `s10_c2b_dynamics_anti_smoothing` | dynamics + anti-smoothing objective | clean + stress | `-0.623252%` clean | `15 / 16` clean | `0.9375` clean | `12 / 16` clean | best: clean mechanism pass |

## Best Candidate

`s10_c2b_dynamics_anti_smoothing` is the best Stage 10 candidate.

Mechanism:

- Trigger: local dynamics inconsistency score from context/prediction boundary jump, first-difference mismatch, and curvature mismatch.
- Correction: decaying boundary/derivative continuity adjustment. It is not FFT attenuation and not moving-average smoothing.
- Calibration: validation-only search over trigger quantile and correction strength.
- Exploit change versus `s10_c2`: validation objective includes an anti-smoothing term, so the selected policy must compete against matched smoothing on validation.

## Clean Full Result

Clean full table:

- Path: `experiments/halluguard/results/stage10_open_mechanism_search/clean_full_table/s10_c2b_dynamics_anti_smoothing/`
- Completed configs: `16 / 16`
- Variant rows: `112`
- `test_threshold_leakage=False`

Key clean metrics for `candidate_main`:

- mean MSE: `6.9345446817`
- mean MAE: `1.9671979918`
- mean MSE delta vs `no_correction`: `-0.6232515127%`
- improved configs: `15 / 16`
- beats `random_trigger`: `15 / 16`
- paired rule-vs-random win rate: `0.9375`
- beats `matched_smoothing_control`: `12 / 16`
- max MSE harm: `0.0675740991%`
- max MAE harm: `0.0403744224%`
- mean correction rate: `0.3287353516`

Controls:

- `stage9_incumbent` mean MSE delta: `-0.0580377963%`
- `matched_smoothing_control` mean MSE delta: `-0.5918539722%`
- `random_trigger` mean MSE delta: `-0.3650315909%`
- `naive_smoothing` mean MSE delta: `-1.3898261116%`

Clean gate verdict: passed.

## Stress Result

Stress-only table:

- Path: `experiments/halluguard/results/stage10_open_mechanism_search/stress_table/s10_c2b_dynamics_anti_smoothing/`
- Completed stress configs: `32 / 32`
- Stress types: `trend_drift`, `high_frequency_perturbation`
- `test_threshold_leakage=False`

Overall stress metrics for `candidate_main`:

- mean MSE delta: `-0.6224247593%`
- improved stress configs: `31 / 32`
- beats `random_trigger`: `32 / 32`
- paired rule-vs-random win rate: `0.95625`
- beats `matched_smoothing_control`: `15 / 32`
- max MSE harm: `0.0061982166%`

By stress type:

| Stress Type | Candidate Mean MSE Delta | Random Mean MSE Delta | Matched Smoothing Mean MSE Delta | Beats Random | Beats Matched |
| --- | ---: | ---: | ---: | ---: | ---: |
| `trend_drift` | `-0.716644%` | `-0.420488%` | `-0.626357%` | `16 / 16` | `11 / 16` |
| `high_frequency_perturbation` | `-0.528205%` | `-0.269893%` | `-0.607659%` | `16 / 16` | `4 / 16` |

Stress read:

- The dynamics mechanism strongly separates from random on both stress types.
- The clean-passing mechanism is best justified as a dynamics-continuity correction, not as a frequency-specific correction.
- High-frequency stress still has a smoothing explanation: matched smoothing is stronger on average there.
- Trend-drift stress is more aligned with the mechanism: candidate beats random in `16 / 16`, beats matched in `11 / 16`, and has better mean MSE delta than matched smoothing.

## Diagnostic-Only Read

No Stage 10 diagnostic-only route passed.

Best diagnostic metrics stayed weak:

- clean mean Spearman for `s10_c2b`: `0.067396`
- clean mean AUROC top-20% error: `0.542799`
- clean top10/bottom50 error lift: `1.261541`

This is below the diagnostic-only success bar of top-decile error lift `>= 2x`.

## Conclusion

Stage 10 found a new correction mechanism with clean ETT evidence:

- It is not the old trend/frequency HalluGuard trigger.
- It is not explained by matched random triggering.
- It beats matched smoothing on the clean full table in `12 / 16` configs.
- It remains validation-only and safe under the 3% harm rule.

The claim should be narrowed to `HalluGuard-Dynamics`: a local dynamics-continuity test-time correction. It should not be presented as a frequency-repair method, and naive smoothing remains a stronger pure point-MSE baseline.

