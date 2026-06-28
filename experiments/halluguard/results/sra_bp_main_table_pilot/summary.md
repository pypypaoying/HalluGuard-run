# SRA-BP Main-Table Pilot Summary

## Scope

- Metrics input: `experiments\halluguard\results\research_direction_validation\forecast_inputs\combined_metrics.csv`
- Stage5 params: `experiments\halluguard\results\lrbn_sra_bp_stage5`
- Test configs: `8`
- Test samples: `768`
- Strict comparison scope: sample-aligned compact pilot.
- Official adapter rows without aligned prediction files are reference-only and are not included in the aligned mean.
- Test threshold leakage: `False`

## Aligned Method Summary

| method | completed_configs | mean_mse | mean_mae | mean_mse_delta_pct_vs_raw | mean_mse_delta_pct_vs_lrbn | improved_configs_vs_lrbn | harmed_configs_vs_lrbn | mean_harm_rate_vs_lrbn | mean_coverage | max_mse_harm_pct_vs_lrbn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN-SRA-BP-balanced | 8 | 4.766983 | 1.645627 | -22.302684 | -2.975935 | 8 | 0 | 0.104167 | 0.436198 | 0.000000 |
| LRBN-SRA-BP-safe | 8 | 4.813149 | 1.660950 | -21.562981 | -1.996233 | 8 | 0 | 0.035156 | 0.187500 | 0.000000 |
| HalluGuard-LRBN | 8 | 4.894158 | 1.682162 | -20.028831 | 0.000000 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 |
| ema_smoothing | 8 | 6.060487 | 1.848221 | -5.922068 | 24.694504 | 4 | 4 | 0.438802 | 0.000000 | 56.972766 |
| naive_smoothing | 8 | 6.072069 | 1.849256 | -5.763867 | 25.002055 | 4 | 4 | 0.430990 | 0.000000 | 58.508535 |
| median_smoothing | 8 | 6.129929 | 1.860337 | -4.849369 | 26.440553 | 4 | 4 | 0.449219 | 0.000000 | 61.845645 |
| matched_sparse_smoothing | 8 | 6.222475 | 1.877730 | -2.971133 | 28.340476 | 3 | 5 | 0.584635 | 0.000000 | 59.379329 |
| raw_no_correction | 8 | 6.427221 | 1.914908 | 0.000000 | 32.849732 | 0 | 8 | 0.641927 | 0.000000 | 69.827690 |

## Official Baseline Availability

| method | same_config_reference_available_for_pilot | same_config_reference_rows | reference_completed_rows | reference_mean_mse | mean_raw_mse_gap_pct_reference_vs_pilot | comparison_status |
| --- | --- | --- | --- | --- | --- | --- |
| RevIN | False | 0 | 0 |  |  | not_available_locally |
| DishTS | True | 12 | 20 | 7.212027 | 47.914700 | same_config_reference_only_raw_baseline_mismatch |
| SAN | True | 12 | 20 | 7.516128 | 47.914700 | same_config_reference_only_raw_baseline_mismatch |
| NST | False | 0 | 0 |  |  | not_available_locally |
| TAFAS | False | 0 | 0 |  |  | not_available_locally |

## Pilot Verdict

- Safe-SRA mean MSE delta vs LRBN: `-1.996233%`; harmed configs: `0`.
- Balanced-SRA mean MSE delta vs LRBN: `-2.975935%`; harmed configs: `0`.
- Both SRA variants beat the aligned raw/LRBN/smoothing-control set in this compact pilot.
- This is enough to enter a real core-table integration as a candidate, but not enough to claim superiority over RevIN/DishTS/SAN/NST/TAFAS until those baselines are run on the same rows and prediction schema.
