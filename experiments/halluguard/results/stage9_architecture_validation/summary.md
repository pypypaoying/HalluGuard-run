# Stage 9 Architecture Validation Summary

## Verdict

- Status: `oracle_space_found_but_deployable_selector_missing`
- Recommended directions: `candidate_pool_redesign`
- Test threshold leakage: `False`

## Prototype Verdicts

```json
{
  "restricted_oracle": {
    "prototype": "restricted_oracle",
    "current_deployable_mse": 4.2176680130401145,
    "expanded_deployable_mse": 3.908519687469408,
    "expanded_extra_delta_pct_vs_current_oracle": -7.3298401837674945,
    "expanded_extra_delta_pct_vs_lrbn": -20.139071682633965,
    "pool_redesign_promising": true,
    "test_threshold_leakage": false
  },
  "pairwise_distillation": {
    "prototype": "pairwise_distillation",
    "global_gain_calib_roc_auc": 0.8780523799124965,
    "global_harm_calib_roc_auc": 0.8416769887352533,
    "stage7_gain_calib_roc_auc": 0.8654654924713209,
    "stage7_harm_calib_roc_auc": 0.8115309894449362,
    "test_top2_oracle_hit": 0.16796875,
    "representation_promising": false,
    "test_threshold_leakage": false
  },
  "mrc_v2_multiscale": {
    "prototype": "mrc_v2_multiscale",
    "safe_delta_pct_vs_lrbn": -0.37370996758226965,
    "tradeoff_delta_pct_vs_lrbn": -0.37370996758226965,
    "safe_harm": 0.10026041666666667,
    "tradeoff_harm": 0.10026041666666667,
    "mrc_v2_safe_pass": false,
    "mrc_v2_tradeoff_pass": false,
    "test_threshold_leakage": false
  },
  "energy_feasibility": {
    "prototype": "energy_feasibility",
    "score_gain_spearman": 0.29276082244860224,
    "mse_delta_pct_vs_lrbn": 4.918739604857358,
    "harm_rate": 0.16666666666666666,
    "max_config_harm": 0.40625,
    "energy_promising": false,
    "test_threshold_leakage": false
  },
  "online_spectral": {
    "prototype": "online_spectral_matured_label",
    "spectral_delta_pct_vs_lrbn": -1.158871438027399,
    "rolling_delta_pct_vs_lrbn": -0.5977366751270848,
    "spectral_minus_rolling_pct": -0.5611347629003142,
    "spectral_harm": 0.4674479166666667,
    "coverage_gap_pp": 6.658664279513882,
    "protocol_guard_pass": true,
    "online_promising": false,
    "test_threshold_leakage": false
  }
}
```

## Deployable / Diagnostic Metrics

| variant | mse | mae | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | config_improved_ratio | coverage | oracle_gain_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_expanded_all | 3.871354 | 1.445849 | -20.898458 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| oracle_expanded_deployable | 3.908520 | 1.453407 | -20.139072 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| TAE-oracle-best | 4.004776 | 1.473463 | -18.172307 | 0.000000 | 0.000000 | nan | 0.000000 | 1.000000 |
| oracle_current_all | 4.004776 | 1.473463 | -18.172307 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| oracle_current_deployable | 4.217668 | 1.523301 | -13.822391 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 |
| sra_balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.197917 | nan | 0.000000 | 0.142993 |
| SafeTAE-safe | 4.804843 | 1.660837 | -1.824928 | 0.018229 | 0.083333 | nan | 0.519531 | 0.100424 |
| spectral_adapter | 4.837441 | 1.680623 | -1.158871 | 0.467448 | nan | nan | 0.000000 | nan |
| rolling_mean_residual | 4.864903 | 1.699420 | -0.597737 | 0.531250 | nan | nan | 0.000000 | nan |
| MRC-v2-multiscale-safe | 4.875868 | 1.678373 | -0.373710 | 0.100260 | 0.302083 | 0.750000 | 0.300781 | nan |
| MRC-v2-multiscale-tradeoff | 4.875868 | 1.678373 | -0.373710 | 0.100260 | 0.302083 | 0.750000 | 0.300781 | nan |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | 0.000000 |
| no_update | 4.894158 | 1.682162 | 0.000000 | 0.000000 | nan | nan | 0.000000 | nan |
| time_ema_residual | 4.901755 | 1.706658 | 0.155240 | 0.541667 | nan | nan | 0.000000 | nan |
| Energy-feasibility-reranker | 5.134888 | 1.705022 | 4.918740 | 0.166667 | 0.406250 | 0.500000 | 0.333333 | nan |

## Interpretation

The expanded candidate pool opens additional oracle space, but current deployable selectors do not safely capture enough of it. The next promising direction is a redesigned hierarchical arbitrator with a stronger candidate pool, not another threshold sweep.
