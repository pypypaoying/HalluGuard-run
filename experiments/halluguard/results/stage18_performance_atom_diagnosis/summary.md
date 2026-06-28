# Stage18 Performance Atom Extraction Diagnosis

Status: `atom_route_mechanism_pass_distillation_not_ready`.

## Oracle over SRA-BP-balanced

| oracle_pool | incremental_oracle_delta_pct | oracle_gain_fraction_vs_LRBN_oracle | non_parent_selection_rate | ci95_high_delta_raw |
| --- | --- | --- | --- | --- |
| tae_old_pool | -11.523317 | 0.301922 | 0.899740 | -0.499537 |
| cga_new_family_pool | -20.385746 | 0.534127 | 0.914062 | -0.880683 |
| union_full_pool | -35.498668 | 0.930100 | 0.959635 | -1.441710 |
| sra_complement_atom_pool | -21.280744 | 0.557577 | 0.934896 | -0.923250 |

## Family Leave-One-Out

| family_group | removed_families | full_oracle_mse | leave_one_out_mse | leave_one_out_degradation_pct_vs_full | leave_one_out_degradation_pct_vs_parent | family_oracle_share | candidate_count | test_threshold_leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| residual_distribution | residual_distribution | 3.752533 | 4.044054 | 7.768633 | -15.165334 | 0.450521 | 4 | False |
| smoothing_teacher | smoothing_teacher | 3.752533 | 3.960635 | 5.545641 | -16.915257 | 0.201823 | 5 | False |
| old_residual | residual | 3.752533 | 3.791452 | 1.037140 | -20.464315 | 0.167969 | 2 | False |
| retrieval_memory | retrieval_memory | 3.752533 | 3.760161 | 0.203281 | -21.120723 | 0.053385 | 2 | False |
| volatility_amplitude_level | amplitude,level,volatility | 3.752533 | 3.756126 | 0.095745 | -21.205374 | 0.059896 | 4 | False |
| ensemble | ensemble | 3.752533 | 3.752533 | 0.000002 | -21.280743 | 0.001302 | 1 | False |

## PCA

| component | explained_variance_ratio | cumulative_evr |
| --- | --- | --- |
| 1.000000 | 0.521031 | 0.521031 |
| 2.000000 | 0.168821 | 0.689852 |
| 3.000000 | 0.075156 | 0.765008 |
| 4.000000 | 0.023592 | 0.788601 |
| 5.000000 | 0.017511 | 0.806112 |
| 6.000000 | 0.015508 | 0.821620 |
| 7.000000 | 0.011287 | 0.832907 |
| 8.000000 | 0.010338 | 0.843245 |

## Atom Alignment

| atom_id | n | coverage_within_test_oracle_selected | mean_delta_mse_vs_parent | harm_rate_if_oracle_supported | A_gt1_rate | mean_A | mean_cosine_with_parent_residual | mean_win_size | mean_loss_size | early_delta_mse | mid_delta_mse | late_delta_mse | top_family_group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 220 | 0.306407 | -0.747266 | 0.000000 | 1.000000 | 4.654235 | 0.588547 | 0.747266 | 0.000000 | -0.232542 | -0.846985 | -1.162270 | old_residual |
| 1 | 165 | 0.229805 | -1.260361 | 0.000000 | 1.000000 | 8.504311 | 0.610078 | 1.260361 | 0.000000 | -0.824977 | -1.204640 | -1.751467 | residual_distribution |
| 3 | 92 | 0.128134 | -2.758750 | 0.000000 | 1.000000 | 2.980947 | 0.759319 | 2.758750 | 0.000000 | -1.075088 | -3.264158 | -3.937004 | smoothing_teacher |
| 4 | 241 | 0.335655 | -0.634581 | 0.000000 | 1.000000 | 8.740615 | 0.492246 | 0.634581 | 0.000000 | -0.200798 | -0.827945 | -0.875000 | residual_distribution |

## Distillability

| atom_id | status | chosen_model | activation_train_rate | activation_calib_rate | activation_test_rate | activation_auroc_calib | activation_auroc_test | activation_pr_auc_test | coefficient_sign_accuracy_test | coefficient_r2_test | test_threshold_leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | completed | random_forest | 0.305970 | 0.293103 | 0.286458 | 0.729017 | 0.565411 | 0.327318 | 0.417969 | -0.211325 | False |
| 1 | completed | random_forest | 0.201493 | 0.202586 | 0.214844 | 0.822082 | 0.699040 | 0.350102 | 0.740885 | 0.078126 | False |
| 2 | completed | logistic | 0.001866 | 0.004310 | 0.000000 | 1.000000 | nan | nan | 0.994792 | nan | False |
| 3 | completed | logistic | 0.033582 | 0.030172 | 0.119792 | 0.961905 | 0.840574 | 0.444039 | 0.666667 | 0.155127 | False |
| 4 | completed | random_forest | 0.410448 | 0.435345 | 0.313802 | 0.748394 | 0.715299 | 0.565637 | 0.742188 | 0.035587 | False |

## Prototype Diagnostics

| variant | parent | n | mse | mae | parent_mse | parent_mae | mse_delta_vs_parent | mse_delta_pct_vs_parent | mae_delta_pct_vs_parent | harm_rate_vs_parent | win_rate_vs_parent | max_config_harm | improved_configs | total_configs | test_threshold_leakage | coverage | selected_count | selected_harm_rate | ci95_low_delta_raw | ci95_high_delta_raw | p_bootstrap_delta_lt_zero | atom_id | tau | threshold_quantile | shrink | calib_score | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atom_1_prototype | SRA-BP-balanced | 768 | 4.766983 | 1.645627 | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 8 | False | 0.000000 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1 | 0.588666 | 0.900000 | 1.000000 | -0.942297 | diagnostic_only |
| atom_4_prototype | SRA-BP-balanced | 768 | 4.766983 | 1.645627 | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 8 | False | 0.000000 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 4 | 0.699954 | 0.950000 | 1.000000 | -0.285685 | diagnostic_only |
| atom_0_prototype | SRA-BP-balanced | 768 | 4.788220 | 1.648496 | 4.766983 | 1.645627 | 0.021238 | 0.445520 | 0.174336 | 0.040365 | 0.026042 | 0.125000 | 1 | 8 | False | 0.066406 | 51 | 0.607843 | 0.000433 | 0.042585 | 0.023000 | 0 | 0.743027 | 0.950000 | 1.000000 | -0.988297 | diagnostic_only |
| atom_2_prototype | SRA-BP-balanced | 768 | 4.798364 | 1.656524 | 4.766983 | 1.645627 | 0.031381 | 0.658307 | 0.662188 | 0.048177 | 0.026042 | 0.166667 | 0 | 8 | False | 0.074219 | 57 | 0.649123 | 0.006002 | 0.056298 | 0.007500 | 2 | 0.026710 | 0.950000 | 0.250000 | 4.053167 | diagnostic_only |
| atom_3_prototype | SRA-BP-balanced | 768 | 4.806500 | 1.647341 | 4.766983 | 1.645627 | 0.039517 | 0.828973 | 0.104139 | 0.151042 | 0.145833 | 0.531250 | 2 | 8 | False | 0.296875 | 228 | 0.508772 | 0.006648 | 0.071503 | 0.008000 | 3 | 0.883721 | 0.950000 | 0.250000 | 7.362457 | diagnostic_only |

## Verdict

```json
{
  "stage": "stage18_performance_atom_diagnosis",
  "status": "atom_route_mechanism_pass_distillation_not_ready",
  "atom_route_pass": true,
  "prototype_pass": false,
  "union_full_oracle_gain_pct_vs_sra_balanced": -35.49866774318966,
  "sra_complement_atom_pool_oracle_gain_pct_vs_sra_balanced": -21.280744097566625,
  "max_leave_one_out_degradation_pct_vs_full": 7.768633285887863,
  "top5_atom_explained_variance_ratio": 0.8061119297448998,
  "max_atom_A_gt1_rate": 1.0,
  "best_prototype": {
    "variant": "atom_1_prototype",
    "parent": "SRA-BP-balanced",
    "n": 768,
    "mse": 4.766982513125085,
    "mae": 1.6456270785485227,
    "parent_mse": 4.766982513125085,
    "parent_mae": 1.6456270785485227,
    "mse_delta_vs_parent": 0.0,
    "mse_delta_pct_vs_parent": 0.0,
    "mae_delta_pct_vs_parent": 0.0,
    "harm_rate_vs_parent": 0.0,
    "win_rate_vs_parent": 0.0,
    "max_config_harm": 0.0,
    "improved_configs": 0,
    "total_configs": 8,
    "test_threshold_leakage": false,
    "coverage": 0.0,
    "selected_count": 0,
    "selected_harm_rate": 0.0,
    "ci95_low_delta_raw": 0.0,
    "ci95_high_delta_raw": 0.0,
    "p_bootstrap_delta_lt_zero": 0.0,
    "atom_id": 1,
    "tau": 0.5886655296366551,
    "threshold_quantile": 0.9,
    "shrink": 1.0,
    "calib_score": -0.9422969877940585,
    "status": "diagnostic_only"
  },
  "recommendation": "Keep atom extraction as a promising mechanism diagnosis, but do not deploy until distillation/prototype safety improves.",
  "test_threshold_leakage": false,
  "runtime_seconds": 82.23726987838745
}
```

Output directory: `experiments\halluguard\results\stage18_performance_atom_diagnosis`