# Stage19 Residual Quantile / Non-Boundary Shape Atom Validation

Status: `compact_failed_stop_performance_atom_route`.

## Compact Variant Metrics

| variant | family | mse | mae | mse_delta_pct_vs_sra | harm_rate_vs_sra | max_config_harm | coverage | safe_gate | tradeoff_gate | test_threshold_leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SRA-BP-balanced | reference | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | False | False | False |
| RQA-PCA-Coef | residual_distribution | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | False | False | False |
| RQA-DCT-Coef | residual_distribution | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | False | False | False |
| RQA-HarmAwareCoef | residual_distribution | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | False | False | False |
| NBSA-DCT-Shape | smoothing_teacher | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | False | False | False |
| NBSA-RoughnessAdapter | smoothing_teacher | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | False | False | False |
| NBSA-NonBoundaryOnly | smoothing_teacher | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 0.747396 | False | False | False |
| NBSA-LocalShapeEnvelope | smoothing_teacher | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | False | False | False |
| RQA+NBSA | combined | 4.766983 | 1.645627 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | False | False | False |
| RQA-QuantileHead | residual_distribution | 4.782471 | 1.647065 | 0.324914 | 0.149740 | 0.281250 | 0.313802 | False | False | False |
| SRA-BP-safe | reference | 4.813149 | 1.660950 | 0.968457 | 0.187500 | 0.270833 | 1.000000 | False | False | False |
| LRBN | reference | 4.894158 | 1.682162 | 2.667832 | 0.332031 | 0.500000 | 1.000000 | False | False | False |

## Basis Report

| variant | family | basis_type | component | explained_variance_ratio | cumulative_evr |
| --- | --- | --- | --- | --- | --- |
| RQA-PCA-Coef | residual_distribution | pca | 1 | 0.686241 | 0.686241 |
| RQA-PCA-Coef | residual_distribution | pca | 2 | 0.213571 | 0.899811 |
| RQA-PCA-Coef | residual_distribution | pca | 3 | 0.076165 | 0.975976 |
| RQA-PCA-Coef | residual_distribution | pca | 4 | 0.011342 | 0.987319 |
| RQA-PCA-Coef | residual_distribution | pca | 5 | 0.002723 | 0.990042 |
| RQA-DCT-Coef | residual_distribution | dct | 1 | nan | nan |
| RQA-DCT-Coef | residual_distribution | dct | 2 | nan | nan |
| RQA-DCT-Coef | residual_distribution | dct | 3 | nan | nan |
| RQA-DCT-Coef | residual_distribution | dct | 4 | nan | nan |
| RQA-DCT-Coef | residual_distribution | dct | 5 | nan | nan |
| RQA-DCT-Coef | residual_distribution | dct | 6 | nan | nan |
| RQA-DCT-Coef | residual_distribution | dct | 7 | nan | nan |
| RQA-DCT-Coef | residual_distribution | dct | 8 | nan | nan |
| RQA-HarmAwareCoef | residual_distribution | pca | 1 | 0.686241 | 0.686241 |
| RQA-HarmAwareCoef | residual_distribution | pca | 2 | 0.213571 | 0.899811 |
| RQA-HarmAwareCoef | residual_distribution | pca | 3 | 0.076165 | 0.975976 |
| RQA-HarmAwareCoef | residual_distribution | pca | 4 | 0.011342 | 0.987319 |
| RQA-HarmAwareCoef | residual_distribution | pca | 5 | 0.002723 | 0.990042 |
| NBSA-DCT-Shape | smoothing_teacher | dct | 1 | nan | nan |
| NBSA-DCT-Shape | smoothing_teacher | dct | 2 | nan | nan |
| NBSA-DCT-Shape | smoothing_teacher | dct | 3 | nan | nan |
| NBSA-DCT-Shape | smoothing_teacher | dct | 4 | nan | nan |
| NBSA-DCT-Shape | smoothing_teacher | dct | 5 | nan | nan |
| NBSA-DCT-Shape | smoothing_teacher | dct | 6 | nan | nan |

## Coefficient Fit

| variant | family | basis_type | coefficient_r2 | coefficient_sign_accuracy | policy_shrink | policy_cap_ratio | policy_coverage | policy_score_tau | policy_segment | policy_calibration_score | policy_rqa_shrink | policy_nbsa_shrink | policy_best_rqa | policy_best_nbsa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQA-PCA-Coef | residual_distribution | pca | -0.005419 | 0.564253 | 0.000000 | 0.030000 | 1.000000 | -inf | all | 0.000000 | nan | nan | nan | nan |
| RQA-DCT-Coef | residual_distribution | dct | -0.008819 | 0.652670 | 0.000000 | 0.030000 | 1.000000 | -inf | all | 0.000000 | nan | nan | nan | nan |
| RQA-HarmAwareCoef | residual_distribution | pca | -0.005419 | 0.564253 | 0.000000 | 0.030000 | 1.000000 | -inf | all | 0.000000 | nan | nan | nan | nan |
| NBSA-DCT-Shape | smoothing_teacher | dct | 0.202890 | 0.637946 | 0.000000 | 0.030000 | 1.000000 | -inf | all | 0.000000 | nan | nan | nan | nan |
| NBSA-RoughnessAdapter | smoothing_teacher | dct | 0.202890 | 0.637946 | 0.000000 | 0.030000 | 1.000000 | -inf | all | 0.000000 | nan | nan | nan | nan |
| NBSA-NonBoundaryOnly | smoothing_teacher | dct | 0.202890 | 0.637946 | 0.000000 | 0.030000 | 1.000000 | -inf | all | 0.000000 | nan | nan | nan | nan |
| NBSA-LocalShapeEnvelope | smoothing_teacher | dct | 0.202890 | 0.637946 | 0.000000 | 0.030000 | 1.000000 | -inf | early | 0.000000 | nan | nan | nan | nan |
| RQA-QuantileHead | residual_distribution | quantile_head | nan | nan | 1.000000 | 0.350000 | 0.200000 | 3.329664 | all | -0.551496 | nan | nan | nan | nan |
| RQA+NBSA | combined | composed | nan | nan | 1.000000 | 0.030000 | 1.000000 | -inf | all | 0.000000 | 0.000000 | 0.000000 | RQA-QuantileHead | NBSA-DCT-Shape |

## Complementarity

| best_rqa_variant | best_nbsa_variant | rqa_gain_rate | nbsa_gain_rate | overlap_gain_rate | rqa_only_gain_rate | nbsa_only_gain_rate | combo_gain_rate | combo_gain_over_best_single_pp | test_threshold_leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQA-PCA-Coef | NBSA-DCT-Shape | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | False |

## Verdict

```json
{
  "stage": "stage19_performance_atom_validation",
  "status": "compact_failed_stop_performance_atom_route",
  "compact_protocol_completed": true,
  "mini_extension_ran": false,
  "reason_no_mini_extension": "Compact gate did not pass.",
  "rqa_pass": false,
  "nbsa_pass": false,
  "combo_pass": false,
  "best_variant": {
    "variant": "RQA-PCA-Coef",
    "family": "residual_distribution",
    "status": "completed",
    "n": 768,
    "mse": 4.766982513125085,
    "mae": 1.6456270785485227,
    "parent": "SRA-BP-balanced",
    "parent_mse": 4.766982513125085,
    "parent_mae": 1.6456270785485227,
    "mse_delta_vs_sra": 0.0,
    "mse_delta_pct_vs_sra": 0.0,
    "mae_delta_pct_vs_sra": 0.0,
    "harm_rate_vs_sra": 0.0,
    "win_rate_vs_sra": 0.0,
    "max_config_harm": 0.0,
    "improved_configs": 0,
    "total_configs": 8,
    "coverage": 1.0,
    "selected_count": 768,
    "selected_harm_rate": 0.0,
    "test_threshold_leakage": false,
    "ci95_low_delta_raw": 0.0,
    "ci95_high_delta_raw": 0.0,
    "p_bootstrap_delta_lt_zero": 0.0,
    "A_gt1_rate": 0.0,
    "mean_A": 0.0,
    "mean_cosine_with_residual": 0.0,
    "delta_norm_ratio": 0.0,
    "safe_gate": false,
    "tradeoff_gate": false,
    "boundary_delta": 0.0,
    "non_boundary_delta": 0.0
  },
  "best_rqa_variant": "RQA-PCA-Coef",
  "best_nbsa_variant": "NBSA-DCT-Shape",
  "best_single_delta_pct_vs_sra": 0.0,
  "combo_delta_pct_vs_sra": 0.0,
  "test_threshold_leakage": false,
  "recommendation": "Do not promote Stage19 adapters; current continuous atom application does not beat SRA-BP-balanced safely.",
  "runtime_seconds": 178.69553184509277
}
```

Output directory: `experiments\halluguard\results\stage19_performance_atom_validation`