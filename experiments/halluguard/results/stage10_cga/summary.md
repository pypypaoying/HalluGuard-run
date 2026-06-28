# Stage 10 CGA Compact Validation

Status: `mechanism_pass_selector_still_insufficient`.

## Headline Metrics

| variant | mse | mae | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | config_improved_ratio | coverage | oracle_gain_fraction | test_threshold_leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan | False |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.197917 | 1.000000 | 0.000000 | nan | False |
| oracle_stage10_cga_full | 3.074767 | 1.322182 | -37.174740 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 1.000000 | False |
| Safe-CGA | 4.831768 | 1.664919 | -1.274774 | 0.052083 | 0.260417 | 0.875000 | 0.179688 | 0.034291 | False |
| Balanced-CGA | 4.831768 | 1.664919 | -1.274774 | 0.052083 | 0.260417 | 0.875000 | 0.179688 | 0.034291 | False |

## Selector Top-k

| candidate_top2_hit | family_top2_hit | family_top1_hit | family_minus_candidate_top2_pp |
| --- | --- | --- | --- |
| 0.093750 | 0.619792 | 0.424479 | 52.604167 |

## Verdict

```json
{
  "stage": "stage10_cga",
  "status": "mechanism_pass_selector_still_insufficient",
  "mechanism_pass": true,
  "safe_cga_pass": false,
  "balanced_cga_pass": false,
  "oracle_improvement_pct_vs_old_deployable": -27.097931390800035,
  "new_family_oracle_share": 0.6015624999999999,
  "family_top2_hit": 0.6197916666666666,
  "candidate_top2_hit": 0.09375,
  "family_minus_candidate_top2_pp": 52.604166666666664,
  "new_family_non_boundary_improves": true,
  "safe_cga": {
    "mse_delta_pct_vs_lrbn": -1.2747736369025215,
    "harm_rate": 0.052083333333333336,
    "max_config_harm": 0.2604166666666667,
    "config_improved_ratio": 0.875,
    "oracle_gain_fraction": 0.0342913932000899
  },
  "balanced_cga": {
    "mse_delta_pct_vs_lrbn": -1.2747736369025215,
    "harm_rate": 0.052083333333333336,
    "max_config_harm": 0.2604166666666667,
    "config_improved_ratio": 0.875,
    "oracle_gain_fraction": 0.0342913932000899
  },
  "safe_policy": {
    "variant": "Safe-CGA",
    "tau_leave": 0.55,
    "tau_family_gain": 0.55,
    "tau_family_harm": 0.15,
    "tau_candidate_gain": 0.55,
    "tau_candidate_harm": 0.15,
    "beta_harm": 2.0,
    "lambda_existing": 1.0,
    "lambda_smoothing": 1.0,
    "lambda_residual": 0.75,
    "lambda_memory": 0.75
  },
  "balanced_policy": {
    "variant": "Balanced-CGA",
    "tau_leave": 0.55,
    "tau_family_gain": 0.55,
    "tau_family_harm": 0.15,
    "tau_candidate_gain": 0.55,
    "tau_candidate_harm": 0.15,
    "beta_harm": 2.0,
    "lambda_existing": 1.0,
    "lambda_smoothing": 1.0,
    "lambda_residual": 0.75,
    "lambda_memory": 0.75
  },
  "test_threshold_leakage": false
}
```

## Artifacts

Output directory: `experiments\halluguard\results\stage10_cga`