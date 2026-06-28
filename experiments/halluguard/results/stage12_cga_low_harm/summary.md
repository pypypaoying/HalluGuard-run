# Stage 12 CGA Low-Harm Priority Validation

Status: `compact_failed_stop_before_mini_extension`.

## Overall

| variant | mse | mae | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | coverage | oracle_gain_fraction | ci95_high_delta_raw | test_threshold_leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan | 0.000000 | False |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.197917 | 0.000000 | 0.069900 | -0.099202 | False |
| oracle_stage12_cga_full | 3.074767 | 1.322182 | -37.174740 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | -1.564427 | False |
| Sparse-Family-CGA | 4.823736 | 1.665719 | -1.438899 | 0.251302 | 0.489583 | 0.998698 | 0.038706 | -0.037877 | False |
| Sparse-Residual-Simplex-CGA | 4.849834 | 1.671598 | -0.905652 | 0.226562 | 0.479167 | 0.998698 | 0.024362 | -0.025583 | False |
| NoHarm-Selective-CGA | 4.848960 | 1.671547 | -0.923492 | 0.235677 | 0.479167 | 0.998698 | 0.024842 | -0.026707 | False |
| LambdaVeto-CGA | 4.845183 | 1.670993 | -1.000674 | 0.208333 | 0.458333 | 0.990885 | 0.026918 | -0.031733 | False |

## Verdict

```json
{
  "stage": "stage12_cga_low_harm",
  "status": "compact_failed_stop_before_mini_extension",
  "compact_pass": false,
  "best_variant": "Sparse-Family-CGA",
  "best_mse": 4.823735595698971,
  "best_mae": 1.665719496129702,
  "best_mse_delta_pct_vs_lrbn": -1.4388993157512555,
  "best_harm_rate": 0.2513020833333333,
  "best_max_config_harm": 0.4895833333333333,
  "best_oracle_gain_fraction": 0.0387063716909454,
  "lambda_veto_mse_delta_pct_vs_lrbn": -1.0006744331409079,
  "lambda_veto_harm_rate": 0.20833333333333334,
  "lambda_veto_max_config_harm": 0.4583333333333333,
  "lambda_veto_oracle_gain_fraction": 0.026918128410225856,
  "known_harmed_config_delta_pct": -0.05985874528772339,
  "boundary_like_worst_slice_delta_pct": -0.6178030441697414,
  "residual_simplex_extra_delta_pp_vs_sparse": 0.5332474943500435,
  "family_top2_hit": 0.6197916666666666,
  "candidate_top2_hit": 0.09375,
  "gates": {
    "sparse_family_pass": false,
    "residual_simplex_pass": false,
    "noharm_selective_pass": false,
    "lambda_veto_pass": false,
    "oracle_gain_fraction_min": 0.08,
    "max_config_harm_max": 0.18,
    "final_max_config_harm_target": 0.12
  },
  "stop_reason": "compact gate failed: sparse_family_pass, residual_simplex_pass, noharm_selective_pass, lambda_veto_pass",
  "test_threshold_leakage": false
}
```

Output directory: `experiments\halluguard\results\stage12_cga_low_harm`