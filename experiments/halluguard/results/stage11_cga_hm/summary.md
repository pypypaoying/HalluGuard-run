# Stage 11 CGA-HM Mechanism Validation

Status: `stage1_failed_stop_before_mini_extension`.

## Overall

| variant | mse | mae | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | coverage | oracle_gain_fraction | mean_safe_weight | test_threshold_leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan | nan | False |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.197917 | 0.000000 | nan | nan | False |
| oracle_stage11_cga_full | 3.074767 | 1.322182 | -37.174740 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | nan | False |
| CGA-HM-safe | 4.841243 | 1.670342 | -1.081187 | 0.204427 | 0.479167 | 0.998698 | 0.029084 | 0.800260 | False |
| CGA-HM-balanced | 4.841243 | 1.670342 | -1.081187 | 0.204427 | 0.479167 | 0.998698 | 0.029084 | 0.800260 | False |
| CGA-HM-veto-safe | 4.839807 | 1.670458 | -1.110515 | 0.199219 | 0.468750 | 0.990885 | 0.029873 | 0.801823 | False |
| CGA-HM-veto-balanced | 4.839807 | 1.670458 | -1.110515 | 0.199219 | 0.468750 | 0.990885 | 0.029873 | 0.801823 | False |

## Verdict

```json
{
  "stage": "stage11_cga_hm",
  "status": "stage1_failed_stop_before_mini_extension",
  "stage1_pass": false,
  "best_variant": "CGA-HM-veto-safe",
  "best_mse": 4.839807225691508,
  "best_mae": 1.6704583084943356,
  "best_mse_delta_pct_vs_lrbn": -1.110515326383568,
  "best_harm_rate": 0.19921875,
  "best_max_config_harm": 0.46875,
  "best_oracle_gain_fraction": 0.029872846919140977,
  "oracle_full_mse": 3.074767225864992,
  "stage1_gate": {
    "oracle_gain_fraction_min": 0.08,
    "max_config_harm_max": 0.18,
    "mse_delta_negative": true
  },
  "stop_reason": "best deployable policy failed gate: oracle_gain_fraction=0.029873, max_config_harm=0.468750, mse_delta_pct_vs_lrbn=-1.110515",
  "policies": {
    "CGA-HM-safe": {
      "variant": "CGA-HM-safe",
      "family_rep": "median",
      "tau_leave": 0.55,
      "tau_family_gain": 0.5,
      "tau_family_harm": 0.06,
      "safe_floor": 0.8,
      "beta_harm": 2.0,
      "temperature": 1.0,
      "family_mass_cap": 0.25,
      "boundary_veto": false,
      "boundary_veto_mult": 1.0,
      "boundary_gap_quantile": 0.75,
      "boundary_repair_quantile": 0.5
    },
    "CGA-HM-balanced": {
      "variant": "CGA-HM-balanced",
      "family_rep": "median",
      "tau_leave": 0.55,
      "tau_family_gain": 0.5,
      "tau_family_harm": 0.06,
      "safe_floor": 0.8,
      "beta_harm": 2.0,
      "temperature": 1.0,
      "family_mass_cap": 0.25,
      "boundary_veto": false,
      "boundary_veto_mult": 1.0,
      "boundary_gap_quantile": 0.75,
      "boundary_repair_quantile": 0.5
    },
    "CGA-HM-veto-safe": {
      "variant": "CGA-HM-veto-safe",
      "family_rep": "median",
      "tau_leave": 0.55,
      "tau_family_gain": 0.5,
      "tau_family_harm": 0.06,
      "safe_floor": 0.8,
      "beta_harm": 2.0,
      "temperature": 1.0,
      "family_mass_cap": 0.25,
      "boundary_veto": true,
      "boundary_veto_mult": 0.0,
      "boundary_gap_quantile": 0.75,
      "boundary_repair_quantile": 0.5
    },
    "CGA-HM-veto-balanced": {
      "variant": "CGA-HM-veto-balanced",
      "family_rep": "median",
      "tau_leave": 0.55,
      "tau_family_gain": 0.5,
      "tau_family_harm": 0.06,
      "safe_floor": 0.8,
      "beta_harm": 2.0,
      "temperature": 1.0,
      "family_mass_cap": 0.25,
      "boundary_veto": true,
      "boundary_veto_mult": 0.0,
      "boundary_gap_quantile": 0.75,
      "boundary_repair_quantile": 0.5
    }
  },
  "test_threshold_leakage": false
}
```

Output directory: `experiments\halluguard\results\stage11_cga_hm`