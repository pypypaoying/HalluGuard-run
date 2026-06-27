# Stage 7 Safe-TAE Summary

## Setup

- Validation samples: `768`
- Inner train/calib: `536` / `232`
- Test samples: `768`
- Test configs: `8`
- Test threshold leakage: `False`

## Verdict

- Safe pass: `True`
- Balanced pass: `False`
- Promote to TableA: `True`
- H1 pairwise safer than top-1: `True`
- H2 no-change gate protects LRBN: `False`
- H3 residual blend safer than hard: `True`
- H4 MRC consistency helped: `False`
- H5 beats SRA frontier: `True`

## Overall

```
                 variant      mse      mae  mse_delta_pct_vs_lrbn  harm_rate  max_per_config_harm_rate  oracle_gain_fraction  ci95_high_mse_delta
         TAE-oracle-best 4.004776 1.473463             -18.172307   0.000000                  0.000000              1.000000            -0.803251
      sra_basic_ablation 4.722565 1.640118              -3.506063   0.113281                  0.177083              0.192934            -0.110182
            sra_balanced 4.766983 1.645627              -2.598508   0.104167                  0.197917              0.142993            -0.099202
mrc_ridge_residual_blend 4.786189 1.644985              -2.206079   0.411458                  0.500000              0.121398            -0.026203
   SafeTAE-pairwise-hard 4.796918 1.659825              -1.986843   0.018229                  0.083333              0.109334            -0.074111
    SafeTAE-tiered-blend 4.803770 1.661128              -1.846842   0.016927                  0.083333              0.101629            -0.070790
 SafeTAE-mrc-consistency 4.804843 1.660837              -1.824928   0.018229                  0.083333              0.100424            -0.067803
            SafeTAE-safe 4.804843 1.660837              -1.824928   0.018229                  0.083333              0.100424            -0.067803
        SafeTAE-balanced 4.804843 1.660837              -1.824928   0.018229                  0.083333              0.100424            -0.067803
                sra_safe 4.813149 1.660950              -1.655217   0.035156                  0.114583              0.091085            -0.058200
  SafeTAE-pairwise-blend 4.831928 1.668265              -1.271508   0.014323                  0.083333              0.069970            -0.049437
       mrc_ridge_abstain 4.835280 1.668420              -1.203025   0.026042                  0.114583              0.066201            -0.031990
                    LRBN 4.894158 1.682162               0.000000   0.000000                  0.000000              0.000000             0.000000
       TAE-router-stage6 5.329668 1.722548               8.898583   0.305990                       NaN                   NaN                  NaN
       TAE-ranker-stage6 5.913439 1.823557              20.826486   0.394531                       NaN                   NaN                  NaN
```

## Selected Params

```json
{
  "SafeTAE-pairwise-hard": {
    "variant": "SafeTAE-pairwise-hard",
    "tau_leave": 0.3,
    "tau_gain": 0.5,
    "tau_harm": 0.2,
    "risk_beta": 1.0,
    "lambda_safe": 1.0,
    "lambda_balanced": 1.0,
    "lambda_aggressive": 0.0,
    "cos_min": null,
    "hard_replacement": true,
    "allow_aggressive": false
  },
  "SafeTAE-pairwise-blend": {
    "variant": "SafeTAE-pairwise-blend",
    "tau_leave": 0.3,
    "tau_gain": 0.5,
    "tau_harm": 0.2,
    "risk_beta": 1.0,
    "lambda_safe": 0.5,
    "lambda_balanced": 0.5,
    "lambda_aggressive": 0.0,
    "cos_min": null,
    "hard_replacement": false,
    "allow_aggressive": false
  },
  "SafeTAE-tiered-blend": {
    "variant": "SafeTAE-tiered-blend",
    "tau_leave": 0.3,
    "tau_gain": 0.5,
    "tau_harm": 0.2,
    "risk_beta": 1.0,
    "lambda_safe": 1.0,
    "lambda_balanced": 0.75,
    "lambda_aggressive": 0.05,
    "cos_min": null,
    "hard_replacement": false,
    "allow_aggressive": false
  },
  "SafeTAE-mrc-consistency": {
    "variant": "SafeTAE-mrc-consistency",
    "tau_leave": 0.3,
    "tau_gain": 0.5,
    "tau_harm": 0.2,
    "risk_beta": 1.0,
    "lambda_safe": 1.0,
    "lambda_balanced": 0.75,
    "lambda_aggressive": 0.05,
    "cos_min": 0.0,
    "hard_replacement": false,
    "allow_aggressive": false
  },
  "SafeTAE-safe": {
    "variant": "SafeTAE-safe",
    "tau_leave": 0.3,
    "tau_gain": 0.5,
    "tau_harm": 0.2,
    "risk_beta": 1.0,
    "lambda_safe": 1.0,
    "lambda_balanced": 0.75,
    "lambda_aggressive": 0.05,
    "cos_min": 0.0,
    "hard_replacement": false,
    "allow_aggressive": false
  },
  "SafeTAE-balanced": {
    "variant": "SafeTAE-balanced",
    "tau_leave": 0.3,
    "tau_gain": 0.5,
    "tau_harm": 0.2,
    "risk_beta": 1.0,
    "lambda_safe": 1.0,
    "lambda_balanced": 0.75,
    "lambda_aggressive": 0.05,
    "cos_min": 0.0,
    "hard_replacement": false,
    "allow_aggressive": false
  },
  "selection_source": "validation_inner_calib_only",
  "test_threshold_leakage": false,
  "mrc_validation_params": {
    "risk_threshold": 0.35,
    "shrink_cap_params": {
      "shrink": 0.4,
      "cap_mult": 2.0,
      "selection_score": 13.212884412317946
    }
  }
}
```