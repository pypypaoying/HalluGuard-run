# Stage 8 Safe-TAE Pareto Summary

## Verdict

- Status: `conservative_limit_confirmed`
- Safe pass: `False`
- Balanced pass: `False`
- Test threshold leakage: `False`

## Overall

```
                             variant      mse      mae  mse_delta_pct_vs_lrbn  harm_rate  max_config_harm  coverage  oracle_gain_fraction  ci95_high_mse_delta
                     TAE-oracle-best 4.004776 1.473463             -18.172307   0.000000         0.000000  0.000000              1.000000            -0.803251
                     SRA-BP-balanced 4.766983 1.645627              -2.598508   0.104167         0.197917  0.000000              0.142993            -0.099202
SafeTAE-pareto-aggressive-diagnostic 4.791336 1.656973              -2.100914   0.070312         0.302083  0.759115              0.115611            -0.076363
                        SafeTAE-safe 4.804843 1.660837              -1.824928   0.018229         0.083333  0.519531              0.100424            -0.067803
                         SRA-BP-safe 4.813149 1.660950              -1.655217   0.035156         0.114583  0.000000              0.091085            -0.058200
           SafeTAE-expert-thresholds 4.822207 1.665389              -1.470127   0.049479         0.229167  0.727865              0.080899            -0.056813
             SafeTAE-pareto-balanced 4.823968 1.665204              -1.434156   0.026042         0.083333  0.575521              0.078920            -0.050077
                SafeTAE-mrc-features 4.827036 1.666710              -1.371459   0.044271         0.208333  0.692708              0.075470            -0.052051
               SafeTAE-expert-lambda 4.828304 1.667077              -1.345549   0.044271         0.208333  0.692708              0.074044            -0.051251
             SafeTAE-no-mrc-features 4.829784 1.667312              -1.315322   0.039062         0.208333  0.677083              0.072381            -0.050327
                   MRC-ridge-abstain 4.835280 1.668420              -1.203025   0.026042         0.114583  0.000000              0.066201            -0.031990
                 SafeTAE-slice-aware 4.841244 1.669806              -1.081163   0.027344         0.072917  0.600260              0.059495            -0.040280
                 SafeTAE-config-veto 4.848048 1.671446              -0.942144   0.018229         0.072917  0.550781              0.051845            -0.033248
                 SafeTAE-pareto-safe 4.848048 1.671446              -0.942144   0.018229         0.072917  0.550781              0.051845            -0.033248
                                LRBN 4.894158 1.682162               0.000000   0.000000         0.000000  0.000000              0.000000             0.000000
                   TAE-router-stage6 5.329668 1.722548               8.898583   0.305990              NaN  0.000000                   NaN                  NaN
                   TAE-ranker-stage6 5.913439 1.823557              20.826486   0.394531              NaN  0.000000                   NaN                  NaN
```

## Mechanism Flags

```json
{
  "stage": "stage8_safe_tae_pareto",
  "test_threshold_leakage": false,
  "safe_pass": false,
  "balanced_pass": false,
  "status": "conservative_limit_confirmed",
  "next": "stop_safe_tae_expansion_or_redesign_expert_pool",
  "h1_expert_specific_releases_coverage": false,
  "h2_expert_thresholds_help": false,
  "h3_expert_lambda_help": false,
  "h4_slice_aware_help": false,
  "h5_config_veto_help": true,
  "h6_mrc_features_help": true,
  "h7_new_pareto_frontier": false,
  "safe_delta_pct_vs_lrbn": -0.9421441343426503,
  "balanced_delta_pct_vs_lrbn": -1.4341556931008765,
  "stage7_safe_delta_pct_vs_lrbn": -1.824927935286098,
  "sra_balanced_delta_pct_vs_lrbn": -2.59850812165216
}
```