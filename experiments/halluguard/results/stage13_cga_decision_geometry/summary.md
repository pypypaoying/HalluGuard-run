# Stage 13 CGA Decision-Geometry Validation

Status: `compact_failed_stop_before_mini_extension`.

## Overall

| variant | mse | mae | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | coverage | oracle_gain_fraction | accept_precision | expected_observed_harm_gap_pp | ci95_high_delta_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan | nan | nan | 0.000000 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.197917 | 0.000000 | 0.069900 | nan | nan | -0.099202 |
| oracle_stage13_cga_full | 3.074767 | 1.322182 | -37.174740 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | nan | nan | -1.564427 |
| Residual-Prior Convex Mixer | 4.880528 | 1.679483 | -0.278490 | 0.268229 | 0.479167 | 0.998698 | 0.007491 | 0.731421 | 26.710439 | -0.010835 |
| Time-Step Gated Hybrid Editor | 4.893841 | 1.679466 | -0.006465 | 0.277344 | 0.572917 | 0.500000 | 0.000174 | 0.708333 | 18.360731 | 0.012587 |
| Selection-Conditional Conformal Family Editor | 4.844880 | 1.680459 | -1.006873 | 0.470052 | 0.531250 | 0.994792 | 0.027085 | 0.527487 | 4.360225 | -0.024156 |
| Retrieval-Augmented Local Residual Editor | 4.906838 | 1.680344 | 0.259102 | 0.472656 | 0.593750 | 1.000000 | -0.006970 | 0.527344 | 41.296539 | 0.029028 |
| Conservative Challenger Comparator | 4.875935 | 1.680378 | -0.372335 | 0.436198 | 0.552083 | 0.998698 | 0.010016 | 0.563233 | 43.602492 | -0.011526 |

## Gate Table

| variant | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | oracle_gain_fraction | coverage | accept_precision | selected_nonharm_rate | expected_observed_harm_gap_pp | q4_boundary_delta_pct | non_boundary_delta_pct | low_gap_high_repair_delta_pct | known_harmed_config_delta_pct | bootstrap_high_delta_raw | compact_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Residual-Prior Convex Mixer | -0.278490 | 0.268229 | 0.479167 | 0.007491 | 0.998698 | 0.731421 | 0.731421 | 26.710439 | -0.265701 | -0.282505 | -0.245921 | 0.113004 | -0.010835 | False |
| Time-Step Gated Hybrid Editor | -0.006465 | 0.277344 | 0.572917 | 0.000174 | 0.500000 | 0.708333 | 0.708333 | 18.360731 | 0.026578 | -0.016837 | -0.000954 | 0.581869 | 0.012587 | False |
| Selection-Conditional Conformal Family Editor | -1.006873 | 0.470052 | 0.531250 | 0.027085 | 0.994792 | 0.527487 | 0.527487 | 4.360225 | -1.332105 | -0.904783 | -0.904919 | -0.756869 | -0.024156 | False |
| Retrieval-Augmented Local Residual Editor | 0.259102 | 0.472656 | 0.593750 | -0.006970 | 1.000000 | 0.527344 | 0.527344 | 41.296539 | -0.099088 | 0.371538 | 0.544797 | 0.067113 | 0.029028 | False |
| Conservative Challenger Comparator | -0.372335 | 0.436198 | 0.552083 | 0.010016 | 0.998698 | 0.563233 | 0.563233 | 43.602492 | -0.395448 | -0.365080 | -0.338642 | -0.095093 | -0.011526 | False |

## Verdict

```json
{
  "stage": "stage13_cga_decision_geometry",
  "status": "compact_failed_stop_before_mini_extension",
  "compact_pass": false,
  "passed_variants": [],
  "best_variant": "Selection-Conditional Conformal Family Editor",
  "best_mse": 4.844879639426057,
  "best_mae": 1.6804593331810989,
  "best_mse_delta_pct_vs_lrbn": -1.0068731034946523,
  "best_harm_rate": 0.4700520833333333,
  "best_max_config_harm": 0.53125,
  "best_oracle_gain_fraction": 0.02708487255700192,
  "family_top2_hit": 0.6197916666666666,
  "candidate_top2_hit": 0.09375,
  "test_threshold_leakage": false,
  "stop_reason": "no decision-geometry candidate passed compact gates"
}
```

Output directory: `experiments\halluguard\results\stage13_cga_decision_geometry`