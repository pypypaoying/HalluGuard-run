# Stage 14 Selector Mechanism Validation

Status: `compact_failed_stop_before_mini_extension`.

## Overall

| variant | mse | mae | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | coverage | selected_nonharm_rate | oracle_gain_fraction | ci95_high_delta_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan | nan | 0.000000 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.197917 | 0.000000 | nan | 0.069900 | -0.099202 |
| oracle_stage14_cga_full | 3.074767 | 1.322182 | -37.174740 | 0.000000 | 0.000000 | 0.000000 | nan | 1.000000 | -1.564427 |
| Stage10 hard selector | 4.835257 | 1.671961 | -1.203497 | 0.000000 | 0.000000 | 0.134115 | 1.000000 | 0.032374 | -0.047564 |
| FamilyMix Selector | 4.827022 | 1.669096 | -1.371750 | 0.002604 | 0.010417 | 0.669271 | 0.996109 | 0.036900 | -0.058987 |
| Two-stage Cost-Sensitive Router | 4.851787 | 1.674523 | -0.865746 | 0.000000 | 0.000000 | 0.350260 | 1.000000 | 0.023289 | -0.035613 |
| ListSafe Top-k Selector | 4.829996 | 1.669808 | -1.310979 | 0.002604 | 0.010417 | 0.679688 | 0.996169 | 0.035265 | -0.056170 |
| Retrieval-Prior Selector | 4.845259 | 1.672923 | -0.999119 | 0.002604 | 0.010417 | 0.500000 | 0.994792 | 0.026876 | -0.042011 |
| Bayes-Abstain Selector | 4.863888 | 1.676672 | -0.618487 | 0.000000 | 0.000000 | 0.319010 | 1.000000 | 0.016637 | -0.024828 |

## Selector Top-k

| variant | family_top2_hit | family_top1_hit | candidate_top2_hit |
| --- | --- | --- | --- |
| FamilyMix Selector | 0.682292 | 0.434896 | 0.109375 |
| Two-stage Cost-Sensitive Router | 0.682292 | 0.434896 | 0.109375 |
| ListSafe Top-k Selector | 0.677083 | 0.432292 | 0.082031 |
| Retrieval-Prior Selector | 0.682292 | 0.412760 | 0.109375 |
| Bayes-Abstain Selector | 0.682292 | 0.434896 | 0.054688 |

## Gate Table

| variant | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | selected_nonharm_rate | family_top2_hit | candidate_top2_hit | oracle_gain_fraction | coverage | bootstrap_high_delta_raw | safe_gate_pass | balanced_gate_pass | compact_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FamilyMix Selector | -1.371750 | 0.002604 | 0.010417 | 0.996109 | 0.682292 | 0.109375 | 0.036900 | 0.669271 | -0.058987 | False | False | False |
| Two-stage Cost-Sensitive Router | -0.865746 | 0.000000 | 0.000000 | 1.000000 | 0.682292 | 0.109375 | 0.023289 | 0.350260 | -0.035613 | False | False | False |
| ListSafe Top-k Selector | -1.310979 | 0.002604 | 0.010417 | 0.996169 | 0.677083 | 0.082031 | 0.035265 | 0.679688 | -0.056170 | False | False | False |
| Retrieval-Prior Selector | -0.999119 | 0.002604 | 0.010417 | 0.994792 | 0.682292 | 0.109375 | 0.026876 | 0.500000 | -0.042011 | False | False | False |
| Bayes-Abstain Selector | -0.618487 | 0.000000 | 0.000000 | 1.000000 | 0.682292 | 0.054688 | 0.016637 | 0.319010 | -0.024828 | False | False | False |

## Verdict

```json
{
  "stage": "stage14_selector_mechanism",
  "status": "compact_failed_stop_before_mini_extension",
  "compact_pass": false,
  "passed_variants": [],
  "best_variant": "FamilyMix Selector",
  "best_mse": 4.8270219779235175,
  "best_mae": 1.6690956961026686,
  "best_mse_delta_pct_vs_lrbn": -1.3717502272030029,
  "best_harm_rate": 0.0026041666666666665,
  "best_max_config_harm": 0.010416666666666666,
  "best_selected_nonharm_rate": 0.9961089494163424,
  "best_oracle_gain_fraction": 0.03690006213779931,
  "test_threshold_leakage": false,
  "stop_reason": "no selector candidate passed compact selected-subset safety/balanced gates"
}
```

Output directory: `experiments\halluguard\results\stage14_selector_mechanism`