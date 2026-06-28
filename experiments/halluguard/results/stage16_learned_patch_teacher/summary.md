# Stage 16 Learned Patch / Teacher Projector

Status: `compact_failed_stop_before_mini_extension`.

## Overall

| variant | mse | mae | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | coverage | oracle_gain_fraction | lrbn_equiv_rate | active_patch_ratio | ci95_high_delta_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan | 1.000000 | 0.000000 | 0.000000 |
| SRA-BP-safe | 4.813149 | 1.660950 | -1.655217 | 0.035156 | 0.114583 | 0.000000 | 0.044525 | 0.812500 | 0.052228 | -0.058200 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.197917 | 0.000000 | 0.069900 | 0.563802 | 0.121528 | -0.099202 |
| oracle_stage16_cga_full | 3.074767 | 1.322182 | -37.174740 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.961372 | -1.564427 |
| H2L Learned Patch Residual Editor | 4.899526 | 1.682689 | 0.109683 | 0.072917 | 0.177083 | 0.136719 | -0.002950 | 0.863281 | 0.051215 | 0.024563 |
| H6 Denoising Teacher Manifold Projector | 4.843571 | 1.672442 | -1.033610 | 0.024740 | 0.083333 | 0.148438 | 0.027804 | 0.851562 | 0.109086 | -0.034574 |
| H2H6 Learned Patch Teacher Hybrid | 4.806725 | 1.665870 | -1.786478 | 0.105469 | 0.187500 | 0.378906 | 0.048056 | 0.621094 | 0.211806 | -0.060479 |
| H6-Safe Sparse Teacher Projector | 4.844570 | 1.672768 | -1.013201 | 0.016927 | 0.041667 | 0.102865 | 0.027255 | 0.897135 | 0.088252 | -0.031398 |
| H2H6-Safe Sparse Learned Teacher Hybrid | 4.878476 | 1.678922 | -0.320413 | 0.059896 | 0.156250 | 0.143229 | 0.008619 | 0.856771 | 0.072338 | -0.002487 |
| SafeTAE-safe (Stage7 table) | 4.804843 | 1.660837 | -1.824928 | 0.018229 | nan | 0.519531 | 0.100424 | nan | nan | nan |
| Stage14 FamilyMix Selector | 4.827022 | 1.669096 | -1.371750 | 0.002604 | 0.010417 | 0.669271 | 0.036900 | nan | nan | -0.058987 |
| H1 Residual Atom Simplex Editor (Stage15) | 4.825331 | 1.669162 | -1.406308 | 0.007812 | 0.020833 | 0.923177 | 0.037830 | 0.766927 | 0.237703 | -0.055759 |
| H2 Prototype Codebook Local Editor (Stage15) | 4.893026 | 1.682025 | -0.023125 | 0.018229 | 0.041667 | 0.050781 | 0.000622 | 0.976562 | 0.007812 | -0.000432 |

## Gate Table

| variant | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | oracle_gain_fraction | lrbn_equiv_rate | active_patch_ratio | edit_energy_ratio | q4_boundary_delta_pct | known_harmed_config_delta_pct | bootstrap_high_delta_raw | safe_gate_pass | tradeoff_gate_pass | mechanism_gate_pass | compact_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H2L Learned Patch Residual Editor | 0.109683 | 0.072917 | 0.177083 | -0.002950 | 0.863281 | 0.051215 | 0.000169 | -0.365725 | 0.403311 | 0.024563 | False | False | False | False |
| H6 Denoising Teacher Manifold Projector | -1.033610 | 0.024740 | 0.083333 | 0.027804 | 0.851562 | 0.109086 | 0.000153 | -0.720079 | -0.913356 | -0.034574 | False | False | False | False |
| H2H6 Learned Patch Teacher Hybrid | -1.786478 | 0.105469 | 0.187500 | 0.048056 | 0.621094 | 0.211806 | 0.000490 | -1.597931 | -1.500589 | -0.060479 | False | False | False | False |
| H6-Safe Sparse Teacher Projector | -1.013201 | 0.016927 | 0.041667 | 0.027255 | 0.897135 | 0.088252 | 0.000263 | -0.439223 | -0.799539 | -0.031398 | False | False | False | False |
| H2H6-Safe Sparse Learned Teacher Hybrid | -0.320413 | 0.059896 | 0.156250 | 0.008619 | 0.856771 | 0.072338 | 0.000077 | -0.507295 | -0.442659 | -0.002487 | False | False | False | False |

## Verdict

```json
{
  "stage": "stage16_learned_patch_teacher",
  "status": "compact_failed_stop_before_mini_extension",
  "compact_pass": false,
  "passed_variants": [],
  "best_variant": "H2H6 Learned Patch Teacher Hybrid",
  "best_mse": 4.806724537686366,
  "best_mae": 1.6658698035605797,
  "best_mse_delta_pct_vs_lrbn": -1.7864781920779689,
  "best_harm_rate": 0.10546875,
  "best_max_config_harm": 0.1875,
  "best_oracle_gain_fraction": 0.04805623865644542,
  "test_threshold_leakage": false,
  "stop_reason": "no learned patch/teacher variant passed compact safe/tradeoff gates"
}
```

Output directory: `experiments\halluguard\results\stage16_learned_patch_teacher`