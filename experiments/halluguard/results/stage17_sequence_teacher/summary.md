# Stage 17 Sequence Teacher Projection

Status: `compact_failed_stop_before_mini_extension`.

## Overall

| variant | mse | mae | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | coverage | oracle_gain_fraction | lrbn_equiv_rate | active_patch_ratio | ci95_high_delta_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan | 1.000000 | 0.000000 | 0.000000 |
| SRA-BP-safe | 4.813149 | 1.660950 | -1.655217 | 0.035156 | 0.114583 | 0.000000 | 0.044525 | 0.812500 | 0.052228 | -0.058200 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.197917 | 0.000000 | 0.069900 | 0.563802 | 0.121528 | -0.099202 |
| oracle_stage17_cga_full | 3.074767 | 1.322182 | -37.174740 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.961372 | -1.564427 |
| SL-TMP Sequence Teacher Minimal-Norm Projection | 4.892852 | 1.681732 | -0.026667 | 0.108073 | 0.239583 | 0.235677 | 0.000717 | 0.764323 | 0.216291 | 0.000461 |
| UTRE Uncertainty Teacher Residual Envelope | 4.992662 | 1.680256 | 2.012688 | 0.316406 | 0.437500 | 0.645833 | -0.054141 | 0.354167 | 0.650174 | 0.175351 |
| SSP Structured Sequence Projector | 4.895090 | 1.682190 | 0.019053 | 0.075521 | 0.354167 | 0.141927 | -0.000513 | 0.858073 | 0.125000 | 0.002463 |
| TRAP Teacher-Residual Agreement Projector | 4.890483 | 1.681058 | -0.075090 | 0.087240 | 0.218750 | 0.225260 | 0.002020 | 0.774740 | 0.256076 | -0.001371 |
| IMDR Iterative Minimal-Norm Denoising Refiner | 4.891470 | 1.681722 | -0.054922 | 0.005208 | 0.031250 | 0.028646 | 0.001477 | 0.971354 | 0.034722 | -0.001039 |
| SafeTAE-safe (Stage7 table) | 4.804843 | 1.660837 | -1.824928 | 0.018229 | nan | 0.519531 | 0.100424 | nan | nan | nan |
| FamilyMix Selector (Stage14) | 4.827022 | 1.669096 | -1.371750 | 0.002604 | 0.010417 | 0.669271 | 0.036900 | nan | nan | -0.058987 |
| H1 Residual Atom Simplex Editor (Stage15) | 4.825331 | 1.669162 | -1.406308 | 0.007812 | 0.020833 | 0.923177 | 0.037830 | 0.766927 | 0.237703 | -0.055759 |
| H2 Prototype Codebook Local Editor (Stage15) | 4.893026 | 1.682025 | -0.023125 | 0.018229 | 0.041667 | 0.050781 | 0.000622 | 0.976562 | 0.007812 | -0.000432 |
| H6 Denoising Teacher Manifold Projector (Stage16) | 4.843571 | 1.672442 | -1.033610 | 0.024740 | 0.083333 | 0.148438 | 0.027804 | 0.851562 | 0.109086 | -0.034574 |
| H2H6 Learned Patch Teacher Hybrid (Stage16) | 4.806725 | 1.665870 | -1.786478 | 0.105469 | 0.187500 | 0.378906 | 0.048056 | 0.621094 | 0.211806 | -0.060479 |
| H6-Safe Sparse Teacher Projector (Stage16) | 4.844570 | 1.672768 | -1.013201 | 0.016927 | 0.041667 | 0.102865 | 0.027255 | 0.897135 | 0.088252 | -0.031398 |

## Gate Table

| variant | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | oracle_gain_fraction | lrbn_equiv_rate | active_patch_ratio | edit_energy_ratio | q4_boundary_delta_pct | non_boundary_delta_pct | known_harmed_config_delta_pct | bootstrap_high_delta_raw | teacher_energy_mse_delta_spearman | residual_alignment_A_gt1_rate | safe_gate_pass | tradeoff_gate_pass | mechanism_gate_pass | compact_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SL-TMP Sequence Teacher Minimal-Norm Projection | -0.026667 | 0.108073 | 0.239583 | 0.000717 | 0.764323 | 0.216291 | 0.000003 | -0.048672 | -0.019759 | 0.031020 | 0.000461 | 0.088627 | 0.127604 | False | False | False | False |
| UTRE Uncertainty Teacher Residual Envelope | 2.012688 | 0.316406 | 0.437500 | -0.054141 | 0.354167 | 0.650174 | 0.003655 | 2.869220 | 1.743823 | 0.478881 | 0.175351 | 0.065623 | 0.329427 | False | False | False | False |
| SSP Structured Sequence Projector | 0.019053 | 0.075521 | 0.354167 | -0.000513 | 0.858073 | 0.125000 | 0.000002 | 0.001492 | 0.024565 | 0.036090 | 0.002463 | 0.152683 | 0.066406 | False | False | False | False |
| TRAP Teacher-Residual Agreement Projector | -0.075090 | 0.087240 | 0.218750 | 0.002020 | 0.774740 | 0.256076 | 0.000004 | -0.164906 | -0.046897 | -0.060871 | -0.001371 | 0.283821 | 0.138021 | False | False | False | False |
| IMDR Iterative Minimal-Norm Denoising Refiner | -0.054922 | 0.005208 | 0.031250 | 0.001477 | 0.971354 | 0.034722 | 0.000002 | 0.015107 | -0.076904 | -0.217390 | -0.001039 | 0.630225 | 0.023438 | False | False | False | False |

## Mechanism Metrics

| variant | teacher_energy_delta_mean | teacher_energy_mse_delta_spearman | residual_alignment_A_gt1_rate | uncertainty_width_mean | edit_energy_ratio | active_patch_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| SL-TMP Sequence Teacher Minimal-Norm Projection | -0.000040 | 0.088627 | 0.127604 | nan | 0.000003 | 0.235677 |
| UTRE Uncertainty Teacher Residual Envelope | -0.001604 | 0.065623 | 0.329427 | 2.341665 | 0.003655 | 0.645833 |
| SSP Structured Sequence Projector | 0.000017 | 0.152683 | 0.066406 | nan | 0.000002 | 0.141927 |
| TRAP Teacher-Residual Agreement Projector | 0.000005 | 0.283821 | 0.138021 | nan | 0.000004 | 0.225260 |
| IMDR Iterative Minimal-Norm Denoising Refiner | -0.000002 | 0.630225 | 0.023438 | nan | 0.000002 | 0.028646 |

## Verdict

```json
{
  "stage": "stage17_sequence_teacher_projection",
  "status": "compact_failed_stop_before_mini_extension",
  "compact_pass": false,
  "mechanism_pass_any": false,
  "passed_variants": [],
  "mechanism_passed_variants": [],
  "best_variant": "TRAP Teacher-Residual Agreement Projector",
  "best_mse": 4.890482581896936,
  "best_mae": 1.6810577369664148,
  "best_mse_delta_pct_vs_lrbn": -0.07508981938178093,
  "best_harm_rate": 0.08723958333333333,
  "best_max_config_harm": 0.21875,
  "best_oracle_gain_fraction": 0.0020199151027320487,
  "teacher_training_source": "validation_inner_train_proxy_no_original_train_assets_available",
  "test_threshold_leakage": false,
  "stop_reason": "no Stage17 sequence teacher variant passed compact safe/tradeoff gates"
}
```

Output directory: `experiments\halluguard\results\stage17_sequence_teacher`