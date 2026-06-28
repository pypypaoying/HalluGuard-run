# Stage 15 Endogenous Low-Harm Editors

Status: `compact_failed_stop_before_mini_extension`.

## Overall

| variant | mse | mae | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | coverage | oracle_gain_fraction | lrbn_equiv_rate | active_patch_ratio | edit_energy_ratio | ci95_high_delta_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| SRA-BP-safe | 4.813149 | 1.660950 | -1.655217 | 0.035156 | 0.114583 | 0.000000 | 0.044525 | 0.812500 | 0.052228 | 0.001612 | -0.058200 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 | 0.197917 | 0.000000 | 0.069900 | 0.563802 | 0.121528 | 0.002535 | -0.099202 |
| oracle_stage15_cga_full | 3.074767 | 1.322182 | -37.174740 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.961372 | 0.055124 | -1.564427 |
| H1 Residual Atom Simplex Editor | 4.825331 | 1.669162 | -1.406308 | 0.007812 | 0.020833 | 0.923177 | 0.037830 | 0.766927 | 0.237703 | 0.000447 | -0.055759 |
| H3 Any-Quantile Residual Envelope | 4.890125 | 1.681795 | -0.082403 | 0.451823 | 0.500000 | 0.121094 | 0.002217 | 0.714844 | 0.204427 | 0.000004 | -0.002304 |
| H5 Local-Global Decoupled Sparse Editor | 4.870041 | 1.679941 | -0.492753 | 0.436198 | 0.520833 | 0.988281 | 0.013255 | 0.011719 | 0.943721 | 0.000081 | -0.015154 |
| H2 Prototype Codebook Local Editor | 4.893026 | 1.682025 | -0.023125 | 0.018229 | 0.041667 | 0.050781 | 0.000622 | 0.976562 | 0.007812 | 0.000001 | -0.000432 |
| H4 Retrieval-Conditioned Residual Adapter | 4.906913 | 1.683740 | 0.260622 | 0.076823 | 0.218750 | 0.148438 | -0.007011 | 0.851562 | 0.148438 | 0.000065 | 0.022178 |
| SafeTAE-safe (Stage7 table) | 4.804843 | 1.660837 | -1.824928 | 0.018229 | nan | 0.519531 | 0.100424 | nan | nan | nan | nan |
| Stage14 FamilyMix Selector | 4.827022 | 1.669096 | -1.371750 | 0.002604 | 0.010417 | 0.669271 | 0.036900 | nan | nan | nan | -0.058987 |

## Gate Table

| variant | mse_delta_pct_vs_lrbn | harm_rate | max_config_harm | oracle_gain_fraction | lrbn_equiv_rate | active_patch_ratio | edit_energy_ratio | q4_boundary_delta_pct | known_harmed_config_delta_pct | bootstrap_high_delta_raw | safe_gate_pass | tradeoff_gate_pass | mechanism_gate_pass | compact_gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 Residual Atom Simplex Editor | -1.406308 | 0.007812 | 0.020833 | 0.037830 | 0.766927 | 0.237703 | 0.000447 | -3.368323 | -0.392384 | -0.055759 | False | False | False | False |
| H3 Any-Quantile Residual Envelope | -0.082403 | 0.451823 | 0.500000 | 0.002217 | 0.714844 | 0.204427 | 0.000004 | -0.093007 | 0.008716 | -0.002304 | False | False | False | False |
| H5 Local-Global Decoupled Sparse Editor | -0.492753 | 0.436198 | 0.520833 | 0.013255 | 0.011719 | 0.943721 | 0.000081 | -0.761731 | 0.079131 | -0.015154 | False | False | False | False |
| H2 Prototype Codebook Local Editor | -0.023125 | 0.018229 | 0.041667 | 0.000622 | 0.976562 | 0.007812 | 0.000001 | -0.024570 | 0.000000 | -0.000432 | False | False | False | False |
| H4 Retrieval-Conditioned Residual Adapter | 0.260622 | 0.076823 | 0.218750 | -0.007011 | 0.851562 | 0.148438 | 0.000065 | -0.146881 | 1.848463 | 0.022178 | False | False | False | False |

## Verdict

```json
{
  "stage": "stage15_endogenous_editors",
  "status": "compact_failed_stop_before_mini_extension",
  "compact_pass": false,
  "passed_variants": [],
  "best_variant": "H1 Residual Atom Simplex Editor",
  "best_mse": 4.825330644544388,
  "best_mae": 1.6691624943025232,
  "best_mse_delta_pct_vs_lrbn": -1.406308439571461,
  "best_harm_rate": 0.0078125,
  "best_max_config_harm": 0.020833333333333332,
  "best_oracle_gain_fraction": 0.03782967757250403,
  "test_threshold_leakage": false,
  "stop_reason": "no endogenous editor passed compact safe/tradeoff gates"
}
```

Output directory: `experiments\halluguard\results\stage15_endogenous_editors`