# HalluGuard Research Direction Validation

- Samples: 1536
- Action rows: 7680
- Configs: 8
- Split contract: validation trains/calibrates diagnostics; test evaluates.

## Direction Verdicts

- `E1_residual_alignment`: **promising** — Best action HalluGuard-LRBN test delta -1.53306, A>1 rate 0.642.
- `E2_oracle_action_separability`: **weak** — Accuracy 0.4283854166666667, majority 0.4114583333333333, shuffled 0.40234375.
- `E3_no_harm_selective_correction`: **weak** — Risk AUC 0.434; 50% coverage harm 0.424 vs full 0.358.
- `E4_residual_basis_decomposition`: **promising** — Mean top10 PCA EVR 0.914, test recon EVR 0.940, weight R2 -0.882.
- `E5_dynamic_consistency_projection`: **promising** — Best projection boundary_projection delta -0.224956 (-3.500%).
- `E6_multiscale_amplitude_phase_support`: **promising** — Multiscale edit delta -0.315904; hf mismatch/residual Spearman 0.085.
- `E7_energy_critic_separability`: **promising** — Critic AUC 0.938; gradient alignment not yet evaluated.
- `E8_tsfm_disagreement`: **blocked** — No local Chronos/TimesFM/Moirai forecast files were available; do not fake TSFM disagreement.
- `E9_regime_invariant_correction`: **promising** — Mean top-action rate 0.513, cross-domain consistency 0.667.

## Alignment Summary

- `HalluGuard-LRBN`: test mean delta -1.53306, harm 0.358, A>1 0.642
- `ema_smoothing`: test mean delta -0.366734, harm 0.000, A>1 1.000
- `naive_smoothing`: test mean delta -0.355151, harm 0.000, A>1 1.000
- `median_smoothing`: test mean delta -0.297291, harm 0.025, A>1 0.975
- `matched_sparse_smoothing`: test mean delta -0.204746, harm 0.268, A>1 0.721

## Risk / Coverage

- coverage 0.25: selected delta -0.490053, harm 0.411, risk AUC 0.434
- coverage 0.4: selected delta -0.446126, harm 0.440, risk AUC 0.434
- coverage 0.5: selected delta -0.466301, harm 0.424, risk AUC 0.434
- coverage 0.6: selected delta -0.604054, harm 0.406, risk AUC 0.434
- coverage 0.75: selected delta -1.0592, harm 0.380, risk AUC 0.434
- coverage 1.0: selected delta -1.53306, harm 0.358, risk AUC 0.434

## Key Output Files

- `sample_features.csv`
- `action_alignment.csv`
- `alignment_summary.csv`
- `oracle_action_separability.csv`
- `risk_coverage.csv`
- `basis_summary.csv`
- `projection_summary.csv`
- `multiscale_summary.csv`
- `critic_summary.csv`
- `regime_summary.csv`
- `direction_verdicts.csv`
