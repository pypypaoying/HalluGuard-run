# HalluGuard Next-Stage Validation Report

- Samples: 1536
- Configs: 8
- Contract: validation selects alphas/thresholds/models; test evaluates.
- Scope: second-stage compact validation, not final TableA.

## Verdicts

- `Experiment_A_HalluGuard_BP`: **promising** — BP-global delta -0.441847, harm 0.414, A>1 0.586; BP-domain delta -0.396469.
- `Experiment_B_Critic_Assisted_BP`: **weak** — score-delta/gain Spearman 0.174; at 50% coverage critic-random delta gap -0.0999775.
- `Experiment_C_Residual_Basis_Lite`: **weak** — mean sign accuracy gain vs majority 0.030; basis-lite test delta 1.59501.
- `Experiment_D_Multiscale_Support`: **weak** — best support/residual |Spearman| 0.073; adaptive shrink delta -0.259219.
- `Experiment_E_Regime_Invariant`: **weak** — global accuracy 0.505, purity 0.686, consistency 0.750; regime selector delta -1.23028.

## Experiment A: Boundary Projection

- `HalluGuard-BP-global`: delta -0.441847, harm 0.414, A>1 0.586
- `HalluGuard-BP-domain`: delta -0.396469, harm 0.435, A>1 0.565
- `HalluGuard-LRBN`: delta -1.53306, harm 0.358, A>1 0.642
- `ema_smoothing`: delta -0.366734, harm 0.000, A>1 1.000
- `matched_sparse_smoothing`: delta -0.204746, harm 0.268, A>1 0.721
- `median_smoothing`: delta -0.297291, harm 0.025, A>1 0.975
- `naive_smoothing`: delta -0.355151, harm 0.000, A>1 1.000

## Experiment B: Critic Selector

- split `test`: score_delta/gain Spearman 0.174, score_delta/harm Spearman -0.120
- split `val`: score_delta/gain Spearman 0.139, score_delta/harm Spearman -0.113

## Experiment C/D/E Outputs

- `residual_basis_sign_bucket.csv`
- `residual_basis_lite_correction.csv`
- `multiscale_support_retest.csv`
- `regime_mechanism_stability.csv`
- `stage2_direction_verdicts.csv`
