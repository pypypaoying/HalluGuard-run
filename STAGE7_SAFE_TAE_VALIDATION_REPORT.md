# Stage 7 Safe-TAE Validation Report

## Setup

Stage 7 validates Safe-TAE as a compact, validation-only arbitration layer over the frozen HalluGuard-LRBN parent.

- Input metrics: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
- Stage 5 dependency: `experiments/halluguard/results/lrbn_sra_bp_stage5`
- Stage 6 dependency: `experiments/halluguard/results/stage6_mechanism`
- Output directory: `experiments/halluguard/results/stage7_safe_tae`
- Scope: ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192, seed 2026
- Calibration: validation inner train/calib only
- Evaluation: test split only
- Test threshold leakage: `False`

## Headline Verdict

SafeTAE-safe passes the compact safe gate and can enter the next candidate pool. SafeTAE-balanced does not pass: it is too conservative after validation calibration and does not beat SRA-balanced on point MSE.

| Variant | Test MSE | MSE Delta vs LRBN | Harm | Max Config Harm | Oracle Gain Fraction | CI95 Upper Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TAE-oracle-best | 4.004776 | -18.1723% | 0.0000 | 0.0000 | 1.0000 | -0.803251 |
| SRA-BP-balanced | 4.766983 | -2.5985% | 0.1042 | 0.1979 | 0.1430 | -0.099202 |
| SafeTAE-safe | 4.804843 | -1.8249% | 0.0182 | 0.0833 | 0.1004 | -0.067803 |
| SRA-BP-safe | 4.813149 | -1.6552% | 0.0352 | 0.1146 | 0.0911 | -0.058200 |
| MRC-ridge-abstain | 4.835280 | -1.2030% | 0.0260 | 0.1146 | 0.0662 | -0.031990 |
| LRBN | 4.894158 | 0.0000% | 0.0000 | 0.0000 | 0.0000 | 0.000000 |
| TAE-router-stage6 | 5.329668 | +8.8986% | 0.3060 | n/a | n/a | n/a |
| TAE-ranker-stage6 | 5.913439 | +20.8265% | 0.3945 | n/a | n/a | n/a |

## Hypothesis Results

- H1 pairwise no-harm heads are safer than Stage 6 top-1 TAE: supported. Stage 6 router/ranker harm was 30.60% / 39.45%, while SafeTAE-safe harm is 1.82%.
- H2 no-change gate protects LRBN better than pairwise hard replacement: not formally supported by the current gate. SafeTAE-safe and pairwise-hard have the same sample harm rate, though both are far safer than Stage 6 top-1 TAE.
- H3 residual blending is safer than hard replacement: supported. Pairwise-blend has lower harm than pairwise-hard, and the tiered/blended variants keep max per-config harm at 8.33%.
- H4 MRC residual-prior consistency improves precision: not supported. The selected MRC-consistency policy equals the SafeTAE-safe policy but does not improve over tiered blending.
- H5 Safe-TAE beats the SRA Pareto frontier: partially supported. SafeTAE-safe beats SRA-BP-safe on MSE and harm, but it does not beat SRA-BP-balanced on MSE.

## Mechanism Diagnostics

Validation-only pairwise heads are informative:

| Head | Split | AUROC | PR-AUC | Positive Rate |
| --- | --- | ---: | ---: | ---: |
| leave | inner_train | 0.9694 | 0.9991 | 0.9720 |
| leave | inner_calib | 0.8196 | 0.9983 | 0.9914 |
| gain | inner_train | 0.9041 | 0.9049 | 0.4257 |
| gain | inner_calib | 0.8655 | 0.8507 | 0.4193 |
| harm | inner_train | 0.8936 | 0.6183 | 0.1643 |
| harm | inner_calib | 0.8115 | 0.4136 | 0.1599 |

SafeTAE-safe selected a broad but bounded set of experts:

- `keep_lrbn`: 48.05%
- `volatility_shrink`: 22.53%
- `amplitude_scale_bounded`: 21.61%
- all boundary/MRC/ensemble experts combined: about 7.81%

The strongest slice remains the intended boundary-like region:

- high-gap/low-repair: MSE delta -4.2698%, harm 2.45%
- q4 boundary: MSE delta -4.3896%, harm 3.09%
- non-boundary: MSE delta -1.1694%, harm 1.65%
- low-gap/high-repair: MSE delta -1.0094%, harm 0.00%

## Per-Config Behavior

SafeTAE-safe improves 5/8 compact configs. Three configs are neutral or harmed:

- ETTh1/DLinear/96: 0.0000%
- ETTh1/DLinear/192: 0.0000%
- ETTm1/DLinear/192: +1.0249%

The single harmed config is still inside the safe gate because max per-config harm rate is 8.33% and the overall bootstrap CI upper bound is negative.

## Conclusion

SafeTAE-safe is a valid compact candidate: it converts part of the Stage 6 TAE oracle gap into deployable gain while controlling harm far better than the original TAE router/ranker. It should be carried as a safe arbitration ablation or candidate line.

It is not the new main clean-claim method yet. It does not beat SRA-BP-balanced on MSE, captures only about 10.04% of the oracle gain, and MRC-consistency did not add evidence. The next experiment should either simplify SafeTAE into a safe expert selector over SRA/volatility/amplitude experts or improve calibration around the ETTm1/DLinear/192 failure case without increasing harm.
