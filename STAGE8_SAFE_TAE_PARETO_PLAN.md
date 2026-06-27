# Stage 8 Safe-TAE Pareto Plan

## Objective

Validate whether Safe-TAE can release more of the Stage 6 TAE oracle gap without returning to unsafe top-1 routing. The Stage 8 line keeps LRBN fallback and pairwise no-harm heads, then tests expert-specific thresholds, expert-specific residual strengths, mechanism-slice calibration, config-risk veto, and MRC feature contribution.

## Fixed Compact Protocol

- Parent: frozen `HalluGuard-LRBN`
- Stage 5 dependency: `experiments/halluguard/results/lrbn_sra_bp_stage5`
- Stage 6 dependency: `experiments/halluguard/results/stage6_mechanism`
- Stage 7 dependency: `experiments/halluguard/results/stage7_safe_tae`
- Scope: ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192, seed 2026
- Calibration: validation-only inner train/calib
- Evaluation: test-only
- Bootstrap: 2000
- Output: `experiments/halluguard/results/stage8_safe_tae_pareto`

## Variants

- `SafeTAE-expert-thresholds`
- `SafeTAE-expert-lambda`
- `SafeTAE-slice-aware`
- `SafeTAE-config-veto`
- `SafeTAE-mrc-features`
- `SafeTAE-no-mrc-features`
- `SafeTAE-pareto-safe`
- `SafeTAE-pareto-balanced`
- `SafeTAE-pareto-aggressive-diagnostic`

## Pass Criteria

Safe target passes if MSE delta is at least `<= -2.0%` or improves Stage 7 SafeTAE-safe by 0.3pp, harm `<= 0.03`, max config harm `<= 0.10`, oracle gain fraction `>= 0.13`, CI95 upper below zero, config improved ratio `>= 0.625`, and low-gap/high-repair harm `<= 0.02`.

Balanced target passes if it beats SRA-BP-balanced, recommended MSE delta `<= -2.8%`, harm `<= 0.10`, max config harm `<= 0.20`, oracle gain fraction `>= 0.16`, CI95 upper below zero, config improved ratio `>= 0.625`, and high-gap/low-repair improves over Stage 7 SafeTAE-safe.

## Required Outputs

All outputs are written under `experiments/halluguard/results/stage8_safe_tae_pareto` and summarized in `STAGE8_SAFE_TAE_PARETO_VALIDATION_REPORT.md`.
