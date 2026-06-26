# HalluGuard Next-Stage Validation Plan

## Goal

Run the second-stage validation experiments described in `HalluGuard_next_stage_experiment_plan.docx`, using the first-stage sample-level forecast table as the common input.

## Parent Evidence

- Parent report: `RESEARCH_DIRECTIONS_VALIDATION_REPORT.md`
- Parent sample table: `experiments/halluguard/results/research_direction_validation/sample_features.csv`
- Parent action table: `experiments/halluguard/results/research_direction_validation/action_alignment.csv`

## Slices

1. Experiment A: Boundary Projection extended performance and mechanism validation.
2. Experiment B: Critic-assisted Boundary Projection selector validation.
3. Experiment C: Residual basis sign / bucket predictability.
4. Experiment D: Multiscale support mechanism retest.
5. Experiment E: Regime-invariant correction mechanism validation.

## Output Directory

`experiments/halluguard/results/research_direction_validation_stage2/`

## Key Outputs

- `bp_extended_table.csv`
- `bp_alignment_analysis.csv`
- `bp_boundary_gap_quantile.csv`
- `bp_ablation_alpha_global_vs_domain.csv`
- `critic_score_delta_vs_gain.csv`
- `critic_selected_risk_coverage.csv`
- `critic_vs_boundarygap_selector.csv`
- `critic_gradient_alignment.csv`
- `residual_basis_sign_bucket.csv`
- `residual_basis_lite_correction.csv`
- `multiscale_support_retest.csv`
- `regime_mechanism_stability.csv`
- `stage2_direction_verdicts.csv`
- `summary.md`

## Scope Note

This is still a second-stage pilot on the existing compact real-forecast sample set, not the final paper big table. It is intended to decide which innovation should receive the next full TableA-scale run.
