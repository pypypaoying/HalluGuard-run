# Stage19 Performance Atom Validation Plan

## Selected Idea

Stage19 validates two deployability-oriented performance atom adapters after `SRA-BP-balanced`: Residual Quantile Atom (RQA) and Non-Boundary Shape Atom (NBSA). Stage18 showed that `residual_distribution` and `smoothing_teacher` are the strongest oracle atom sources, but fixed atom centers and binary activation are not deployable; Stage19 therefore tests continuous coefficient prediction and shape-specific composition.

## Run Contract

- Parent: `SRA-BP-balanced`.
- References: `LRBN`, `SRA-BP-safe`, family oracle diagnostics.
- Compact protocol: ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026.
- Split contract: inner-train fits bases and coefficient models; inner-calib selects shrink/cap/coverage policies; test only evaluates.
- Primary metric: MSE delta vs `SRA-BP-balanced`.
- Safety metrics: harm rate vs SRA, max config harm, bootstrap CI, segment/slice harm.
- Mechanism metrics: coefficient sign accuracy/R2, A>1 rate, residual cosine, PCA/DCT explained structure, non-boundary and low-gap/high-repair slice effects.
- Stop condition: if no compact RQA/NBSA/combined variant passes safe/tradeoff gates, do not run mini-extension.

## Candidate Variants

- `RQA-PCA-Coef`
- `RQA-DCT-Coef`
- `RQA-QuantileHead`
- `RQA-HarmAwareCoef`
- `NBSA-DCT-Shape`
- `NBSA-RoughnessAdapter`
- `NBSA-NonBoundaryOnly`
- `NBSA-LocalShapeEnvelope`
- `RQA+NBSA`

## Required Outputs

All outputs are written under `experiments/halluguard/results/stage19_performance_atom_validation/`.

- `stage19_config.json`
- `family_oracle_targets.csv`
- `atom_basis_report.csv`
- `coefficient_fit_report.csv`
- `compact_variant_metrics.csv`
- `compact_per_config.csv`
- `compact_slice_metrics.csv`
- `compact_segment_metrics.csv`
- `complementarity_report.csv`
- `bootstrap_ci.json`
- `stage19_verdict.json`
- `stage19_output_completeness.csv`
- `summary.md`

## Decision Rules

- RQA safe gate: MSE <= -0.5%, harm <= 0.03, max config harm <= 0.10, CI upper < 0.
- RQA tradeoff gate: MSE <= -1.2%, harm <= 0.08, max config harm <= 0.18.
- NBSA safe gate: MSE <= -0.4%, harm <= 0.03, boundary degradation <= 0.3%.
- NBSA tradeoff gate: MSE <= -1.0%, harm <= 0.08, non-boundary improvement <= -2.0%.
- Combined safe gate: MSE <= -0.8%, harm <= 0.04, max config harm <= 0.12.
- Combined tradeoff gate: MSE <= -1.8%, harm <= 0.10, max config harm <= 0.20, combined gain beats best single by at least 0.3 pp.

## Revision Log

- 2026-06-28: Created compact Stage19 plan from the provided validation document.
