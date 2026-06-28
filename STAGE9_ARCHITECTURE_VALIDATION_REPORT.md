# Stage 9 Architecture Validation Report

## Objective

This stage stopped the Safe-TAE micro-adjustment loop and tested the large-architecture hypotheses from `deep-research-report (1).md`. The goal was not to submit a new TableA method, but to answer which direction has credible ceiling-breaking evidence under the existing compact protocol.

## Protocol

- Parent baseline: frozen `HalluGuard-LRBN`.
- Compact scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192`, seed `2026`.
- Calibration: validation-only inner train/calib.
- Evaluation: test-only.
- Bootstrap: `2000`.
- Output directory: `experiments/halluguard/results/stage9_architecture_validation`.
- Test threshold leakage: `False`.

Run command:

```powershell
python experiments\halluguard\run_stage9_architecture_validation.py --metrics-csv experiments\halluguard\results\research_direction_validation\forecast_inputs\combined_metrics.csv --stage5-dir experiments\halluguard\results\lrbn_sra_bp_stage5 --stage6-dir experiments\halluguard\results\stage6_mechanism --stage7-dir experiments\halluguard\results\stage7_safe_tae --stage8-dir experiments\halluguard\results\stage8_safe_tae_pareto --stage3-dir experiments\halluguard\results\lrbn_bp_stage3 --output-dir experiments\halluguard\results\stage9_architecture_validation --seed 2026 --n-bootstrap 2000
```

## Headline Verdict

Status: `oracle_space_found_but_deployable_selector_missing`.

The strongest result is not a deployable new method. It is a mechanism diagnosis: **the current ceiling is more likely candidate-pool + arbitration limited than threshold/lambda limited.** Expanding the candidate pool lowered deployable oracle MSE from `4.217668` to `3.908520`, a `-7.329840%` improvement over the current deployable oracle and `-20.139072%` versus LRBN. However, the deployable prototypes failed to safely capture that space.

Recommended next parent direction: **candidate-pool redesign plus a redesigned hierarchical arbitrator**.

## Prototype Results

| Prototype | Verdict | Key Evidence |
| --- | --- | --- |
| Restricted-oracle candidate-pool benchmark | Supported as diagnostic | Expanded deployable oracle MSE `3.908520` vs current deployable oracle `4.217668`; extra oracle delta `-7.329840%`. |
| Leave-LRBN + eligibility distillation | Partial representation signal, not sufficient | Global gain AUC `0.878052` vs Stage 7 `0.865465`; harm AUC `0.841677` vs `0.811531`; but test top-2 oracle hit only `0.167969`. |
| MRC-v2 multiscale residual distribution | Failed | MSE delta only `-0.373710%`; harm `0.100260`; max config harm `0.302083`; no safe or tradeoff pass. |
| Energy feasibility reranker | Failed as deployable selector | Score-gain Spearman `0.292761`, but selected policy worsened MSE by `+4.918740%`, harm `0.166667`, max config harm `0.406250`. |
| Online spectral meta-calibration | Protocol-clean but unsafe | Spectral adapter delta `-1.158871%`, better than rolling mean by `0.561135pp`, but harm `0.467448` and coverage gap `6.658664pp`. |

## Candidate-Pool Diagnosis

The expanded deployable oracle selected the new candidates frequently:

- `teacher_median_smoothing`: `19.5313%`
- `residual_quantile_median`: `14.8438%`
- `residual_memory_knn`: `8.0729%`
- `residual_ridge_refit`: `3.1250%`
- `jump_aware_boundary`: not a major oracle contributor in this first implementation

This means the old Stage 7/8 expert pool was missing useful trajectory proposals. The largest immediate signal comes from smoothing-teacher and residual-distribution/memory-style candidates, not another boundary-only expert.

## Why Current Selectors Failed

The pairwise representation improved only modestly:

- gain AUROC improved by about `+0.0126`
- harm AUROC improved by about `+0.0301`
- test AUROC remained decent: gain `0.858220`, harm `0.824390`

But ranking the actual best candidate stayed poor: top-2 oracle hit was only `16.7969%`. In other words, sample/candidate features can predict generic gain/harm better than before, but they still do not identify the right expert among a wider candidate pool.

The energy scorer shows the same split: it has nonzero rank signal (`Spearman 0.292761`) but fails deployment utility. This supports the report's warning that a plausibility/energy model can learn a weak ordering while still selecting unsafe trajectories.

## MRC-v2 and Online Findings

The MRC-v2 prototype did not validate the multiscale residual distribution line in its current lightweight form. Its selected policy was not even calibration-feasible, and test harm was too high. It improved high-gap slices slightly but harmed `low_gap_high_repair` and non-boundary slices:

- overall MSE delta `-0.373710%`, harm `0.100260`
- high-gap/low-repair delta `-0.576366%`, harm `0.036810`
- low-gap/high-repair delta `-0.191756%`, harm `0.175439`
- non-boundary delta `-0.325981%`, harm `0.121951`

Online spectral calibration remains a possible deployment story only after serious safety redesign. It was protocol-clean and beat rolling mean on MSE, but harm `46.7448%` is far outside the acceptable range.

## Direction Ranking

1. **Candidate-pool redesign + hierarchical arbitrator: strongest next direction.**
   The oracle gain is large and directly addresses the Stage 8 conservative limit. The next experiment should not be a new threshold sweep; it should redesign the decision objective to choose among a richer pool.

2. **Forecastability fingerprint / suitability representation: promising but incomplete.**
   AUC moved in the right direction, but top-k suitability is not yet usable. This should be merged into the hierarchical arbitrator as a representation layer, not treated as a standalone router yet.

3. **Energy-style scoring: keep as diagnostic only.**
   There is rank signal, but deployment utility is negative. It needs hard-negative construction and explicit harm-aware constrained learning before another full prototype.

4. **MRC-v2 residual distribution: not ready.**
   This compact implementation failed both performance and safety gates. It should not be the next parent unless the residual model is redesigned much more substantially.

5. **Online spectral calibration: secondary/appendix direction.**
   The protocol guard passes, but harm is too high. It can be revisited as a separate online story, not as the immediate offline clean-claim improvement route.

## Output Completeness

All required Stage 9 outputs were generated:

- `stage9_config.json`
- `prototype1_restricted_oracle.csv`
- `prototype1_oracle_distribution.csv`
- `prototype2_pairwise_metrics.csv`
- `prototype2_stage7_head_reference.csv`
- `prototype2_top2_hit.csv`
- `prototype3_mrc_v2_metrics.csv`
- `prototype3_mrc_v2_grid.csv`
- `prototype4_energy_metrics.csv`
- `prototype4_energy_grid.csv`
- `prototype4_energy_selected_distribution.csv`
- `prototype5_online_spectral_metrics.csv`
- `prototype5_online_conformal.csv`
- `stage9_overall.csv`
- `stage9_per_config.csv`
- `stage9_slice_metrics.csv`
- `stage9_verdict.json`
- `summary.md`

## Recommendation

Do not continue with small Safe-TAE Pareto edits. The next credible breakthrough attempt should be a larger **candidate-pool redesign + hierarchical arbitration** stage:

1. Build a richer deployable candidate pool around smoothing-teacher, residual-distribution, and retrieval-memory proposals.
2. Train a two-stage selector: first decide whether to leave LRBN, then choose a candidate family, then choose the specific candidate.
3. Optimize the objective for top-k oracle capture under harm constraints, not only pointwise gain/harm AUC.
4. Keep MRC-v2 and energy scoring as auxiliary features/diagnostics until they demonstrate safe decision utility.
