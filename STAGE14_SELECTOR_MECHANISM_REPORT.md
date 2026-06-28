# Stage 14 Selector Mechanism Report

## Scope

This stage implements the validation plan from `deep-research-report (4).md`.

The key hypothesis was that the CGA candidate families are strong enough, but the selector geometry is wrong. Stage 14 therefore kept the candidate pool fixed and tested whether a family-level, selected-subset-safe selector can safely convert oracle space into deployable gains.

Compact protocol:

- Datasets: `ETTm1`, `ETTh1`
- Backbones: `DLinear`, `PatchTST`
- Horizons: `96`, `192`
- Seed: `2026`
- Baseline: frozen HalluGuard-LRBN
- Candidate pool: Stage 10 CGA families
- Selector heads: fit on validation inner-train only
- Thresholds / lambda / coverage caps: selected on validation inner-calib only
- Evaluation: test-only
- Bootstrap: `2000`
- Test threshold leakage: `False`

Outputs are in `experiments/halluguard/results/stage14_selector_mechanism/`.

## Implemented Selectors

1. `FamilyMix Selector`
   - Learns family-level safe routing, then uses convex residual mixture within selected families.
2. `Two-stage Cost-Sensitive Router`
   - Learns family admission first, then candidate utility minus harm.
3. `ListSafe Top-k Selector`
   - Ranks families/candidates for top-k capture rather than exact argmax.
4. `Retrieval-Prior Selector`
   - Uses validation-memory KNN family outcomes as selector prior only, not as an editor.
5. `Bayes-Abstain Selector`
   - Uses random-forest utility uncertainty and abstains unless lower-confidence utility is positive.

The implementation deliberately builds target-free family/candidate features for test-time scoring. Targets are used only to form inner-train labels and final metrics.

## Overall Results

| Variant | MSE | MAE | MSE delta vs LRBN | Harm | Max config harm | Coverage | Selected non-harm | Oracle capture | CI high raw delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LRBN | 4.894158 | 1.682162 | 0.000000% | 0.000000 | 0.000000 | 0.000000 | NA | NA | 0.000000 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508% | 0.104167 | 0.197917 | 0.000000 | NA | 0.069900 | -0.099202 |
| oracle_stage14_cga_full | 3.074767 | 1.322182 | -37.174740% | 0.000000 | 0.000000 | 0.000000 | NA | 1.000000 | -1.564427 |
| Stage10 hard selector | 4.835257 | 1.671961 | -1.203497% | 0.000000 | 0.000000 | 0.134115 | 1.000000 | 0.032374 | -0.047564 |
| FamilyMix Selector | 4.827022 | 1.669096 | -1.371750% | 0.002604 | 0.010417 | 0.669271 | 0.996109 | 0.036900 | -0.058987 |
| Two-stage Cost-Sensitive Router | 4.851787 | 1.674523 | -0.865746% | 0.000000 | 0.000000 | 0.350260 | 1.000000 | 0.023289 | -0.035613 |
| ListSafe Top-k Selector | 4.829996 | 1.669808 | -1.310979% | 0.002604 | 0.010417 | 0.679688 | 0.996169 | 0.035265 | -0.056170 |
| Retrieval-Prior Selector | 4.845259 | 1.672923 | -0.999119% | 0.002604 | 0.010417 | 0.500000 | 0.994792 | 0.026876 | -0.042011 |
| Bayes-Abstain Selector | 4.863888 | 1.676672 | -0.618487% | 0.000000 | 0.000000 | 0.319010 | 1.000000 | 0.016637 | -0.024828 |

## Gate Verdict

Status: **compact failed; stop before mini-extension**.

No selector passed the compact safe or balanced gate.

Best test MSE selector: `FamilyMix Selector`.

- MSE delta vs LRBN: `-1.371750%`
- Harm rate: `0.2604%`
- Max config harm: `1.0417%`
- Selected non-harm: `99.6109%`
- Oracle capture: `3.6900%`
- Family top-2 hit: `0.682292`
- Candidate top-2 hit: `0.109375`

FamilyMix fixes the Stage 13 selected-subset safety failure, but it does not capture enough oracle gain. It misses the safe gate on MSE improvement, family top-2, and oracle capture; it misses the balanced gate by a wider margin.

## Selector-Specific Findings

### FamilyMix Selector

This is the most useful Stage 14 result. It validates that selected-subset safety can be made real:

- selected non-harm improved from Stage 13 SCCFE's `52.7487%` to `99.6109%`;
- harm dropped to `0.2604%`;
- max config harm dropped to `1.0417%`.

But the selector is too conservative or too low-capture:

- oracle capture is only `3.6900%`;
- mean MSE delta is `-1.371750%`, below the `-1.8%` safe target;
- family top-2 is `0.682292`, below the `0.70` safe target.

It improves boundary-like slices more than non-boundary slices:

- `q4_boundary`: `-2.855463%`
- `high_gap_low_repair`: `-2.778602%`
- `non_boundary`: `-0.906014%`
- known harmed config: `-0.322093%`

### Two-stage Cost-Sensitive Router

Very safe but too weak:

- MSE delta: `-0.865746%`
- harm: `0.000000`
- oracle capture: `2.3289%`

The second-stage exact candidate utility scoring still does not recover enough of the family oracle.

### ListSafe Top-k Selector

ListSafe did not improve top-k behavior:

- MSE delta: `-1.310979%`
- family top-2: `0.677083`
- candidate top-2: `0.082031`
- oracle capture: `3.5265%`

This rejects the first lightweight top-k implementation. It does not reject listwise training in general, but it shows that simply changing the deploy ranking score is insufficient.

### Retrieval-Prior Selector

Retrieval prior was safe but not helpful enough:

- MSE delta: `-0.999119%`
- selected non-harm: `99.4792%`
- oracle capture: `2.6876%`

This supports the report's warning: retrieval can be a prior, but the current nearest-neighbor family prior is not enough to solve deployable selection.

### Bayes-Abstain Selector

Bayes-Abstain was the most conservative:

- MSE delta: `-0.618487%`
- harm: `0.000000`
- coverage: `31.9010%`
- oracle capture: `1.6637%`

The uncertainty abstain rule suppresses harm but also suppresses most useful gain.

## Interpretation

Stage 14 changes the failure diagnosis.

Stage 13 failed because selected subsets were unsafe. Stage 14 shows that selected-subset safety can be controlled, at least on this compact table. The new bottleneck is now gain capture:

- family top-2 remains below the report's target;
- candidate top-2 remains very weak;
- oracle capture remains far below the `8%` safe threshold and `12%` balanced threshold.

Therefore the selector mainline is **partially validated but not deployable**. The strongest next direction is not another threshold sweep. It should be one of:

1. true listwise/contrastive family representation learning, not heuristic top-k scoring;
2. a family-level objective that directly optimizes oracle capture under selected non-harm constraints;
3. candidate-family redesign toward intrinsically lower-harm candidates, because current safe selectors cannot access enough of the oracle space.

## Artifact Completeness

All required compact artifacts were produced:

- `stage14_config.json`
- `stage14_candidate_metadata.csv`
- `stage14_selector_train_metrics.csv`
- `stage14_calibration_grid.csv`
- `stage14_policies.json`
- `stage14_overall.csv`
- `stage14_per_config.csv`
- `stage14_slice_metrics.csv`
- `stage14_selection_distribution.csv`
- `stage14_topk_metrics.csv`
- `stage14_gate_table.csv`
- `stage14_verdict.json`
- `summary.md`

## Final Verdict

`compact_failed_stop_before_mini_extension`.

Do not run the mini-extension from this selector implementation. The results support the broad selector-focused diagnosis only partially: selected-subset safety is fixable, but family/candidate top-k and oracle capture are still too weak for a deployable CGA selector.

