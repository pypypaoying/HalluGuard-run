# HalluGuard Next-Stage Validation Report

## Scope

This report validates the second-stage innovation directions from `HalluGuard_next_stage_experiment_plan.docx` on the existing compact real-forecast validation assets.

- Runner: `experiments/halluguard/run_next_stage_validation.py`
- Input sample table: `experiments/halluguard/results/research_direction_validation/sample_features.csv`
- Input action table: `experiments/halluguard/results/research_direction_validation/action_alignment.csv`
- Output directory: `experiments/halluguard/results/research_direction_validation_stage2/`
- Samples: 1536 total, 768 test
- Configs: 8 compact configs
- Split contract: validation split selects alphas, thresholds, and lightweight models; test split only evaluates.
- Scope boundary: this is a second-stage validation screen, not the final TableA big table.

Command run:

```bash
python experiments/halluguard/run_next_stage_validation.py --sample-features experiments/halluguard/results/research_direction_validation/sample_features.csv --action-alignment experiments/halluguard/results/research_direction_validation/action_alignment.csv --output-dir experiments/halluguard/results/research_direction_validation_stage2
```

## Executive Verdict

The strongest route remains the frozen HalluGuard-LRBN clean-claim line. The next-stage experiments support Boundary Projection as a real but incomplete mechanism component: it improves test MSE and is strongest on high boundary-gap windows, but it is still weaker than LRBN and has a high harm rate. The critic, residual-basis, multiscale, and regime directions are not ready to become the main method.

Recommended next parent line:

1. Keep `HalluGuard-LRBN unified_revin_rdn_hybrid` as the clean-claim model.
2. Continue `HalluGuard-BP` only as a boundary-mechanism enhancement/ablation, not a replacement.
3. If pursuing critic work, use it for gradient-direction refinement experiments, not as a selector yet.
4. Pause residual-basis correction and current multiscale support mechanism.

## Direction Verdicts

| Direction | Verdict | Key Result | Decision |
|---|---:|---:|---|
| Experiment A: HalluGuard-BP | promising | BP-global MSE delta -0.441847; high boundary-gap quartile delta -1.080247 | Continue as boundary component / ablation |
| Experiment B: Critic-assisted BP | weak | score-delta/gain Spearman 0.174; 50% coverage critic-random gap -0.099977 | Do not use as selector; maybe test gradient refinement |
| Experiment C: Residual basis-lite | weak | sign accuracy gain only +0.030; basis-lite MSE delta +1.595007 | Pause correction route |
| Experiment D: Multiscale support | weak | best test support/residual Spearman 0.073; adaptive shrink delta -0.259219 | Treat as shrinkage baseline, not mechanism |
| Experiment E: Regime-invariant validation | weak | global accuracy 0.505; selector delta -1.230275 | Interesting but not robust enough |

## Experiment A: Boundary Projection

Validation selected a global boundary projection policy of `alpha=0.5`, `decay=linear`, `anchor=last`.

Final test summary from `bp_extended_table.csv`:

| Variant | Mean MSE | Delta vs raw | Delta % | Harm rate | Win rate |
|---|---:|---:|---:|---:|---:|
| HalluGuard-BP-global | 5.985373 | -0.441847 | -6.874626% | 0.414063 | 0.585938 |
| HalluGuard-BP-domain | 6.030751 | -0.396469 | -5.043842% | 0.434896 | 0.565104 |
| HalluGuard-LRBN | 4.894158 | -1.533063 | -5.099545% | 0.358073 | 0.641927 |
| matched_sparse_smoothing | 6.222475 | -0.204746 | -6.012603% | 0.268229 | 0.721354 |
| ema_smoothing | 6.060487 | -0.366734 | -10.962127% | 0.000000 | 1.000000 |
| median_smoothing | 6.129929 | -0.297291 | -9.466497% | 0.024740 | 0.975260 |
| naive_smoothing | 6.072069 | -0.355151 | -11.123758% | 0.000000 | 1.000000 |

Boundary-gap quantiles from `bp_boundary_gap_quantile.csv` show the intended mechanism is present:

| Boundary gap bin | Delta vs raw | Delta % | Harm rate | Win rate |
|---|---:|---:|---:|---:|
| q1_low | -0.056233 | -1.010102% | 0.432292 | 0.567708 |
| q2 | -0.128198 | -2.111845% | 0.463542 | 0.536458 |
| q3 | -0.502712 | -8.126518% | 0.375000 | 0.625000 |
| q4_high | -1.080247 | -13.699463% | 0.385417 | 0.614583 |

Interpretation: BP is a legitimate mechanism probe because its gain rises with boundary mismatch, but as a standalone correction it is weaker and more harmful than LRBN. Per-config domain calibration did not help; global validation calibration was better here.

## Experiment B: Critic-Assisted BP

The plausibility critic was tested as a selector for when to apply BP.

From `critic_score_delta_vs_gain.csv`:

| Split | Spearman score delta vs BP gain | Spearman score delta vs BP harm | Mean BP gain | BP harm rate |
|---|---:|---:|---:|---:|
| val | 0.139236 | -0.112887 | 0.497832 | 0.432292 |
| test | 0.173984 | -0.120250 | 0.441847 | 0.414063 |

Selector comparison from `critic_vs_boundarygap_selector.csv`:

| Coverage | Critic delta | Boundary-gap delta | Random delta | Critic minus random |
|---:|---:|---:|---:|---:|
| 0.25 | -0.059706 | -0.270062 | -0.066961 | 0.007255 |
| 0.50 | -0.296586 | -0.395740 | -0.196609 | -0.099977 |
| 0.75 | -0.474331 | -0.427789 | -0.374870 | -0.099461 |
| 1.00 | -0.441847 | -0.441847 | -0.441847 | 0.000000 |

The critic selector is not strong enough: score correlation is weak, and at low coverage it is worse than random. There is one useful signal in `critic_gradient_alignment.csv`: finite-difference critic gradients had mean cosine 0.125765 with the raw residual, `A>1` rate 0.979167, and mean one-step MSE delta -0.037750 on the inspected subset. That does not validate the critic as a selector, but it motivates a future critic-gradient BP refinement experiment.

## Experiment C: Residual Basis Sign / Bucket

The first-stage PCA residual-basis result looked strong as reconstruction evidence, so Stage 2 tested whether the coefficients can be predicted from context/prediction features.

From `residual_basis_sign_bucket.csv`:

- Mean sign accuracy: 0.515885
- Mean majority baseline accuracy: 0.485417
- Mean gain vs majority: about +0.030
- Mean AUC: 0.545170

From `residual_basis_lite_correction.csv`:

- Mean basis-lite test MSE delta: +1.595007
- Mean test MSE delta pct: +23.274209%
- Mean harm rate: 0.696615
- BP-global comparison on the same configs: -0.441847

Interpretation: residual basis remains useful as an analysis representation, but the current sign/bucket prediction is far too weak for correction. This route should be paused as a deployable method.

## Experiment D: Multiscale Support Retest

The second-stage test asked whether multiscale support scores actually predict scale-specific residual energy.

From `multiscale_support_retest.csv`:

- Test energy-support vs high-residual-energy Spearman: 0.072735
- Test phase-support vs mid-residual-energy Spearman: 0.018619
- Test local-band-boundary vs high-residual-energy Spearman: 0.050941
- Validation energy-support Spearman was negative: -0.321062

Adaptive scale shrink still improved point error:

- Mean MSE delta: -0.259219
- Mean delta pct: -4.033144%
- Harm rate: 0.000000
- Win rate vs raw: 0.658854

Interpretation: this is a usable shrinkage baseline but not yet evidence for the intended multiscale-support mechanism. It should not be promoted as a HalluGuard innovation unless the support score starts ranking true residual energy.

## Experiment E: Regime-Invariant Mechanism

The regime experiment created weak-supervised regime labels from action preference and mechanism signals, then tested global and leave-one-domain stability.

From `regime_mechanism_stability.csv`:

| Validation | Heldout | Accuracy | Macro F1 | Majority acc | Purity | Consistency |
|---|---|---:|---:|---:|---:|---:|
| global | - | 0.505208 | 0.262818 | 0.419271 | 0.685743 | 0.750000 |
| leave_one_dataset | ETTh1 | 0.596354 | 0.430865 | 0.497396 | 0.699863 | 0.750000 |
| leave_one_dataset | ETTm1 | 0.367188 | 0.168484 | 0.341146 | 0.704332 | 0.700000 |
| leave_one_backbone | DLinear | 0.567708 | 0.345488 | 0.432292 | 0.701955 | 0.750000 |
| leave_one_backbone | PatchTST | 0.455729 | 0.243510 | 0.406250 | 0.812236 | 0.900000 |

The regime-assisted selector produced:

- Mean MSE delta: -1.230275
- Delta pct: -19.141637%
- Harm rate: 0.363281
- Win rate: 0.623698

Interpretation: this result is interesting but not clean enough. The selector is close to LRBN in aggregate improvement, but the classifier is weak and labels are derived from oracle/action outcomes, so it should not become the main claim. It is a useful diagnostic path for discovering when LRBN/BP/raw should be selected.

## Final Recommendation

The second-stage evidence narrows the next innovation plan:

1. Main clean-claim model remains `HalluGuard-LRBN unified_revin_rdn_hybrid`.
2. Boundary Projection is the only second-stage method with direct mechanism evidence; use it as the next controlled mechanism enhancement, especially on boundary-discontinuity/high-boundary-gap slices.
3. Do not replace LRBN with BP: BP-global improves MSE but is much weaker than LRBN and has higher harm.
4. Do not promote critic-assisted selection yet. If continued, test critic-gradient residual refinement separately because the finite-difference gradient result is better than the selector result.
5. Pause residual-basis correction and current multiscale support claims.
6. Treat regime labels as a research diagnostic, not a deployable selector until cross-domain accuracy and macro-F1 improve.

The next full experiment should be: `LRBN + optional BP boundary enhancement`, with validation-only gating and a full TableA comparison against LRBN alone, smoothing controls, RevIN-style baselines, and test-time adaptation baselines.
