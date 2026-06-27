# Stage 8 Safe-TAE Pareto Mechanism Validation Report

## Objective

Stage 8 tested whether the Stage 7 `SafeTAE-safe` line could release more of the strong TAE oracle gain while preserving low harm. The experiment kept `HalluGuard-LRBN` as the frozen parent and used validation-only calibration for expert thresholds, expert lambdas, mechanism-slice policies, config vetoes, and MRC feature ablations.

## Fixed Protocol

- Scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192`, seed `2026`.
- Input budget: compact validation/test windows inherited from Stage 5-7 artifacts.
- Calibration: validation-only inner train/calib split.
- Evaluation: test-only.
- Bootstrap: `2000`.
- Output directory: `experiments/halluguard/results/stage8_safe_tae_pareto`.
- Test threshold leakage: `False`.

Run command:

```powershell
python experiments\halluguard\run_stage8_safe_tae_pareto.py --metrics-csv experiments\halluguard\results\research_direction_validation\forecast_inputs\combined_metrics.csv --stage5-dir experiments\halluguard\results\lrbn_sra_bp_stage5 --stage6-dir experiments\halluguard\results\stage6_mechanism --stage7-dir experiments\halluguard\results\stage7_safe_tae --stage3-dir experiments\halluguard\results\lrbn_bp_stage3 --output-dir experiments\halluguard\results\stage8_safe_tae_pareto --seed 2026 --n-bootstrap 2000
```

## Verdict

Stage 8 does **not** pass the safe or balanced promotion gates.

- `safe_pass`: `False`
- `balanced_pass`: `False`
- status: `conservative_limit_confirmed`
- recommended next action: stop this Safe-TAE Pareto expansion or redesign the expert pool before another search loop.

The main finding is that additional Pareto machinery can reduce known harmed configs, but it also removes too much useful coverage. The high-performance aggressive policy confirms more oracle gain exists, but its config-level harm is too high for the safe claim.

## Key Overall Metrics

| Variant | MSE | MAE | MSE Delta vs LRBN | Harm Rate | Max Config Harm | Config Improved Ratio | Oracle Gain Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SRA-BP-balanced` | 4.766983 | 1.645627 | -2.598508% | 10.4167% | 19.7917% | 1.000 | 14.2993% |
| `SafeTAE-safe` Stage 7 reference | 4.804843 | 1.660837 | -1.824928% | 1.8229% | 8.3333% | NA | 10.0424% |
| `SafeTAE-pareto-safe` | 4.848048 | 1.671446 | -0.942144% | 1.8229% | 7.2917% | 0.875 | 5.1845% |
| `SafeTAE-pareto-balanced` | 4.823968 | 1.665204 | -1.434156% | 2.6042% | 8.3333% | 0.875 | 7.8920% |
| `SafeTAE-pareto-aggressive-diagnostic` | 4.791336 | 1.656973 | -2.100914% | 7.0312% | 30.2083% | 0.875 | 11.5611% |

Bootstrap CIs for MSE delta were negative for the tested policies, but the promotion gates require both significant improvement and controlled harm. The aggressive diagnostic meets the direction/significance signal but fails max config harm by a wide margin.

## Hypothesis Results

| Hypothesis | Result | Evidence |
| --- | --- | --- |
| H1 expert-specific releases coverage | Not supported | Coverage increases did not produce a new safe Pareto frontier. |
| H2 expert-specific thresholds help | Not supported | `SafeTAE-expert-thresholds` reached -1.470127% MSE delta but harm rose to 4.9479% and max config harm to 22.9167%. |
| H3 expert-specific lambda helps | Not supported | `SafeTAE-expert-lambda` reached -1.345549%, weaker than Stage 7 safe. |
| H4 slice-aware policy helps | Not supported | `SafeTAE-slice-aware` reduced max harm but only reached -1.081163%. |
| H5 config veto helps | Supported narrowly | Known harmed config slice was vetoed to 0% coverage, 0% harm, and 0% delta, but overall gain fell to -0.942144%. |
| H6 MRC features help | Weak support | MRC features improved the method from -1.315322% to -1.371459%; head-level AUC gains were tiny and mixed. |
| H7 new Pareto frontier | Not supported | No deployable Stage 8 variant beats Stage 7 safe while satisfying harm gates. |

## Slice Findings

`SafeTAE-pareto-safe`:

- overall MSE delta: -0.942144%, harm 1.8229%, coverage 55.0781%.
- high-gap / low-repair MSE delta: -1.580734%, harm 2.4540%.
- q4 boundary MSE delta: -1.732681%, harm 3.0928%.
- low-gap / high-repair MSE delta: -0.409607%, harm 3.5088%.
- known harmed config: 0% delta, 0% harm, 0% coverage.

`SafeTAE-pareto-balanced`:

- overall MSE delta: -1.434156%, harm 2.6042%, coverage 57.5521%.
- high-gap / low-repair MSE delta: -2.108140%, harm 5.5215%.
- q4 boundary MSE delta: -2.417952%, harm 5.6701%.
- low-gap / high-repair MSE delta: -0.624809%, harm 3.5088%.
- known harmed config: 0% delta, 0% harm, 0% coverage.

The config veto correctly protects the known harmed region, but the safe policies do not recover enough of the high-gap oracle opportunity.

## MRC Feature Ablation

MRC features are directionally useful but not decisive:

- `SafeTAE-mrc-features`: MSE delta -1.371459%, harm 4.4271%.
- `SafeTAE-no-mrc-features`: MSE delta -1.315322%, harm 3.9062%.
- gain-head inner-calib ROC changed by -0.0025, PR changed by +0.0005.
- harm-head inner-calib ROC changed by +0.0025, PR changed by +0.0003.

This supports keeping MRC as an auxiliary feature, but not as the main mechanism.

## Output Completeness

All required Stage 8 artifacts were generated:

- `stage8_config.json`
- `stage8_candidate_pool.csv`
- `stage8_pairwise_head_metrics.csv`
- `stage8_policy_grid_safe.csv`
- `stage8_policy_grid_balanced.csv`
- `stage8_overall.csv`
- `stage8_per_config.csv`
- `stage8_slice_metrics.csv`
- `stage8_expert_selection_distribution.csv`
- `stage8_expert_lambda_table.csv`
- `stage8_oracle_capture.csv`
- `stage8_keep_lrbn_recoverable_analysis.csv`
- `stage8_harmed_config_analysis.csv`
- `stage8_mrc_feature_ablation.csv`
- `stage8_bootstrap_ci.json`
- `stage8_verdict.json`
- `summary.md`

## Conclusion

Safe-TAE Pareto expansion does not enter the main TableA line. The best safe Stage 8 policy is more conservative than Stage 7 and loses too much MSE gain. The aggressive diagnostic shows that better expert choices could matter, but the current expert pool and gating cannot safely expose that gain.

The correct next research move is not another threshold/lambda sweep on this same pool. A redesigned expert pool or a separate mechanism should be tested if we continue beyond SafeTAE.
