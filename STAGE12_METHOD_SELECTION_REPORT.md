# Stage 12 Method Selection Report

## Question

Should the recommended external HalluGuard-Dynamics method remain `dynamics_full`, or switch to a simpler boundary-focused variant?

Stage 12 uses the full Stage 11 clean and stress tables for this decision, not the compact Stage 12 fixture-smoke batch.

Source summary:

```text
experiments/halluguard/results/stage12_external_batch/method_selection_summary.csv
```

## Selection Rule

Prefer the simplest variant that:

- passes the clean gate
- passes the boundary-discontinuity gate
- beats matched sparse smoothing on clean in at least 10/16 configs
- beats matched sparse smoothing on boundary-discontinuity stress in at least 10/16 configs
- has max MSE and MAE harm under 3%
- behaves stably across DLinear and PatchTST
- does not merely imitate full smoothing

## Full Clean Table Comparison

| Variant | Clean MSE Delta | Clean MAE Delta | Improved | Beats Random | Paired Win | Beats Matched | Max MSE Harm | DLinear Mean | PatchTST Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dynamics_full` | `-0.623359%` | `-0.528044%` | 15 | 15 | 0.9375 | 12 | `0.067576%` | `-1.137802%` | `-0.108915%` |
| `boundary_only` | `-0.654199%` | `-0.567336%` | 15 | 15 | 0.9375 | 12 | `0.147763%` | `-1.148989%` | `-0.159409%` |
| `boundary_first_diff` | `-0.665592%` | `-0.570849%` | 15 | 14 | 0.9000 | 12 | `0.067576%` | `-1.173114%` | `-0.158070%` |
| `boundary_curvature` | `-0.658703%` | `-0.570780%` | 15 | 15 | 0.9375 | 12 | `0.082955%` | `-1.158032%` | `-0.159375%` |
| `first_diff_only` | `-0.024947%` | `-0.015096%` | 14 | 16 | 0.9125 | 1 | `0.006048%` | `-0.037665%` | `-0.012228%` |
| `curvature_only` | `0.000040%` | `0.000009%` | 8 | 7 | 0.5000 | 0 | `0.000280%` | `0.000053%` | `0.000027%` |

Clean interpretation:

- `boundary_first_diff` has the best clean mean MSE, but it loses one config in rule-vs-random count relative to `boundary_only` and uses a larger mechanism.
- `boundary_only` is simpler than `dynamics_full`, slightly stronger on clean MSE/MAE, and has the same clean rule-vs-random and matched-smoothing win counts.
- `boundary_curvature` is close, but `curvature_only` has no useful standalone signal, so adding curvature is not justified.
- `first_diff_only` has a rule-vs-random signal but almost no MSE gain and fails the matched-smoothing gate.

## Stress Comparison

| Variant | Boundary Stress MSE Delta | Boundary Beats Random | Boundary Beats Matched | High-Freq MSE Delta | High-Freq Beats Matched | Variance MSE Delta | Variance Beats Matched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dynamics_full` | `-0.931722%` | 15 | 13 | `-0.528242%` | 4 | `-0.462721%` | 1 |
| `boundary_only` | `-0.946615%` | 15 | 14 | `-0.647086%` | 9 | `-0.498467%` | 3 |
| `boundary_first_diff` | `-0.931201%` | 15 | 13 | `-0.641601%` | 8 | `-0.514730%` | 5 |
| `boundary_curvature` | `-0.945210%` | 15 | 14 | `-0.652458%` | 9 | `-0.505934%` | 3 |
| `first_diff_only` | `-0.023980%` | 1 | 0 | `-0.025185%` | 0 | `-0.024755%` | 0 |
| `curvature_only` | `0.000037%` | 1 | 0 | `0.000045%` | 0 | `0.000033%` | 0 |

Boundary stress interpretation:

- `boundary_only` passes the boundary-discontinuity gate and is slightly stronger than `dynamics_full` on mean MSE.
- `boundary_only` beats matched smoothing in `14/16` boundary-discontinuity configs, above the 10/16 gate.
- `boundary_curvature` is similarly strong but includes curvature despite curvature-only failure.
- High-frequency and variance-shift stress are still mostly smoothing-favorable settings; they should not be used to claim frequency repair.

## Strong Baseline Confrontation

Full clean MSE deltas:

- `boundary_only`: `-0.654199%`
- `matched_smoothing_control`: `-0.591854%`
- `random_trigger`: `-0.363539%`
- `stage9_incumbent`: `-0.058038%`
- `naive_smoothing`: `-1.389826%`
- `ema_smoothing`: `-1.366023%`
- `median_smoothing`: `-1.886801%`

HalluGuard-Dynamics wins over:

- old Stage 9 trend/frequency HalluGuard
- matched-count random correction
- matched sparse smoothing on enough clean and boundary-discontinuity configs to pass the mechanism gate

Smoothing wins on:

- pure clean point MSE
- high-frequency perturbation
- variance shift

This means the final claim should be dynamics-boundary repair with mechanism controls, not generic smoothing replacement.

## Recommendation

Recommend `boundary_only` as the Stage 12 external default variant.

Reasons:

- It is the simplest passing variant.
- It directly matches the validated boundary-discontinuity mechanism.
- It improves full clean mean MSE more than `dynamics_full`: `-0.654199%` vs `-0.623359%`.
- It preserves clean rule-vs-random strength: `15/16`, paired win `0.9375`.
- It beats matched sparse smoothing on clean: `12/16`.
- It passes boundary-discontinuity stress: MSE delta `-0.946615%`, beats random `15/16`, beats matched `14/16`.
- It is stable across both model families: DLinear mean `-1.148989%`, PatchTST mean `-0.159409%`.

Keep `dynamics_full` as a reported ablation and compatibility baseline, not the default external recommendation.

## Limitations

- `boundary_only` does not beat naive, EMA, or median smoothing on pure point MSE.
- The strongest evidence is on ETTm1/ETTh1 with existing DLinear/PatchTST predictions.
- High-frequency and variance-shift stress remain mostly explained by smoothing controls.
- Further external datasets should be evaluated before making broad forecasting-benchmark claims.
