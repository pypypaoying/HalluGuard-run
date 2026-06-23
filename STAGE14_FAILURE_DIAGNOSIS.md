# Stage 14 Failure Diagnosis

Date: 2026-05-15
Branch: `autoresearch/halluguard-stage14-signal-preserve`

## Trusted Inputs

Stage 14 starts from the committed Stage 13 adaptive router line. The trusted evidence is the actual CSV/JSON output under:

- `experiments/halluguard/results/stage13_adaptive_router/clean_full_table/s13_adaptive_halluguard_router/`
- `experiments/halluguard/results/stage13_adaptive_router/stress_table/s13_adaptive_halluguard_router/`
- `experiments/halluguard/results/stage13_adaptive_router/external_batch/s13_adaptive_halluguard_router/`

The Stage 13 deployable method is `rule_router` with validation-only thresholds and no test-threshold leakage.

## 1. DLinear vs PatchTST Gap

Clean full table, MSE delta vs no correction:

| Variant | DLinear Mean | PatchTST Mean | Interpretation |
| --- | ---: | ---: | --- |
| `rule_router` | `-2.281087%` | `-0.297552%` | Strong on DLinear, much weaker on PatchTST. |
| `boundary_only` | `-1.148989%` | `-0.159409%` | Boundary repair helps both but is also DLinear-heavy. |
| `median_smoothing` | `-3.236555%` | `-0.537046%` | Full smoothing remains stronger than the router for point MSE. |
| `boundary_then_median` | `-4.377699%` | `-0.697388%` | The tempting overbuilt action is stronger, but looks smoothing-dominated. |
| `validation_best_single_action` | `-3.113765%` | `-0.642942%` | A validation-selected global action is still stronger than the router. |
| `matched_smoothing_control` | `-1.152790%` | `-0.291369%` | PatchTST router gain is close to sparse smoothing control. |

Main diagnosis: Stage 13 is useful, but its clean advantage is concentrated in DLinear. For PatchTST, the improvement is small and close to what sparse smoothing already explains.

## 2. Gap To Smoothing And Boundary-Then-Median

Stage 13 clean full means:

- `rule_router`: `-1.289319%`
- `median_smoothing`: `-1.886801%`
- `boundary_then_median`: `-2.537543%`
- `validation_best_single_action`: `-1.878354%`
- `naive_smoothing`: `-1.389826%`
- `ema_smoothing`: `-1.366023%`

The router beats matched random action routing and matched sparse smoothing, but it does not close the pure-MSE gap to full median smoothing or the overbuilt `boundary_then_median` action. This creates a claim risk: a reviewer can argue that the gains are still mostly smoothing availability plus boundary repair, not a distinct signal-preserving mechanism.

## 3. High-Frequency Stress Is Mostly Smoothing-Explained

High-frequency perturbation stress, MSE delta vs no correction:

| Variant | Mean Delta | Beats Matched Smoothing |
| --- | ---: | ---: |
| `rule_router` | `-1.282335%` | `7/16` |
| `matched_smoothing_control` | `-1.172253%` | baseline |
| `boundary_only` | `-0.647086%` | `0/16` |
| `median_smoothing` | `-2.495032%` | `16/16` |
| `naive_smoothing` | `-2.436927%` | `16/16` |
| `ema_smoothing` | `-2.367306%` | `16/16` |

The Stage 13 router improves high-frequency stress relative to `boundary_only`, but the strongest gains come from full smoothing. Stage 13 therefore does not prove that it can distinguish unsupported pseudo-noise from real signal changes.

## 4. External Fixture PatchTST Harm

The compact external fixture is integration smoke only, but it reveals a useful harm diagnostic:

- External `rule_router` DLinear mean: `-1.864570%`, harmed `0/8`.
- External `rule_router` PatchTST mean: `0.004431%`, harmed `4/8`.

PatchTST harmed fixture rows:

| Dataset | Horizon | Delta | Action Pattern |
| --- | ---: | ---: | --- |
| ETTh1 | 96 | `0.411351%` | boundary + EMA + abstain |
| ETTh1 | 192 | `0.105013%` | boundary + EMA + abstain |
| ETTh1 | 336 | `0.088351%` | boundary + EMA + abstain |
| ETTm1 | 720 | `0.143746%` | mostly abstain, some EMA/boundary |

Possible mechanism: PatchTST predictions are already smoother or more locally coherent than DLinear predictions. The Stage 13 high-frequency/noise rule may treat real local variation or already-safe residual structure as smoothing-worthy, especially in tiny external validation splits where action thresholds are less stable.

## 5. Minimum Bottleneck For Stage 14

Stage 14 should not simply add another smoothing action. The minimum bottleneck is:

> Decide when high-frequency or rough local structure is supported by the context and should be preserved, versus unsupported pseudo-noise that can be smoothed.

A useful Stage 14 module must therefore:

1. Preserve the Stage 13 boundary-discontinuity advantage.
2. Reduce smoothing overuse on PatchTST and external fixture cases.
3. Improve or at least not harm DLinear performance.
4. Show high-frequency stress behavior that is not fully explained by full smoothing or matched sparse smoothing.
5. Keep all thresholds and routing policy validation-only.

## Route Decision

Proceed to implement `HalluGuard Signal-Preserving Component Router` as a small output-space module. It should add explicit signal-support features and component-wise action rules rather than stacking full smoothing after boundary repair.
