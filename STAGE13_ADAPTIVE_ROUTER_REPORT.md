# Stage 13 Adaptive HalluGuard Router Report

## What Changed From Stage 12

Stage 12 selected `boundary_only` as the default HalluGuard-Dynamics variant.
Stage 13 tests a validation-only router that chooses among:

- `no_correction`
- `boundary_only`
- `dynamics_full`
- `median_smoothing`
- `ema_smoothing`
- `naive_smoothing`

The deployable Stage 13 router is `rule_router`, not a learned metadata router.
It uses only context/prediction features: boundary score, first-difference
score, curvature score, high-frequency excess, spectral distance, variance
ratio, diff-std ratio, context volatility, and horizon. Validation split is
used for thresholds and action policy fitting; test split is evaluation only.

## Implementation

New durable files:

- `experiments/halluguard/halluguard_router.py`
- `experiments/halluguard/run_stage13_adaptive_router.py`
- `experiments/halluguard/configs/halluguard_stage13_adaptive_router.yaml`

The reusable API exposes:

- `extract_router_features(sample, policy_context)`
- `fit_router(validation_samples, config, candidate_actions)`
- `apply_router(sample, router_policy)`
- `evaluate_router(validation_samples, test_samples, config)`

The final router setting uses a conservative rule:

- high boundary score -> `boundary_only`
- high noise/high-frequency score -> validation-best smoothing action
- low or uncertain risk -> `no_correction`

After smoke diagnostics, the learned/harm-aware routers were not selected as
the deployable router because they were smoothing-heavy and close to single
action behavior. The final `rule_router` uses `rule_noise_quantile=0.80` and a
`no_correction` fallback to avoid degenerating into full smoothing.

## Commands Run

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage13_adaptive_router.py --scope smoke --config experiments\halluguard\configs\halluguard_stage13_adaptive_router.yaml
D:\miniconda3\python.exe experiments\halluguard\run_stage13_adaptive_router.py --scope clean_full --config experiments\halluguard\configs\halluguard_stage13_adaptive_router.yaml
D:\miniconda3\python.exe experiments\halluguard\run_stage13_adaptive_router.py --scope stress --config experiments\halluguard\configs\halluguard_stage13_adaptive_router.yaml
D:\miniconda3\python.exe experiments\halluguard\run_stage13_adaptive_router.py --scope external_batch --config experiments\halluguard\configs\halluguard_stage13_adaptive_router.yaml
```

Main outputs:

- `experiments/halluguard/results/stage13_adaptive_router/clean_full_table/s13_adaptive_halluguard_router/`
- `experiments/halluguard/results/stage13_adaptive_router/stress_table/s13_adaptive_halluguard_router/`
- `experiments/halluguard/results/stage13_adaptive_router/external_batch/s13_adaptive_halluguard_router/`
- `experiments/halluguard/results/stage13_adaptive_router/diagnostics/`
- `experiments/halluguard/results/stage13_adaptive_router/candidate_ledger.csv`

## Clean Full Table

Completed `16/16` original configs:
ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192/336/720.

Deployable `rule_router` clean result:

- Mean MSE delta vs no correction: `-1.289319%`
- Mean MAE delta vs no correction: `-0.833485%`
- Improved configs: `16/16`
- Beats matched random action router: `15/16`
- Paired rule-vs-random win rate: `0.9500`
- Beats matched sparse smoothing: `12/16`
- Beats Stage 12 `boundary_only`: `15/16`
- Max MSE harm: `-0.061394%`
- Max MAE harm: `-0.048850%`
- Mean correction/action rate: `0.545288`
- Mean latency: `1.413832 ms/sample`
- Max single-action rate: `0.8789`
- Test threshold leakage: `False`

Clean gate verdict: pass.

Important clean baselines:

| Variant | Mean MSE Delta | Improved | Beats Matched | Beats Boundary |
| --- | ---: | ---: | ---: | ---: |
| `rule_router` | `-1.289319%` | 16 | 12 | 15 |
| `boundary_only` | `-0.654199%` | 15 | 6 | 0 |
| `matched_smoothing_control` | `-0.722079%` | 16 | 0 | 10 |
| `random_action_router` | `-0.993065%` | 16 | 8 | 9 |
| `naive_smoothing` | `-1.389826%` | 16 | 16 | 16 |
| `ema_smoothing` | `-1.366023%` | 16 | 15 | 16 |
| `median_smoothing` | `-1.886801%` | 16 | 16 | 16 |
| `stage9_incumbent` | `-0.058038%` | 15 | 0 | 2 |

Smoothing still wins pure point MSE: `median_smoothing` is stronger than the
router on clean MSE. The router's value is that it beats `boundary_only`,
matched random action routing, and matched sparse smoothing while keeping a
mixed, interpretable action distribution.

## Stress Table

Completed `96/96` stress configs:

- `boundary_discontinuity`
- `trend_drift`
- `slope_break`
- `delayed_level_shift`
- `high_frequency_perturbation`
- `variance_shift`

Overall stress result for `rule_router`:

- Mean MSE delta: `-1.390795%`
- Mean MAE delta: `-0.905077%`
- Improved configs: `96/96`
- Beats random action router: `90/96`
- Paired rule-vs-random win rate: `0.9375`
- Beats matched sparse smoothing: `67/96`
- Beats Stage 12 `boundary_only`: `83/96`
- Max MSE harm: `-0.065193%`
- Max MAE harm: `-0.052430%`
- Test threshold leakage: `False`

Boundary-discontinuity stress is the key mechanism test:

- `rule_router` mean MSE delta: `-1.482901%`
- `boundary_only` mean MSE delta: `-0.946615%`
- `matched_smoothing_control` mean MSE delta: `-0.739298%`
- `random_action_router` mean MSE delta: `-1.068281%`
- Paired random action wins: `15/16`
- Beats matched sparse smoothing: `15/16`
- Beats `boundary_only`: `10/16`
- Max MSE harm: `-0.157956%`

Boundary-discontinuity gate verdict: pass.

Per-stress interpretation:

| Stress | Rule MSE Delta | Paired Random Wins | Beats Matched | Beats Boundary |
| --- | ---: | ---: | ---: | ---: |
| `boundary_discontinuity` | `-1.482901%` | 15/16 | 15/16 | 10/16 |
| `trend_drift` | `-1.347718%` | 15/16 | 14/16 | 13/16 |
| `slope_break` | `-1.284239%` | 15/16 | 12/16 | 15/16 |
| `delayed_level_shift` | `-1.258358%` | 15/16 | 11/16 | 15/16 |
| `high_frequency_perturbation` | `-1.282335%` | 15/16 | 7/16 | 14/16 |
| `variance_shift` | `-1.689217%` | 15/16 | 8/16 | 16/16 |

High-frequency and variance-shift stress improve over `boundary_only`, but
they do not beat matched/full smoothing controls enough to claim frequency
repair. Smoothing still explains much of those stress gains.

## External Batch Fixture

The Stage 12 external-style fixture directory completed `16/16` files/groups:

- Mean MSE delta: `-0.930070%`
- Improved configs: `11/16`
- Beats random action router: `9/16`
- Paired win rate: `0.6000`
- Beats matched sparse smoothing: `9/16`
- Beats `boundary_only`: `11/16`
- Test threshold leakage: `False`

This fixture is only an integration smoke with `32 val` and `32 test` rows per
file. It proves the external path runs, but it is not the scientific evidence
for the Stage 13 claim.

## Action Distribution And Alignment

Clean full `rule_router` max single-action rate is `0.8789`, below the 90%
degeneracy line. The router uses a real mixture of boundary repair, smoothing,
and abstention.

Clean alignment diagnostics:

- Boundary action rate by boundary-score bin:
  - low: `0.0000`
  - mid: `0.0000`
  - high: `0.6173`
- Smoothing action rate by high-frequency/noise bin:
  - low: `0.2584`
  - mid: `0.3551`
  - high: `0.4039`

The boundary alignment is strong and interpretable. Noise-to-smoothing
alignment is positive but weaker, which matches the stress result: the router
is not a proven frequency repair method.

## Strong Baseline Confrontation

The router beats:

- Stage 12 `boundary_only` on clean mean MSE and `15/16` clean configs.
- Matched random action router on clean `15/16`, paired win `0.95`.
- Matched sparse smoothing on clean `12/16`.
- Stage 9 trend/frequency incumbent by a large margin.

The router does not beat:

- `median_smoothing` on clean pure point MSE.
- Full smoothing controls on high-frequency and variance stress.
- The validation-best-single-action baseline on clean MSE.

Therefore the Stage 13 result supports an adaptive robustness layer, not a
claim that HalluGuard-Dynamics replaces smoothing for point-MSE optimization.

## Verdict

Stage 13 passes the core clean and boundary-stress gates for the recommended
deployable router, `rule_router`.

The final recommended parent line is:

`HalluGuard-Dynamics Adaptive Router (rule_router, boundary/noise/no-correction actions)`

The scientific claim should remain narrow:

Validation-only adaptive routing can improve clean and stress robustness over
Stage 12 `boundary_only`, matched random action routing, and matched sparse
smoothing while preserving boundary-discontinuity mechanism evidence. It does
not establish frequency repair, and it does not beat full median smoothing on
pure point MSE.
