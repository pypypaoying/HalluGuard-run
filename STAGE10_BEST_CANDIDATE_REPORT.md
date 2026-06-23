# Stage 10 Best Candidate Report

## Selected Candidate

`s10_c2b_dynamics_anti_smoothing`

Family: dynamics-consistency trigger with anti-smoothing validation objective.

## Why This Is New

The Stage 8/9 incumbent used trend/frequency scores and FFT/trend correction. Stage 10 best candidate uses a different mechanism:

- boundary continuity between context end and prediction start
- first-difference continuity
- local curvature continuity
- decaying correction toward boundary/derivative consistency

The exploit adds a validation-only objective term that penalizes policies whose gain is explained by matched smoothing.

## Clean Gate

Passed.

- completed `16 / 16` configs
- no test threshold leakage
- mean MSE delta `-0.6232515127%`
- beats random `15 / 16`
- paired rule-vs-random win rate `0.9375`
- beats matched smoothing `12 / 16`
- no config MSE/MAE harm above `3%`

## Stress Gate

Partially passed, with a narrowed mechanism claim.

- completed `32 / 32` stress configs
- stress-only outputs are separate from clean table
- beats random `32 / 32`
- trend_drift: beats random `16 / 16`, beats matched `11 / 16`, mean MSE delta `-0.716644%`
- high_frequency_perturbation: beats random `16 / 16`, beats matched `4 / 16`, mean MSE delta `-0.528205%`

Interpretation: stress validates dynamics-continuity robustness, especially trend/boundary-like drift. It does not validate a frequency-specific repair claim because matched smoothing remains stronger on high-frequency perturbations.

## Comparison To Prior Frontier

| Method | Clean Mean MSE Delta | Rule Beats Random | Beats Matched | Paired Win |
| --- | ---: | ---: | ---: | ---: |
| Stage 9 incumbent | `-0.058038%` | `10 / 16` | not passed | `0.525` |
| `s10_c2` | `-0.664967%` | `15 / 16` | `4 / 16` | `0.925` |
| `s10_c2b` | `-0.623252%` | `15 / 16` | `12 / 16` | `0.9375` |

`s10_c2b` trades a small amount of MSE versus `s10_c2` for much stronger anti-smoothing separation.

## Recommendation

Continue with `s10_c2b` as the new best candidate. The next research step should rename and isolate it as a dynamics-continuity correction line, then run a paper-style table that keeps naive smoothing as a strong baseline and adds boundary-discontinuity stress.

