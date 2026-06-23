# Stage 11 HalluGuard-Dynamics Big Table Report

## Scope

Stage 11 promotes the Stage 10 best candidate, `s10_c2b_dynamics_anti_smoothing`, into a reusable method line named `HalluGuard-Dynamics`.

Core claim under test: a validation-calibrated local dynamics-continuity correction can detect and repair boundary, first-difference, and curvature discontinuities with evidence beyond random triggering and matched smoothing controls.

This report uses only validation split policy fitting and test split final evaluation. The Stage 11 runner records `test_threshold_leakage=False` for completed rows.

## Commands

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage11_dynamics.py --scope smoke --config experiments/halluguard/configs/halluguard_stage11_dynamics.yaml
D:\miniconda3\python.exe experiments\halluguard\run_stage11_dynamics.py --scope clean_full --config experiments/halluguard/configs/halluguard_stage11_dynamics.yaml
D:\miniconda3\python.exe experiments\halluguard\run_stage11_dynamics.py --scope stress --config experiments/halluguard/configs/halluguard_stage11_dynamics.yaml
```

## Output Paths

- Clean table: `experiments/halluguard/results/stage11_dynamics/clean_full_table/s11_halluguard_dynamics/`
- Stress table: `experiments/halluguard/results/stage11_dynamics/stress_table/s11_halluguard_dynamics/`
- Candidate ledger: `experiments/halluguard/results/stage11_dynamics/candidate_ledger.csv`
- Diagnostics: `experiments/halluguard/results/stage11_dynamics/diagnostics/`

## What Changed From Stage 10

- Added reusable API in `experiments/halluguard/halluguard_dynamics.py`:
  - `fit_policy(validation_samples, config, variant=None)`
  - `score_sample(context, prediction, policy=None)`
  - `apply_correction(context, prediction, policy, force_trigger=None, strength_scale=1.0)`
  - `evaluate_table(validation_samples, evaluation_samples, config, variants=None, seed=23)`
- Added Stage 11 runner and config:
  - `experiments/halluguard/run_stage11_dynamics.py`
  - `experiments/halluguard/configs/halluguard_stage11_dynamics.yaml`
- Added component ablations for boundary, first-difference, and curvature triggers.
- Added controls: random matched trigger, shuffled-score correction, matched smoothing, naive smoothing, EMA smoothing, median smoothing, and Stage 9 incumbent.
- Added stronger stress types: `boundary_discontinuity`, `variance_shift`, `slope_break`, and `delayed_level_shift`, in addition to `trend_drift` and `high_frequency_perturbation`.

The method itself is intentionally close to the Stage 10 parent line. Stage 11 is therefore best read as an API hardening, ablation, and external-readiness pass, not a new optimization search.

## Clean Full Table

Completed configs: 16/16.

Main `dynamics_full` result:

- Mean MSE delta vs no correction: `-0.623359%`
- Mean MAE delta vs no correction: `-0.528044%`
- Improved configs: `15/16`
- Beats random trigger: `15/16`
- Paired rule-vs-random win rate: `0.9375`
- Beats matched smoothing control: `12/16`
- Max MSE harm: `0.067576%`
- Max MAE harm: `0.040376%`
- Test threshold leakage: `False`
- Gate verdict: `clean_gate_pass`

Stage 11 clean table passes all required clean/base gates.

## Clean Ablation Summary

| Variant | Mean MSE Delta | Improved | Beats Random | Paired Win | Beats Matched | Max MSE Harm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dynamics_full` | `-0.623359%` | 15 | 15 | 0.9375 | 12 | `0.067576%` |
| `boundary_only` | `-0.654199%` | 15 | 15 | 0.9375 | 12 | `0.147763%` |
| `boundary_first_diff` | `-0.665592%` | 15 | 14 | 0.9000 | 12 | `0.067576%` |
| `boundary_curvature` | `-0.658703%` | 15 | 15 | 0.9375 | 12 | `0.082955%` |
| `first_diff_only` | `-0.024947%` | 14 | 16 | 0.9125 | 1 | `0.006048%` |
| `curvature_only` | `0.000040%` | 8 | 7 | 0.5000 | 0 | `0.000280%` |
| `random_trigger` | `-0.363539%` | 15 | n/a | n/a | 0 | `0.014444%` |
| `matched_smoothing_control` | `-0.591854%` | 16 | n/a | n/a | n/a | `-0.015877%` |
| `naive_smoothing` | `-1.389826%` | 16 | n/a | n/a | 16 | `-0.355343%` |
| `ema_smoothing` | `-1.366023%` | 16 | n/a | n/a | 16 | `-0.235126%` |
| `median_smoothing` | `-1.886801%` | 16 | n/a | n/a | 16 | `-0.327743%` |
| `stage9_incumbent` | `-0.058038%` | 15 | n/a | n/a | 5 | `0.000000%` |

Mechanism reading:

- The main signal is boundary continuity, not frequency repair.
- `boundary_only`, `boundary_first_diff`, and `boundary_curvature` are all close to or slightly stronger than `dynamics_full` on clean MSE.
- `first_diff_only` has a reliable rule-vs-random signal but tiny point-error gain.
- `curvature_only` has no useful standalone signal.
- `dynamics_full` beats matched smoothing in 12/16 configs, so its gain is not fully explained by sparse smoothing.
- Naive, EMA, and median smoothing remain stronger on pure point MSE. HalluGuard-Dynamics should not be claimed as beating general smoothing on clean ETT MSE.

## Stress Table

Completed stress configs: 96/96 across six stress types.

Overall `dynamics_full` result:

- Mean MSE delta: `-0.643603%`
- Improved configs: `91/96`
- Beats random trigger: `91/96`
- Paired rule-vs-random win rate: `0.945833`
- Beats matched smoothing: `53/96`
- Max MSE harm: `0.165828%`
- Max MAE harm: `0.100100%`
- Test threshold leakage: `False`

Stress-only results are not clean benchmark claims.

| Stress Type | Dynamics Mean MSE Delta | Beats Random | Beats Matched | Max MSE Harm | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `boundary_discontinuity` | `-0.931722%` | 15/16 | 13/16 | `0.165828%` | Direct mechanism stress; passes Stage 11 boundary gate. |
| `trend_drift` | `-0.716750%` | 16/16 | 11/16 | `-0.005559%` | Good rule-vs-random and moderate matched-control separation. |
| `slope_break` | `-0.620156%` | 15/16 | 12/16 | `0.067413%` | Good dynamics-family stress result. |
| `delayed_level_shift` | `-0.602025%` | 15/16 | 12/16 | `0.066457%` | Useful but less direct than boundary discontinuity. |
| `high_frequency_perturbation` | `-0.528242%` | 16/16 | 4/16 | `0.006198%` | Rule beats random, but smoothing explains much of the point-error gain. |
| `variance_shift` | `-0.462721%` | 15/16 | 1/16 | `0.083320%` | Not a good mechanism match; smoothing dominates. |

Boundary stress gate:

- Completed: `16/16`
- Beats random: `15/16`
- Beats matched smoothing: `13/16`
- Mean MSE delta: `-0.931722%`
- Max MSE harm: `0.165828%`

Boundary stress passes all required Stage 11 stress gates.

## Gate Verdict

Stage 11 is successful for the required original 16 clean configs and the required 16 boundary-discontinuity stress configs.

HalluGuard-Dynamics is stronger than the old trend/frequency HalluGuard line and essentially preserves the Stage 10 best signal while making it reusable and externally callable.

## Limitations

- The method does not beat naive, EMA, or median smoothing on clean point MSE. These remain stronger baselines.
- The evidence supports dynamics continuity repair, especially boundary discontinuity repair, not a frequency-repair claim.
- `variance_shift` and `high_frequency_perturbation` are largely explained by smoothing controls.
- Expanded datasets beyond ETTm1/ETTh1 were not run because no local prediction files for ETTm2, ETTh2, Weather, ECL/Electricity, or Traffic were available in the Stage 7 prediction directory. Stage 11 focused on the required original 16-config table and external prediction-file readiness.

## Recommendation

Use HalluGuard-Dynamics as the next parent line for external forecast-table integration and mechanism-facing evidence. For paper-style claims, frame it as a boundary/dynamics-continuity test-time correction with honest smoothing baselines, not as a universal MSE smoother or frequency repair method.
