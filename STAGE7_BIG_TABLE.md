# STAGE7 BIG TABLE V1

## Completion

Big Table v1 completed all `16 / 16` real prediction configurations.

No configurations were blocked.

Required files:

- `experiments/halluguard/results/stage7_big_table/combined_metrics.csv`
- `experiments/halluguard/results/stage7_big_table/combined_metrics.json`
- `experiments/halluguard/results/stage7_big_table/combined_ablation_table.md`
- `experiments/halluguard/results/stage7_big_table/summary.md`

Each completed configuration has:

- an independent prediction file under `experiments/halluguard/results/stage7_big_table/predictions/`
- an independent HalluGuard output directory under `experiments/halluguard/results/stage7_big_table/runs/`
- validation-only threshold calibration with `test_threshold_leakage=False`
- all six variants: `no_correction`, `naive_smoothing`, `trend_only`, `frequency_only`, `trend_frequency`, `random_trigger`

## Experimental Contract

Datasets:

- `ETTm1`
- `ETTh1`

Models:

- `DLinear`
- `PatchTST`

Horizons:

- `96`
- `192`
- `336`
- `720`

Prediction pipeline:

- Public ETT `OT` column.
- Train split used for model fitting and normalization statistics.
- Validation split used only for HalluGuard threshold calibration.
- Test split used only for final HalluGuard evaluation.
- `512` validation windows and `512` test windows exported per configuration.

## Headline Result

`trend_frequency` improved MSE versus `no_correction` in `11 / 16` configurations.

It never worsened MSE by more than `3%`; the largest observed MSE delta was `+0.0091%`.

Average MSE delta versus `no_correction`:

| Variant | Mean MSE delta | Improved configs |
| --- | ---: | ---: |
| `trend_only` | `-0.0035%` | `8 / 16` |
| `frequency_only` | `-0.0025%` | `9 / 16` |
| `trend_frequency` | `-0.0060%` | `11 / 16` |
| `random_trigger` | `-0.0030%` | `9 / 16` |
| `naive_smoothing` | `-1.3898%` | `16 / 16` |

## Per-Configuration `trend_frequency`

| Dataset | Model | Horizon | MSE delta vs no_correction | HallucinationRate |
| --- | --- | ---: | ---: | ---: |
| ETTh1 | DLinear | 96 | `-0.0300%` | `0.0605` |
| ETTh1 | DLinear | 192 | `0.0000%` | `0.0449` |
| ETTh1 | DLinear | 336 | `+0.0091%` | `0.0312` |
| ETTh1 | DLinear | 720 | `-0.0140%` | `0.0234` |
| ETTh1 | PatchTST | 96 | `-0.0120%` | `0.0566` |
| ETTh1 | PatchTST | 192 | `-0.0041%` | `0.0312` |
| ETTh1 | PatchTST | 336 | `-0.0023%` | `0.0254` |
| ETTh1 | PatchTST | 720 | `-0.0097%` | `0.0156` |
| ETTm1 | DLinear | 96 | `-0.0016%` | `0.0742` |
| ETTm1 | DLinear | 192 | `-0.0215%` | `0.1328` |
| ETTm1 | DLinear | 336 | `-0.0069%` | `0.0664` |
| ETTm1 | DLinear | 720 | `+0.0031%` | `0.0312` |
| ETTm1 | PatchTST | 96 | `-0.0073%` | `0.0234` |
| ETTm1 | PatchTST | 192 | `+0.0004%` | `0.0215` |
| ETTm1 | PatchTST | 336 | `-0.0014%` | `0.0234` |
| ETTm1 | PatchTST | 720 | `+0.0024%` | `0.0430` |

## Mechanism Read

The real-prediction signal exists but is weak.

Compared with the synthetic stress benchmark, real ETT predictions trigger few HalluGuard violations. The average post-correction violation rates for `trend_frequency` are:

- TrendViolationRate: `0.0396`
- FreqViolationRate: `0.0045`

This means the Big Table v1 signal is mostly trend-triggered rather than frequency-triggered, unlike the synthetic stress result where frequency correction dominated.

`trend_frequency` is slightly better than either component alone on average, but the absolute gains are tiny:

- `trend_only`: `-0.0035%` mean MSE delta
- `frequency_only`: `-0.0025%` mean MSE delta
- `trend_frequency`: `-0.0060%` mean MSE delta

## Rule Trigger Versus Random Trigger

Rule trigger has a small edge but not a decisive one:

- `trend_frequency` MSE is lower than `random_trigger` in `13 / 16` configurations.
- The average MSE advantage of rule over random is only about `0.000254` absolute MSE.
- The combined summary flags random as near rule by tight MSE tolerance in `16 / 16` configurations.

This weakens the mechanism claim. It suggests the current real ETT setting has too few strong trend/frequency violations for HalluGuard to clearly separate itself from a trigger-frequency control.

## Dataset / Model / Horizon Pattern

Average `trend_frequency` MSE delta:

- ETTm1: `-0.0041%`, improved `5 / 8`
- ETTh1: `-0.0079%`, improved `6 / 8`
- DLinear: `-0.0077%`, improved `5 / 8`
- PatchTST: `-0.0042%`, improved `6 / 8`

No dataset/model/horizon group shows harmful behavior above the stop threshold. ETTh1 and DLinear have slightly larger average gains, but still small.

## Blocked Configurations

None.

## Conclusion

HalluGuard is safe on this Big Table v1: no completed real configuration shows meaningful MSE/MAE degradation, and validation-only threshold calibration works across all 16 configurations.

However, this is not yet strong paper-style evidence for the proposed mechanism. The real ETT gains are very small, naive smoothing is much stronger on MSE, and random trigger is close to rule trigger by tight tolerance. The current result supports HalluGuard as a safe diagnostic/evaluation layer, but not yet as a compelling correction method on standard ETT predictions.

Recommended next step: do not expand directly to a larger paper table. First run a method-repair round focused on real-prediction trigger sharpness, or add targeted real stress slices / harder exported forecasts where trend/frequency violations are more prevalent.
