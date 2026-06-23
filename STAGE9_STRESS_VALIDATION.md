# Stage 9 Stress Validation

## Status

Stress-only validation completed for `stage9_c2_random_separation`.

The stress table is separate from clean ETT and is labeled stress-only. It uses Stage 7 real prediction files as the base and injects deterministic perturbations into both `val` and `test` predictions:

- `trend_drift`
- `high_frequency_perturbation`

Validation split remains the only calibration split. Test split remains final evaluation only.

Output:

- `experiments/halluguard/results/stage9_mechanism_separation/stress_table/stage9_c2_random_separation/`

## Stress Table Completion

- Completed stress configs: `32 / 32`
- Variant rows: `224`
- Test threshold leakage: `False`
- Stress types: `trend_drift`, `high_frequency_perturbation`

## Overall Stress Result

- `trend_frequency` mean MSE delta: `-0.104201%`
- improved stress configs: `31 / 32`
- rule beats random configs: `19 / 32`
- paired rule win rate: `0.58125`
- matched smoothing mean MSE delta: `-0.276547%`
- naive smoothing mean MSE delta: `-1.910448%`
- max MSE harm: `0.000000%`
- max MAE harm: `0.000000%`

## By Stress Type

### `high_frequency_perturbation`

| Variant | Mean MSE Delta | Improved Configs | Mean Correction Rate |
| --- | ---: | ---: | ---: |
| `trend_only` | `-0.007931%` | `6 / 16` | `0.068359` |
| `frequency_only` | `-0.122335%` | `15 / 16` | `0.064941` |
| `trend_frequency` | `-0.130268%` | `15 / 16` | `0.126099` |
| `random_trigger` | `-0.116393%` | `15 / 16` | `0.119019` |
| `matched_smoothing_control` | `-0.317690%` | `15 / 16` | `0.126343` |
| `naive_smoothing` | `-2.436927%` | `16 / 16` | `1.000000` |

Mechanism read:

- `frequency_only` beats random in `9 / 16` configs.
- Mean MSE advantage of `frequency_only` over random is `0.000381`.
- `trend_frequency` beats random in `10 / 16` configs.

This is a weak positive frequency-stress signal, but not a decisive separation.

### `trend_drift`

| Variant | Mean MSE Delta | Improved Configs | Mean Correction Rate |
| --- | ---: | ---: | ---: |
| `trend_only` | `-0.015124%` | `12 / 16` | `0.108765` |
| `frequency_only` | `-0.063021%` | `16 / 16` | `0.083374` |
| `trend_frequency` | `-0.078133%` | `16 / 16` | `0.183105` |
| `random_trigger` | `-0.074740%` | `16 / 16` | `0.179688` |
| `matched_smoothing_control` | `-0.235405%` | `16 / 16` | `0.183228` |
| `naive_smoothing` | `-1.383969%` | `16 / 16` | `1.000000` |

Mechanism read:

- `trend_only` beats random in only `1 / 16` configs.
- Mean MSE advantage of `trend_only` over random is `-0.003845`, meaning random is better on average.
- `trend_frequency` beats random in `9 / 16` configs.

Trend-stress validation failed. The current trend correction and trigger do not isolate injected trend drift better than matched random.

## Stress Verdict

Stress validation is mixed and does not pass the full Stage 9 stress gate.

Passed:

- full stress-only table completed
- no test leakage
- no MSE/MAE harm above 3%
- high-frequency perturbation shows a weak frequency-aware signal

Failed:

- trend-stress rule separation failed
- frequency-stress rule separation is not decisive
- matched smoothing control remains stronger than HalluGuard

## Pivot Conclusion

The Stage 9 evidence supports a pivot to `HalluGuard as diagnostic/stress robustness layer`, not a clean point-error correction claim. Future work should focus on making the trigger identify specific dynamics violations rather than exploiting broadly beneficial sparse smoothing.
