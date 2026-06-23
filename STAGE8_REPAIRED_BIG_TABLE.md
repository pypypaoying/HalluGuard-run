# Stage 8 Repaired Big Table

## Completion

Repaired Big Table completed all `16 / 16` real prediction configurations.

No configurations were blocked.

Required outputs:

- `experiments/halluguard/results/stage8_real_signal_repair/full_table/candidate1_validation_calibrated_margin/combined_metrics.csv`
- `experiments/halluguard/results/stage8_real_signal_repair/full_table/candidate1_validation_calibrated_margin/combined_metrics.json`
- `experiments/halluguard/results/stage8_real_signal_repair/full_table/candidate1_validation_calibrated_margin/combined_ablation_table.md`
- `experiments/halluguard/results/stage8_real_signal_repair/full_table/candidate1_validation_calibrated_margin/summary.md`

Diagnostics:

- `experiments/halluguard/results/stage8_real_signal_repair/diagnostics/full_candidate1_validation_calibrated_margin_trigger_precision_proxy.csv`
- `experiments/halluguard/results/stage8_real_signal_repair/diagnostics/full_candidate1_validation_calibrated_margin_error_conditioned_analysis.csv`
- `experiments/halluguard/results/stage8_real_signal_repair/diagnostics/full_candidate1_validation_calibrated_margin_rule_vs_random_paired.csv`
- `experiments/halluguard/results/stage8_real_signal_repair/diagnostics/full_candidate1_validation_calibrated_margin_score_bin_table.csv`
- `experiments/halluguard/results/stage8_real_signal_repair/diagnostics/full_candidate1_validation_calibrated_margin_naive_smoothing_comparison.csv`

## Headline

`trend_frequency` improved from the Stage 7 mean MSE delta of `-0.0060%` to `-0.058038%`.

This passes the Stage 8 MSE target but fails the trigger-sharpness target.

## Full Table Metrics

| Variant | Mean MSE Delta vs No Correction | Improved Configs |
| --- | ---: | ---: |
| `trend_only` | `-0.004041%` | `6 / 16` |
| `frequency_only` | `-0.053997%` | `15 / 16` |
| `trend_frequency` | `-0.058038%` | `15 / 16` |
| `random_trigger` | `-0.053277%` | `15 / 16` |
| `naive_smoothing` | `-1.389826%` | `16 / 16` |

Safety:

- max `trend_frequency` MSE harm: `0.000000%`
- max `trend_frequency` MAE harm: `0.000000%`
- test threshold leakage: `False`

Rule-vs-random:

- rule beats random configs: `10 / 16`
- paired random win rate: `0.525`
- mean rule-vs-random MSE advantage: `0.000139822`

## Interpretation

The repair successfully finds a validation-supported correction strength, mostly through frequency correction. It does not make the rule trigger decisively better than matched random trigger.

The diagnostic evidence says:

- frequency score bins are meaningful, especially the top score bins
- trend score contributes little
- error-conditioned gains are not concentrated in the highest no-correction error group
- naive smoothing remains the dominant point-error baseline

## Stage 8 Gate

Verdict: `partial_success_fail_rule_random`.

The repaired table is valid and safe, but it does not satisfy the trigger-sharpness gate. This is not yet paper-level evidence for HalluGuard as a clean ETT point-error correction method.

The result supports a narrower claim: validation-calibrated HalluGuard can safely act as a conservative frequency-oriented correction layer, but the current trigger does not yet isolate hallucination-like failures better than random matched smoothing.
