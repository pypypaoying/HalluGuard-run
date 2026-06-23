# Stage 8 Real-Signal Repair

## Status

Stage 8 candidate 1 completed smoke and full repaired-table evaluation.

Candidate:

- `candidate1_validation_calibrated_margin`
- validation-calibrated trigger margin and strength
- calibration split: `val`
- final evaluation split: `test`
- test threshold leakage: `False`

Outputs:

- `experiments/halluguard/results/stage8_real_signal_repair/candidate_ledger.csv`
- `experiments/halluguard/results/stage8_real_signal_repair/diagnostics/`
- `experiments/halluguard/results/stage8_real_signal_repair/smoke/candidate1_validation_calibrated_margin/`
- `experiments/halluguard/results/stage8_real_signal_repair/full_table/candidate1_validation_calibrated_margin/`

## Candidate 1 Mechanism

The repaired method uses validation samples only to select:

- trigger quantile
- trend trigger margin
- frequency trigger margin
- trend correction strength
- frequency correction strength

The selected policy is then frozen for test evaluation. No test target is used for threshold, lambda, trigger rule, or candidate selection.

## Smoke Result

Smoke configs:

- ETTm1 / DLinear / 192
- ETTm1 / PatchTST / 720
- ETTh1 / DLinear / 336
- ETTh1 / PatchTST / 720

Smoke completed `4 / 4` configs.

Key smoke metrics:

- `trend_frequency` mean MSE delta: `-0.041995%`
- improved configs: `4 / 4`
- rule beats random configs: `3 / 4`
- max MSE harm: `-0.026355%`
- max MAE harm: `-0.009630%`
- naive smoothing mean MSE delta: `-1.130033%`
- test threshold leakage: `False`

Smoke was strong enough to run the full 16-config table.

## Full Table Result

Full table completed `16 / 16` configs and `96` variant rows.

Key full-table metrics:

- `trend_frequency` mean MSE delta: `-0.058038%`
- Stage 7 `trend_frequency` mean MSE delta: `-0.0060%`
- improved configs: `15 / 16`
- rule beats random configs: `10 / 16`
- paired rule-vs-random win rate: `0.525`
- rule-vs-random mean MSE advantage: `0.000139822`
- max MSE harm: `0.000000%`
- max MAE harm: `0.000000%`
- naive smoothing mean MSE delta: `-1.389826%`
- test threshold leakage: `False`

## Gate Verdict

Candidate 1 is a partial Stage 8 success and a trigger-sharpness failure.

Passed:

- full 16-config repaired table completed
- mean MSE delta improved clearly over Stage 7
- target mean MSE delta `<= -0.05%` was reached
- `15 / 16` configs improved MSE
- no MSE or MAE harm above 3%
- validation-only calibration preserved
- naive smoothing reported side by side

Failed:

- rule trigger beats random in only `10 / 16` configs, below the `14 / 16` gate
- paired random comparison is close: rule wins only `52.5%` of random-seed comparisons
- naive smoothing remains much stronger on MSE

## Diagnostics

Validation trigger precision proxy:

- combined trigger rate: `0.2434`, triggered MSE delta: `-0.4391%`
- frequency trigger rate: `0.2012`, triggered MSE delta: `-0.4840%`
- trend trigger rate: `0.0458`, triggered MSE delta: `-0.1010%`

Test error-conditioned analysis:

- error quantile `0.00-0.25`: `-0.2080%`
- error quantile `0.25-0.50`: `-0.1051%`
- error quantile `0.50-0.75`: `-0.0545%`
- error quantile `0.75-0.90`: `-0.0349%`
- error quantile `0.90-1.00`: `-0.0396%`

This does not show the desired high-error concentration. The method improves clean low-error windows more than the highest-error windows.

Score-bin analysis:

- frequency score is useful: top `0.95-1.00` bin has mean MSE delta `-0.3256%`
- trend score is weak: top `0.95-1.00` bin has mean MSE delta `-0.0408%`

Component contribution:

- `trend_only`: `-0.004041%`
- `frequency_only`: `-0.053997%`
- `trend_frequency`: `-0.058038%`
- `random_trigger`: `-0.053277%`

The gain is dominated by frequency correction. Because random trigger is nearly as strong, the current repaired method is closer to validation-selected smoothing than a sharp hallucination detector.

## Conclusion

Stage 8 candidate 1 repairs the clean ETT MSE weakness but does not repair trigger sharpness. HalluGuard now has a stronger safe real-prediction correction signal than Stage 7, but the mechanism claim remains weak because matched random triggers stay close.

Recommended next route: do not scale this directly as paper-style evidence. The next candidate should attack trigger specificity, not MSE alone. The strongest local direction is a frequency repair or risk gate that separates true high-frequency excess from broadly beneficial smoothing, plus a targeted stress-only table to show mechanism validity under real prediction perturbations.
