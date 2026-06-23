# Stage 9 Mechanism Separation

## Status

Stage 9 completed two mechanism-candidate smokes and promoted the stronger candidate to a clean 16-config full table plus a targeted real-stress table.

Best candidate:

- `stage9_c2_random_separation`
- validation random-separation objective
- variants include `matched_smoothing_control`
- calibration split: `val`
- final evaluation split: `test`
- test threshold leakage: `False`

Outputs:

- `experiments/halluguard/results/stage9_mechanism_separation/candidate_ledger.csv`
- `experiments/halluguard/results/stage9_mechanism_separation/smoke/`
- `experiments/halluguard/results/stage9_mechanism_separation/clean_full_table/stage9_c2_random_separation/`
- `experiments/halluguard/results/stage9_mechanism_separation/stress_table/stage9_c2_random_separation/`
- `experiments/halluguard/results/stage9_mechanism_separation/diagnostics/`

## Candidate Smokes

Smoke configs:

- ETTm1 / DLinear / 192
- ETTm1 / PatchTST / 720
- ETTh1 / DLinear / 336
- ETTh1 / PatchTST / 720

| Candidate | Mean MSE Delta | Improved | Rule Beats Random | Paired Win Rate | Matched Smoothing Delta | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `stage9_c1_freq_repair` | `-0.025165%` | `4 / 4` | `2 / 4` | `0.500` | `-0.049431%` | weaker |
| `stage9_c2_random_separation` | `-0.041995%` | `4 / 4` | `3 / 4` | `0.700` | `-0.177790%` | promoted |

`stage9_c2_random_separation` was promoted because it had stronger smoke rule-vs-random separation and better MSE while preserving no-leakage.

## Clean Full Table

Clean full table completed `16 / 16` configs and `112` variant rows.

| Variant | Mean MSE Delta | Improved Configs | Mean Correction Rate |
| --- | ---: | ---: | ---: |
| `trend_only` | `-0.004041%` | `6 / 16` | `0.058350` |
| `frequency_only` | `-0.053997%` | `15 / 16` | `0.076782` |
| `trend_frequency` | `-0.058038%` | `15 / 16` | `0.131592` |
| `random_trigger` | `-0.053277%` | `15 / 16` | `0.122681` |
| `matched_smoothing_control` | `-0.159881%` | `15 / 16` | `0.131836` |
| `naive_smoothing` | `-1.389826%` | `16 / 16` | `1.000000` |

Clean gate:

- MSE target: passed. `trend_frequency` stayed at `-0.058038%`, better than the `<= -0.05%` target.
- Safety: passed. Max MSE harm and max MAE harm were both `0.000000%`.
- Leakage: passed. `test_threshold_leakage=False`.
- Rule-vs-random: failed. Rule beat random in only `10 / 16`, paired win rate was `0.525`.
- Anti-smoothing: failed. `matched_smoothing_control` was stronger than HalluGuard on MSE.

## Mechanism Verdict

Stage 9 did not separate clean ETT HalluGuard triggers from matched random or matched smoothing controls.

The best candidate preserved Stage 8's MSE repair but did not improve clean trigger specificity. The matched smoothing control being stronger than `trend_frequency` is the key negative evidence: on clean ETT, sparse smoothing itself explains more of the point-error gain than the current trend/frequency trigger.

## Recommendation

Pivot the clean-table claim. HalluGuard should not currently be presented as a strong clean ETT point-error correction method. The defensible direction is narrower:

- HalluGuard is safe under validation-only calibration.
- HalluGuard can be useful as a diagnostic and stress-robustness layer.
- More method work is needed before claiming mechanism-separated clean forecast correction.
