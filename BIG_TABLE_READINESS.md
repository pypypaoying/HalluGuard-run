# BIG TABLE READINESS

## Gate Status

Ready for a small real-prediction pilot, not a broad benchmark sweep.

Stage 2 passed on three synthetic seeds with the current config:

- `trend_frequency` mean MSE improves over `no_correction`.
- HallucinationRate mean drops by more than 20%.
- Rule trigger beats random trigger in 3 / 3 seeds.
- Real turning-point MSE harm is below 3% but not below the preferred 1% target.
- Clean slice MSE harm is below 1%.
- Threshold sensitivity is not extreme: all 90 / 95 / 99 percentile settings still improve mean MSE.

The remaining caution is turning-point safety. The first real table should stay small and inspect turning-point slices before expanding.

## Required Prediction Format

Real model frameworks should export JSONL or CSV rows with:

```text
sample_id, dataset, model, split, context, prediction, target
```

Rules:

- `split=val` is used only for threshold calibration.
- `split=test` is used only for final evaluation.
- `context`, `prediction`, and `target` are numeric arrays.
- `prediction` and `target` must have the same horizon length.
- HalluGuard does not require model checkpoints, hidden states, dataloaders, or training logs.

Full schema: `experiments/halluguard/EXTERNAL_PREDICTION_SCHEMA.md`.

## First Real Small Table

Recommended first table:

| Dataset | Model | Variants |
| --- | --- | --- |
| ETTm1 or ETTh1 | DLinear or PatchTST exported predictions | no_correction, naive_smoothing, trend_only, frequency_only, trend_frequency, random_trigger |

Do not implement a large training framework in this repository for the first real pass. Use existing prediction exports from an external framework when available.

## Real Evaluation Command

JSONL:

```bash
python experiments/halluguard/evaluate_predictions.py ^
  --config experiments/halluguard/configs/halluguard_mvp.yaml ^
  --input path/to/ettm1_dlinear_predictions.jsonl ^
  --calibration-split val ^
  --split test ^
  --output-dir experiments/halluguard/results/ettm1_dlinear_external
```

CSV:

```bash
python experiments/halluguard/evaluate_predictions.py ^
  --config experiments/halluguard/configs/halluguard_mvp.yaml ^
  --input path/to/ettm1_dlinear_predictions.csv ^
  --calibration-split val ^
  --split test ^
  --output-dir experiments/halluguard/results/ettm1_dlinear_external
```

## Output Table Fields

The first real table should include:

```text
dataset
model
variant
mse
mae
hallucination_rate
trend_violation_rate
freq_violation_rate
spectral_consistency
turning_point_false_correction_rate
correction_rate
inference_latency_ms
threshold_quantile
lambda_trend
lambda_freq
```

## Go / No-Go Checks For Real Pilot

Proceed beyond the first real table only if:

- `trend_frequency` does not worsen MSE/MAE by more than 3%.
- Rule trigger beats random trigger on MSE or on violation reduction without MSE harm.
- Clean or low-error windows are not materially harmed.
- Turning-point or sharp-change windows do not show uncontrolled false correction.
- Thresholds are still calibrated only from validation predictions.

Stop and return to diagnostics if:

- Random trigger is close to rule trigger.
- MSE/MAE worsens by more than 3%.
- Turning-point harm exceeds 3%.
- The method only works on synthetic stress and not on real exported predictions.
