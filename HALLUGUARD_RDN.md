# HalluGuard-RDN: Reversible Dynamics Normalization

HalluGuard-RDN is a first runnable version of the input-layer idea discussed
after the RevIN/NST comparison: keep the RevIN placement pattern, but replace
mean/std normalization with a HalluGuard-style local dynamics baseline.

## Method

For every sample, the normalizer uses only the input context:

```text
context -> fit local level/slope/scale -> residual input -> backbone -> residual forecast -> inverse dynamics transform
```

Default variant:

```text
level_slope_scale
```

For a context window, the script fits a local OLS slope on the tail of the
context, shrinks it conservatively, estimates residual scale, and extrapolates
the baseline into the forecast horizon. The training target is transformed by
the same context-only baseline, so validation/test targets are not used to fit
the normalizer.

This is not a post-hoc smoothing method. It changes the representation seen by
the forecasting backbone and then reverses the transform on the output.

## One-command run

```bash
bash scripts/run_halluguard_rdn_table.sh
```

Useful server overrides:

```bash
DEVICE=cuda EPOCHS=10 MAX_TRAIN_WINDOWS=8192 MAX_EVAL_WINDOWS=1024 \
  bash scripts/run_halluguard_rdn_table.sh
```

Fast smoke:

```bash
DATASET_SET=ETTm1 MODELS=DLinear HORIZONS=96 RDN_VARIANTS=level_slope_scale \
EPOCHS=1 MAX_TRAIN_WINDOWS=128 MAX_EVAL_WINDOWS=16 \
  bash scripts/run_halluguard_rdn_table.sh
```

Run all built-in ablations:

```bash
RDN_VARIANTS=level_only,level_scale,level_slope,level_slope_scale \
  bash scripts/run_halluguard_rdn_table.sh
```

## Outputs

Prediction files:

```text
baseline_predictions/halluguard_rdn/*.jsonl
baseline_predictions/halluguard_rdn_raw/*.jsonl
```

Metrics and summary:

```text
experiments/halluguard/results/halluguard_rdn/rdn_metrics.csv
experiments/halluguard/results/halluguard_rdn/rdn_summary.csv
experiments/halluguard/results/halluguard_rdn/rdn_metrics.json
experiments/halluguard/results/halluguard_rdn/summary.md
```

Every JSONL row follows the external prediction schema:

```text
sample_id, dataset, model, split, context, prediction, target
```

Additional metadata fields include `backbone`, `method`, `variant`,
`adapter_mode=reversible_dynamics_normalization`, and
`test_threshold_leakage=false`.

The metrics CSV includes `raw_no_correction` rows and
`mse_delta_pct_vs_raw` / `mae_delta_pct_vs_raw` for HalluGuard-RDN rows, so a
single run can verify whether the reversible dynamics wrapper improved or
harmed the same backbone configuration.

## Claim Boundary

This script verifies whether the reversible input-layer dynamics idea is
executable and comparable under the existing ETT/DLinear/PatchTST contract. It
does not claim to beat RevIN, NST, smoothing, or HalluGuard-SP unless the
generated metrics show that on the same run settings.
