# Core Table Plan

## Objective

Freeze the current HalluGuard method line and prepare a reproducible core-table
evaluation harness for same-position test-time/adaptation baselines. This repo
must not continue Stage 14 autosearch or tune on test targets.

## Methods

The fixed method set is:

1. raw_no_correction
2. HalluGuard-SP frozen
3. HalluGuard stable-harm ablation
4. matched_sparse_smoothing
5. naive_smoothing
6. ema_smoothing
7. median_smoothing
8. RevIN
9. Dish-TS
10. SAN
11. Non-stationary Transformer / NST
12. TAFAS

## Route

- Use local frozen HalluGuard configs for the two HalluGuard lines.
- Use local HalluGuard controls for smoothing baselines.
- Fetch official baseline repos under `external/plugin_baselines/`.
- Fetch public long-term forecasting datasets through the THUML
  Time-Series-Library Hugging Face dataset.
- Require every external method to export
  `sample_id,dataset,model,split,context,prediction,target`.

## Acceptance

- Dependencies, dataset setup, official repo fetch scripts, method manifest, and
  run guide are committed and pushed.
- No large generated outputs, cloned third-party repos, or downloaded datasets
  are committed.
- The documented protocol keeps validation calibration and test evaluation
  separate.
