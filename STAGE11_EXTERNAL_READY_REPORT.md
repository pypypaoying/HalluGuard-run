# Stage 11 External-Ready Report

## Status

HalluGuard-Dynamics is now usable as a small reusable implementation layer and as a single-file external prediction evaluator.

External smoke completed with the Stage 11 JSONL fixture:

- Scope: `external_smoke`
- Completed configs: `1/1`
- Main MSE delta: `-0.513137%`
- Beats random: `1/1`
- Beats matched smoothing: `1/1`
- Test threshold leakage: `False`

An explicit `external_eval` scope was also added and run on the same fixture.

## Input Schema

External forecast outputs can be provided as JSONL or CSV. Required fields:

| Field | Type | Notes |
| --- | --- | --- |
| `sample_id` | string | Unique sample id. |
| `dataset` | string | Dataset label used in outputs. |
| `model` | string | Forecast model label used in outputs. |
| `split` | string | Must include `val` and `test`. `val` fits policy; `test` is final evaluation. |
| `context` | list[float] | Historical context window. |
| `prediction` | list[float] | Baseline forecast to correct. |
| `target` | list[float] | Future ground truth for evaluation. |

For CSV, `context`, `prediction`, and `target` should be JSON-encoded arrays in their cells.

The schema is compatible with `experiments/halluguard/EXTERNAL_PREDICTION_SCHEMA.md`.

## Reusable API

Implementation file:

- `experiments/halluguard/halluguard_dynamics.py`

Primary API:

```python
from halluguard_dynamics import fit_policy, score_sample, apply_correction, evaluate_table

policy = fit_policy(validation_samples, config)
score = score_sample(context, prediction, policy)
corrected, info = apply_correction(context, prediction, policy)
result = evaluate_table(validation_samples, test_samples, config)
```

Contract:

- `fit_policy(...)` must only receive validation samples.
- `apply_correction(...)` and `evaluate_table(...)` can be used after the policy is frozen.
- Test targets are used only for final metrics, not policy fitting.

## External Evaluation Command

For a user-provided prediction file:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage11_dynamics.py --scope external_eval --config experiments/halluguard/configs/halluguard_stage11_dynamics.yaml --input path\to\predictions.jsonl
```

The output directory is:

```text
experiments/halluguard/results/stage11_dynamics/external_eval/s11_halluguard_dynamics/
```

It contains:

- `combined_metrics.csv`
- `combined_metrics.json`
- `combined_ablation_table.md`
- `summary.md`
- per-run `metrics.csv`, `metrics.json`, `ablation_table.md`, and diagnostics

## Smoke Fixture

Generated fixture:

```text
experiments/halluguard/results/stage11_dynamics/external_smoke/fixtures/external_predictions_smoke.jsonl
```

Smoke commands run:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage11_dynamics.py --scope external_smoke --config experiments/halluguard/configs/halluguard_stage11_dynamics.yaml
D:\miniconda3\python.exe experiments\halluguard\run_stage11_dynamics.py --scope external_eval --config experiments/halluguard/configs/halluguard_stage11_dynamics.yaml --input experiments/halluguard/results/stage11_dynamics/external_smoke/fixtures/external_predictions_smoke.jsonl
```

External smoke summary:

- `dynamics_full`: MSE delta `-0.513137%`
- `random_trigger`: MSE delta `0.167504%`
- `matched_smoothing_control`: MSE delta `-0.396285%`
- `naive_smoothing`: MSE delta `-6.054983%`
- Leakage: `False`

## How To Plug In A New Forecast Model

1. Export the model's forecasts into JSONL or CSV using the required schema.
2. Include both `split="val"` and `split="test"` rows in the same file.
3. Run `external_eval` with the file path.
4. Read `combined_metrics.csv` for the six controls plus Stage 11 ablations.
5. Use `dynamics_full` as the main HalluGuard-Dynamics row, but keep `naive_smoothing`, `matched_smoothing_control`, `random_trigger`, and `stage9_incumbent` visible in reports.

## Readiness Verdict

External readiness passes for single-file external prediction tables. The method is ready to be connected to another framework's exported forecasts without retraining the forecast model or rewriting the HalluGuard-Dynamics implementation.

Remaining limitation: the Stage 11 runner currently evaluates one external file at a time. Multi-dataset external batch orchestration can be added around `external_eval` or directly around the reusable API.
