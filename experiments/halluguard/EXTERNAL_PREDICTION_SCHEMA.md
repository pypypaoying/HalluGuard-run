# External Prediction Schema

HalluGuard-Dynamics evaluates exported forecast tables without importing,
training, or modifying the external forecasting framework. The external
framework only needs to export context, prediction, and target arrays.

## Split Contract

- `split="val"` is used only to fit the HalluGuard-Dynamics policy.
- `split="test"` is used only for final evaluation.
- Every evaluated file or dataset/model/horizon group must contain both `val`
  and `test` rows.
- Thresholds, trigger quantiles, and correction strengths must never be tuned on
  the test split.
- When this contract is respected, outputs record `test_threshold_leakage=False`.

## Required Fields

Each sample must contain:

| field | type | description |
| --- | --- | --- |
| `sample_id` | string | Stable unique sample id. |
| `dataset` | string | Dataset name, for example `ETTm1`. |
| `model` | string | Forecast model name, for example `DLinear` or `PatchTST`. |
| `split` | string | `val` for policy fitting or `test` for final evaluation. |
| `context` | list of float | Historical input window. |
| `prediction` | list of float | External model forecast for the future horizon. |
| `target` | list of float | Ground-truth future horizon. |

`prediction` and `target` must have identical lengths. `context` may have a
different length. Optional metadata such as `horizon`, `source_repo`,
`checkpoint`, or `stress_type` is allowed.

## JSONL Format

Use one JSON object per line:

```json
{"sample_id":"val_0000","dataset":"ETTm1","model":"DLinear","split":"val","context":[1.0,1.1],"prediction":[1.2],"target":[1.22]}
{"sample_id":"test_0000","dataset":"ETTm1","model":"DLinear","split":"test","context":[1.1,1.2],"prediction":[1.3],"target":[1.28]}
```

## CSV Format

Use the same required columns. The `context`, `prediction`, and `target` cells
must be JSON arrays:

```csv
sample_id,dataset,model,split,context,prediction,target
val_0000,ETTm1,DLinear,val,"[1.0,1.1]","[1.2]","[1.22]"
test_0000,ETTm1,DLinear,test,"[1.1,1.2]","[1.3]","[1.28]"
```

## One-File Evaluation

Stage 11 single-file command:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage11_dynamics.py --scope external_eval --config experiments/halluguard/configs/halluguard_stage11_dynamics.yaml --input path\to\predictions.jsonl
```

Stage 12 batch runner can also evaluate one file:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage12_external_batch.py --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml --input path\to\predictions.jsonl --output-dir experiments/halluguard/results/my_external_eval
```

## Batch Evaluation

Evaluate every JSONL/CSV file in a directory:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage12_external_batch.py --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml --input-dir path\to\prediction_dir --output-dir experiments/halluguard/results/my_external_batch
```

Stage 13 adaptive router external-batch integration command:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage13_adaptive_router.py --scope external_batch --config experiments\halluguard\configs\halluguard_stage13_adaptive_router.yaml --external-input-dir path\to\prediction_dir --output-root experiments\halluguard\results\my_stage13_router_batch
```

The Stage 13 runner evaluates the same JSONL/CSV schema and split contract,
but reports the adaptive `rule_router` alongside `boundary_only`,
`dynamics_full`, full smoothing baselines, matched sparse smoothing,
random-trigger/action controls, shuffled-feature control, and the Stage 9
incumbent. The compact Stage 12 fixture directory can be used as an integration
smoke, but claim-level evidence should use full prediction tables.

Evaluate files listed in a manifest:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage12_external_batch.py --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml --manifest path\to\manifest.csv --output-dir experiments/halluguard/results/my_external_batch
```

Manifest formats:

- `.txt`: one prediction path per line.
- `.csv`: a `path`, `prediction_path`, `input_path`, or `file` column.
- `.jsonl`: one object per line with a `path`, `prediction_path`,
  `input_path`, or `file` field.

## Output Directory Structure

For Stage 12 batch evaluation:

```text
<output-dir>/
  batch_eval/
    s12_halluguard_dynamics_batch/
      batch_metrics.csv
      batch_metrics.json
      batch_variant_summary.csv
      batch_files.csv
      batch_configs.csv
      batch_report.md
      files/
        <input_file_id>/
          runs/
            <dataset>_<model>_<horizon>/
              metrics.csv
              metrics.json
              ablation_table.md
              summary.md
              diagnostics/
  diagnostics/
    batch_s12_halluguard_dynamics_batch_*.csv
  batch_ledger.csv
```

If one external file contains multiple dataset/model/horizon groups, Stage 12
fits and evaluates each group separately.

## Interpreting Variants And Baselines

Stage 12 reports the recommended HalluGuard-Dynamics variant plus ablations:

- `boundary_only`: the Stage 12 recommended conservative dynamics variant.
- `dynamics_full`: boundary + first-difference + curvature scoring with
  boundary + first-difference repair; kept as a richer ablation.
- `boundary_first_diff`, `boundary_curvature`, `first_diff_only`,
  `curvature_only`: mechanism ablations.
- `random_trigger`: matched-count random triggering with the same correction
  vector as the main variant.
- `shuffled_score_correction`: correction after shuffling the trigger scores.
- `matched_smoothing_control`: sparse smoothing on the same triggered windows
  as the main variant.
- `naive_smoothing`, `ema_smoothing`, `median_smoothing`: full smoothing
  baselines applied to all predictions.
- `stage9_incumbent`: the old trend/frequency HalluGuard line, retained only as
  a baseline.

Current evidence supports boundary/dynamics-continuity repair, not frequency
repair. Naive/EMA/median smoothing can be stronger on pure point MSE and should
not be hidden; HalluGuard-Dynamics is mainly justified when its rule-triggered
boundary repair beats matched random and matched sparse smoothing controls.
