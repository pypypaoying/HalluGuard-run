# Stage 12 External Batch Report

## Scope

Stage 12 upgraded HalluGuard-Dynamics from a single-file external evaluator into a batch evaluation framework for external forecast tables.

The batch runner supports:

- one JSONL/CSV file via `--input`
- a directory of JSONL/CSV files via `--input-dir`
- a TXT/CSV/JSONL manifest via `--manifest`

The runner groups samples by `dataset`, `model`, and prediction horizon. For each group, `split="val"` is used only for policy fitting and `split="test"` is used only for final evaluation.

## Implementation

New files:

- `experiments/halluguard/run_stage12_external_batch.py`
- `experiments/halluguard/configs/halluguard_stage12_external_batch.yaml`

Updated documentation:

- `experiments/halluguard/EXTERNAL_PREDICTION_SCHEMA.md`

The Stage 12 config sets `boundary_only` as the main external variant and keeps `dynamics_full`, component ablations, random controls, matched sparse smoothing, full smoothing baselines, and the Stage 9 incumbent in the output table.

## Fixture Set

Created external-style fixtures from existing Stage 7 local predictions:

```text
experiments/halluguard/results/stage12_external_batch/fixtures/external_batch_clean/
```

Fixture coverage:

- Datasets: `ETTm1`, `ETTh1`
- Models: `DLinear`, `PatchTST`
- Horizons: `96`, `192`, `336`, `720`
- Files: `16`
- Rows per file: `32 val` + `32 test`

The fixture files are intentionally compact external-style smoke fixtures, not a replacement for the Stage 11 full clean table.

## Commands Run

Create fixtures and run the directory batch evaluation:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage12_external_batch.py --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml --make-fixtures
```

The first run completed all 16 prediction files but then tried to parse `manifest.csv` as a prediction CSV. The runner was fixed to skip manifest files during directory scans.

Final directory batch evaluation:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage12_external_batch.py --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml --input-dir experiments/halluguard/results/stage12_external_batch/fixtures/external_batch_clean
```

Single-file smoke:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage12_external_batch.py --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml --input experiments/halluguard/results/stage12_external_batch/fixtures/external_batch_clean/ETTm1_DLinear_192.jsonl --output-dir experiments/halluguard/results/stage12_external_batch/single_file_smoke
```

Manifest smoke:

```powershell
D:\miniconda3\python.exe experiments\halluguard\run_stage12_external_batch.py --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml --manifest experiments/halluguard/results/stage12_external_batch/fixtures/external_batch_clean/manifest.csv --limit-files 2 --output-dir experiments/halluguard/results/stage12_external_batch/manifest_smoke
```

## Output Paths

Main batch output:

```text
experiments/halluguard/results/stage12_external_batch/batch_eval/s12_halluguard_dynamics_batch/
```

Key files:

- `batch_metrics.csv`
- `batch_metrics.json`
- `batch_variant_summary.csv`
- `batch_files.csv`
- `batch_configs.csv`
- `batch_report.md`
- per-file run directories under `files/`

Diagnostics:

```text
experiments/halluguard/results/stage12_external_batch/diagnostics/
```

## Batch Result

Directory batch evaluation completed:

- Completed files: `16/16`
- Completed dataset/model/horizon groups: `16/16`
- Failed files: `0`
- Test threshold leakage: `False`

Main variant in this Stage 12 batch run: `boundary_only`.

Fixture-batch headline:

- Mean MSE delta: `-0.464343%`
- Mean MAE delta: `-0.379894%`
- Improved configs: `13/16`
- Beats random configs: `12/16`
- Paired rule-vs-random win rate: `0.7875`
- Beats matched sparse smoothing: `10/16`
- Max MSE harm: `0.034210%`
- Max MAE harm: `0.033612%`

These are fixture-smoke numbers from a compact 32-val/32-test subset per config. The scientific method-selection conclusion uses the full Stage 11 clean/stress tables.

## Baseline Visibility

The batch output includes the required strong baselines:

- `naive_smoothing`
- `ema_smoothing`
- `median_smoothing`
- `matched_smoothing_control`
- `random_trigger`
- `stage9_incumbent`

On the compact fixture batch, full smoothing baselines remain stronger on point MSE:

- `naive_smoothing`: `-1.500901%`
- `ema_smoothing`: `-0.798739%`
- `median_smoothing`: `-1.877721%`
- `boundary_only`: `-0.464343%`

This confirms the Stage 11 limitation: HalluGuard-Dynamics is a dynamics-triggered repair layer, not a universal replacement for smoothing when the only objective is point MSE.

## Readiness Verdict

Stage 12 external batch evaluation passes.

The runner is ready for external forecast directories or manifests as long as every file/group contains both `val` and `test` rows under the documented schema.

## Remaining Limitations

- The fixture set is compact and derived from local Stage 7 predictions; it verifies integration, not new scientific generalization.
- No additional local prediction files were available for ETTm2, ETTh2, Weather, ECL/Electricity, or Traffic. Network expansion was not required to pass Stage 12 and was not used.
- Batch output reports smoothing baselines because smoothing still dominates pure MSE in several settings.
