# Full Baseline Table Run Guide

This guide is the run contract for reproducing HalluGuard tables and for adding
top-conference plug-in baselines on the same forecasting backbones.

## 1. Environment

Preferred:

```bash
bash scripts/setup_env.sh
conda activate halluguard-run
```

Manual conda route:

```bash
conda env create -f environment.yml
conda activate halluguard-run
```

Manual venv route:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the server has CUDA, install the matching PyTorch wheel according to the
official PyTorch instructions. The committed `environment.yml` intentionally
uses a CPU-safe `torch>=2.1` default so it does not pin the wrong CUDA wheel.

## 2. Sanity Checks

```bash
python experiments/halluguard/run_mvp.py \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --quick

python external/halluguard_real_pipeline/export_predictions.py \
  --dataset ETTm1 \
  --model DLinear \
  --horizon 96 \
  --data-root external/ETDataset \
  --output experiments/halluguard/results/smoke_predictions/ETTm1_DLinear_96.jsonl \
  --epochs 1 \
  --max-train-windows 512 \
  --max-eval-windows 64 \
  --batch-size 128 \
  --device auto
```

Then evaluate the smoke prediction file:

```bash
python experiments/halluguard/evaluate_predictions.py \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --input experiments/halluguard/results/smoke_predictions/ETTm1_DLinear_96.jsonl \
  --calibration-split val \
  --split test \
  --output-dir experiments/halluguard/results/smoke_eval/ETTm1_DLinear_96
```

## 3. Regenerate the Original Stage 7 Clean Table

This trains lightweight DLinear/PatchTST forecasters on ETTm1/ETTh1 and exports
validation/test predictions for horizons 96/192/336/720.

```bash
python experiments/halluguard/run_real_table.py \
  --scope stage7 \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --data-root external/ETDataset \
  --epochs 2 \
  --max-train-windows 4096 \
  --max-eval-windows 512 \
  --device auto
```

Primary outputs:

```text
experiments/halluguard/results/stage7_big_table/predictions/*.jsonl
experiments/halluguard/results/stage7_big_table/combined_metrics.csv
experiments/halluguard/results/stage7_big_table/combined_ablation_table.md
experiments/halluguard/results/stage7_big_table/summary.md
```

## 4. Run the Current HalluGuard Router Line

After Stage 7 predictions exist:

```bash
python experiments/halluguard/run_stage13_adaptive_router.py \
  --scope clean_full \
  --config experiments/halluguard/configs/halluguard_stage13_adaptive_router.yaml

python experiments/halluguard/run_stage14_autosearch.py \
  --scope clean_full \
  --config experiments/halluguard/configs/halluguard_stage14_autosearch.yaml
```

For stress tables, use the resume helper because the full 96-config stress table
can exceed terminal timeouts:

```bash
python experiments/halluguard/run_stage14_stress_resume.py \
  --config experiments/halluguard/configs/halluguard_stage14_autosearch.yaml \
  --stress-types boundary_discontinuity

python experiments/halluguard/run_stage14_stress_resume.py \
  --config experiments/halluguard/configs/halluguard_stage14_autosearch.yaml \
  --stress-types trend_drift

python experiments/halluguard/run_stage14_stress_resume.py \
  --config experiments/halluguard/configs/halluguard_stage14_autosearch.yaml \
  --stress-types slope_break

python experiments/halluguard/run_stage14_stress_resume.py \
  --config experiments/halluguard/configs/halluguard_stage14_autosearch.yaml \
  --stress-types delayed_level_shift

python experiments/halluguard/run_stage14_stress_resume.py \
  --config experiments/halluguard/configs/halluguard_stage14_autosearch.yaml \
  --stress-types high_frequency_perturbation

python experiments/halluguard/run_stage14_stress_resume.py \
  --config experiments/halluguard/configs/halluguard_stage14_autosearch.yaml \
  --stress-types variance_shift
```

## 5. Advanced Plug-in Baseline Table

For each top-conference plug-in baseline, train the same backbone/dataset/horizon
and export predictions using the schema in
`experiments/halluguard/EXTERNAL_PREDICTION_SCHEMA.md`.

Required fields:

```text
sample_id, dataset, model, split, context, prediction, target
```

Recommended model labels:

```text
DLinear
DLinear+RevIN
DLinear+DishTS
DLinear+SAN
DLinear+SIN
DLinear+FAN
DLinear+DDN
DLinear+CCM
DLinear+LIFT
PatchTST
PatchTST+RevIN
PatchTST+DishTS
PatchTST+SAN
PatchTST+SIN
PatchTST+FAN
PatchTST+DDN
PatchTST+CCM
PatchTST+LIFT
```

Put all exported baseline prediction files in:

```text
baseline_predictions/
```

Then run:

```bash
python experiments/halluguard/run_stage12_external_batch.py \
  --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml \
  --input-dir baseline_predictions \
  --recursive \
  --continue-on-error \
  --output-dir experiments/halluguard/results/plugin_baseline_external_batch
```

This evaluates every external prediction table with validation-only calibration
and produces aggregate CSV/markdown outputs.

## 6. Required Reporting Fields

For each dataset/model/horizon/plugin row, keep:

- dataset
- backbone
- plug-in module
- horizon
- n validation samples
- n test samples
- MSE
- MAE
- delta vs original backbone
- delta vs HalluGuard router
- validation-only calibration flag
- output prediction path
- blocker reason, if any

Do not tune thresholds, routers, or baseline choices on test targets.
